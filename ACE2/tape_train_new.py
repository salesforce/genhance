import typing
import os
import logging
from timeit import default_timer as timer
import json
from pathlib import Path
import inspect
import pickle as pkl

from transformers import MT5ForConditionalGeneration, MT5Config, MT5EncoderModel
from modeling_progeny import ProgenyForSequenceToSequenceClassification, ProgenyForValuePrediction, ProgenyForSequenceClassification, ProgenyForContactPrediction, ProgenyConfig
from progeny_tokenizer import TAPETokenizer
from tape.optimization import AdamW

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tape.optimization import WarmupLinearSchedule
from tape.datasets import SecondaryStructureDataset, StabilityDataset, RemoteHomologyDataset,  ProteinnetDataset, FluorescenceDataset
from tape.metrics import spearmanr

from torch.utils.data import DataLoader, RandomSampler, Dataset, BatchSampler
from torch.utils.data.distributed import DistributedSampler
from tape.optimization import AdamW

from tape.registry import registry

from tape.utils import get_effective_batch_size
from tape.utils._sampler import BucketBatchSampler


from tape import utils
from tape import errors
from tape import visualization
from tape.registry import registry
from tape.models.modeling_utils import ProteinModel
from tape import ProteinBertForSequenceToSequenceClassification
try:
    from apex import amp
    import amp_C
    import apex_C
    from apex.amp import _amp_state
    from apex.parallel.distributed import flat_dist_call
    from apex.parallel.distributed import DistributedDataParallel as DDP
    APEX_FOUND = True
except ImportError:
    APEX_FOUND = False

import pandas as pd
import numpy as np


# try:
#     from torch.utils.tensorboard import SummaryWriter
# except ImportError:
#     from tensorboardX import SummaryWriter
    
logger = logging.getLogger(__name__)

MetricsDict = typing.Dict[str, float]
LossAndMetrics = typing.Tuple[float, MetricsDict]

OutputDict = typing.Dict[str, typing.Any]

def setup_loader(dataset: Dataset,
                 batch_size: int,
                 local_rank: int,
                 n_gpu: int,
                 gradient_accumulation_steps: int,
                 num_workers: int) -> DataLoader:
    #import pdb; pdb.set_trace()
    sampler = DistributedSampler(dataset) if local_rank != -1 else RandomSampler(dataset)
    batch_size = get_effective_batch_size(
        batch_size, local_rank, n_gpu, gradient_accumulation_steps) * n_gpu
    # WARNING: this will fail if the primary sequence is not the first thing the dataset returns
    batch_sampler = BatchSampler(sampler,batch_size, False)
#    batch_sampler = BucketBatchSampler(
#        sampler, batch_size, False, lambda x: len(x[0]), dataset)

    loader = DataLoader(
        dataset,
        collate_fn=dataset.collate_fn,  # type: ignore
        batch_sampler=batch_sampler)

    return loader

# def setup_loader(dataset: Dataset,
#                  batch_size: int,
#                  local_rank: int,
#                  n_gpu: int,
#                  gradient_accumulation_steps: int,
#                  num_workers: int) -> DataLoader:
#     sampler = DistributedSampler(dataset) if local_rank != -1 else RandomSampler(dataset)
#     batch_size = get_effective_batch_size(
#         batch_size, local_rank, n_gpu, gradient_accumulation_steps) * n_gpu
#     # WARNING: this will fail if the primary sequence is not the first thing the dataset returns
#     batch_sampler = BucketBatchSampler(
#         sampler, batch_size, False, lambda x: len(x[0]), dataset)

#     loader = DataLoader(
#         dataset,
#         num_workers=num_workers,
#         collate_fn=dataset.collate_fn,  # type: ignore
#         batch_sampler=batch_sampler)

#     return loader

def setup_optimizer(model,
                    learning_rate: float, decay_rate: float = 0.01, sgd: bool = False ):
    """Create the AdamW optimizer for the given model with the specified learning rate. Based on
    creation in the pytorch_transformers repository.
    Args:
        model (PreTrainedModel): The model for which to create an optimizer
        learning_rate (float): Default learning rate to use when creating the optimizer
    Returns:
        optimizer (AdamW): An AdamW optimizer
    """
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": decay_rate,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    if sgd:
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=learning_rate)
    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    #optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=learning_rate)
    return optimizer


class ForwardRunner:

    def __init__(self,
                 model: ProteinModel,
                 device: torch.device = torch.device('cuda:0'),
                 n_gpu: int = 1,
                 fp16: bool = False,
                 local_rank: int = -1,
                  model_parallel: bool = False):

        self.model = model
        self.model_parallel = model_parallel
        self.device = device
        self.n_gpu = n_gpu
        self.fp16 = fp16
        self.local_rank = local_rank

        forward_arg_keys = inspect.getfullargspec(model.forward).args
        forward_arg_keys = forward_arg_keys[1:]  # remove self argument
        self._forward_arg_keys = forward_arg_keys
        assert 'input_ids' in self._forward_arg_keys

    def initialize_distributed_model(self):
        if self.local_rank != -1:
            if not self.fp16:
                self.model = DDP(self.model)
            else:
                flat_dist_call([param.data for param in self.model.parameters()],
                               torch.distributed.broadcast, (0,))
        elif self.n_gpu > 1:
            self.model = nn.DataParallel(self.model)

    def forward(self,
                batch: typing.Dict[str, torch.Tensor],
                return_outputs: bool = False,
                no_loss: bool = False,
                max_len: int = -1):
        # Filter out batch items that aren't used in this model
        # Requires that dataset keys match the forward args of the model
        # Useful if some elements of the data are only used by certain models
        # e.g. PSSMs / MSAs and other evolutionary data

        # print("A batch: ", batch)
        batch = {name: tensor for name, tensor in batch.items()
                 if name in self._forward_arg_keys}

        # print("B batch: ", batch)

        if max_len>0:
            if batch["input_ids"].shape[1]>max_len:

                 batch["input_ids"] =  batch["input_ids"][:,:max_len]
                 batch["input_mask"] =  batch["input_mask"][:,:max_len]

                 if len( batch["targets"].shape)>2:
                     batch["targets"] =  batch["targets"][:,1:max_len-1,1:max_len-1]
                 if len( batch["targets"].shape)==2:

                     batch["targets"] =  batch["targets"][:,:max_len]



                 if 'protein_length' in batch:
                     batch['protein_length'] = torch.clamp(batch['protein_length'],max=max_len)

#        import pdb; pdb.set_trace()

        # if self.model.train:
        #     mask = torch.rand(batch["input_ids"].shape)>.1
        #     batch["input_ids"] = mask*batch["input_ids"] +(~mask)

        if self.device.type == 'cuda':
            batch = {name: tensor.cuda(device=self.device, non_blocking=True)
                     for name, tensor in batch.items()}


        if self.model_parallel:
            batch["targets"] = batch["targets"].to("cuda:"+str(torch.cuda.device_count()-1))


        #import pdb; pdb.set_trace()

        outputs = self.model(**batch)
        # print("outputs: ", outputs)

        if no_loss:
            return outputs

        if isinstance(outputs[0], tuple):
            # model also returned metrics
            loss, metrics = outputs[0]
        else:
            # no metrics
            loss = outputs[0]
            metrics = {}

        if self.n_gpu > 1:  # pytorch DataDistributed doesn't mean scalars
            loss = loss.mean()
            metrics = {name: metric.mean() for name, metric in metrics.items()}

        if return_outputs:
            return loss, metrics, outputs
        else:
            return loss, metrics

    def train(self):
        self.model.train()
        self.model.bert.encoder.train()
#        self.model.classify.train()
        return self

    def eval(self):
        self.model.eval()
        self.model.bert.encoder.eval()
#        self.model.classify.eval()
        return self


class BackwardRunner(ForwardRunner):

    def __init__(self,
                 model: ProteinModel,
                 optimizer: optim.Optimizer,  # type: ignore
                 gradient_accumulation_steps: int = 1,
                 device: torch.device = torch.device('cuda:0'),
                 n_gpu: int = 1,
                 fp16: bool = False,
                 local_rank: int = -1,
                 max_grad_norm: float = 1.0,
                 warmup_steps: int = 0,
                 num_train_optimization_steps: int = 1000000,
                 model_parallel: bool = False):

        super().__init__(model, device, n_gpu, fp16, local_rank, model_parallel)
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self._global_step = 0
        self._local_rank = local_rank
        self._overflow_buf = torch.cuda.IntTensor([0])  # type: ignore
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self._delay_accumulation = fp16 and local_rank != -1

        self.scheduler = WarmupLinearSchedule(
            self.optimizer, warmup_steps, num_train_optimization_steps)
#        self.scheduler=None

        #self.scheduler =torch.optim.lr_scheduler.CyclicLR(optimizer, 2e-3 , 2e-5, step_size_up=500,cycle_momentum=False)

    def initialize_fp16(self):
        if self.fp16:
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level="O2", loss_scale="dynamic",
                master_weights=True)
            _amp_state.loss_scalers[0]._loss_scale = 2 ** 20

    def resume_from_checkpoint(self, checkpoint_dir: str) -> int:
        checkpoint = torch.load(
            os.path.join(checkpoint_dir, 'checkpoint.bin'), map_location=self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.fp16:
            self.optimizer._lazy_init_maybe_master_weights()
            self.optimizer._amp_stash.lazy_init_called = True
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            for param, saved in zip(
                    amp.master_params(self.optimizer), checkpoint['master params']):
                param.data.copy_(saved.data)
            amp.load_state_dict(checkpoint['amp'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        return start_epoch

    def save_state(self, save_directory: typing.Union[str, Path], epoch_id: int):
        save_directory = Path(save_directory)
        if not save_directory.exists():
            save_directory.mkdir()
        else:
            assert save_directory.is_dir(), "Save path should be a directory"
        model_to_save = getattr(self.model, 'module', self.model)
        model_to_save.save_pretrained(save_directory)
        optimizer_state: typing.Dict[str, typing.Any] = {
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': epoch_id}
        if APEX_FOUND:
            optimizer_state['master params'] = list(amp.master_params(self.optimizer))
            try:
                optimizer_state['amp'] = amp.state_dict()
            except AttributeError:
                pass
        torch.save(optimizer_state, save_directory / 'checkpoint.bin')

    def backward(self, loss) -> None:
        if not self._delay_accumulation:
            loss = loss / self.gradient_accumulation_steps
        if self.fp16:
            with amp.scale_loss(loss, self.optimizer,
                                delay_overflow_check=self._delay_accumulation) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

    def step(self) -> None:
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        if self._local_rank == -1:
            self._step()
        elif not self.fp16:
            # TODO: Can you do this allreduce after accumulation also?
            self._step()
        else:
            self._step_distributed_fp16()
        # if self.optimizer.prior_decay>0.0:
        #
        #     factor = self.scheduler.get_lr()[0]/self.scheduler.base_lrs[0]
        #     for param in self.model.bert.encoder.parameters():
        #         param.data = param.data + (self.optimizer.prior_decay)*(param.data0-param.data)



    def _step(self) -> None:
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()  # type: ignore
        self._global_step += 1

    def _step_distributed_fp16(self) -> None:
        # manually allreduce gradients after all accumulation steps
        # check for Inf/NaN
        # 1. allocate an uninitialized buffer for flattened gradient
        scaler = _amp_state.loss_scalers[0]
        master_grads = [p.grad for p in amp.master_params(self.optimizer) if p.grad is not None]
        flat_grad_size = sum(p.numel() for p in master_grads)
        # allreduce_dtype = torch.float16 if args.allreduce_post_accumulation_fp16 else \
            # torch.float32
        allreduce_dtype = torch.float16
        flat_raw = torch.empty(flat_grad_size, device='cuda', dtype=allreduce_dtype)
        # 2. combine unflattening and predivision of unscaled 'raw' gradient
        allreduced_views = apex_C.unflatten(flat_raw, master_grads)
        self._overflow_buf.zero_()
        amp_C.multi_tensor_scale(
            65536,
            self._overflow_buf,
            [master_grads, allreduced_views],
            scaler.loss_scale() / (
                torch.distributed.get_world_size() * self.gradient_accumulation_steps))
        # 3. sum gradient across ranks. Because of the predivision, this averages the gradient
        torch.distributed.all_reduce(flat_raw)
        # 4. combine unscaling and unflattening of allreduced gradient
        self._overflow_buf.zero_()
        amp_C.multi_tensor_scale(
            65536,
            self._overflow_buf,
            [allreduced_views, master_grads],
            1. / scaler.loss_scale())
        # 5. update loss scale
        scaler = _amp_state.loss_scalers[0]
        old_overflow_buf = scaler._overflow_buf
        scaler._overflow_buf = self._overflow_buf
        had_overflow = scaler.update_scale()
        scaler._overfloat_buf = old_overflow_buf
        # 6. call optimizer step function
        if had_overflow == 0:
            self._step()
        else:
            # Overflow detected, print message and clear gradients
            logger.info(f"Gradient overflow.  Skipping step, reducing loss scale to "
                        f"{scaler.loss_scale()}")
            if _amp_state.opt_properties.master_weights:
                for param in self.optimizer._amp_stash.all_fp32_from_fp16_params:
                    param.grad = None
        for param in self.model.parameters():
            param.grad = None

    @property
    def global_step(self) -> int:
        return self._global_step


def run_train_epoch(epoch_id: int,
                    train_loader: DataLoader,
                    runner: BackwardRunner,
                    viz: typing.Optional[visualization.TAPEVisualizer] = None,
                    num_log_iter: int = 20,
                    gradient_accumulation_steps: int = 1,
                    average: bool = False) -> LossAndMetrics:
    if viz is None:
        viz = visualization.DummyVisualizer()
    smoothing = 1 - 1 / num_log_iter
    accumulator = utils.MetricsAccumulator(smoothing)

    torch.set_grad_enabled(True)
    runner.train()

    def make_log_str(step: int, time: float) -> str:
        ep_percent = epoch_id + step / len(train_loader)
        if runner.scheduler is not None:
            curr_lr = runner.scheduler.get_lr()[0]  # type: ignore
        else:
            curr_lr = runner.optimizer.param_groups[0]['lr']

        print_str = []
        print_str.append(f"[Ep: {ep_percent:.2f}]")
        print_str.append(f"[Iter: {runner.global_step}]")
        print_str.append(f"[Time: {time:5.2f}s]")
        print_str.append(f"[Loss: {accumulator.loss():.5g}]")

        for name, value in accumulator.metrics().items():
            print_str.append(f"[{name.capitalize()}: {value:.5g}]")

        print_str.append(f"[LR: {curr_lr:.5g}]")
        return ''.join(print_str)

    start_t = timer()
    for step, batch in enumerate(train_loader):
        # print("batch: ", batch)
        # print("batch['input_ids'][0]: ", batch['input_ids'][0])
        # print("batch['input_ids'].shape: ", batch['input_ids'].shape)
        loss, metrics = runner.forward(batch, max_len=512)  # type: ignore
        runner.backward(loss)
        accumulator.update(loss, metrics, step=False)
        if (step + 1) % gradient_accumulation_steps == 0:
            runner.step()
            if average:
                for param in runner.model.parameters():
                    param.sum_data += param.data
                    param.count+=1
            viz.log_metrics(accumulator.step(), "train", runner.global_step)
            if runner.global_step % num_log_iter == 0:
                end_t = timer()
                logger.info(make_log_str(step, end_t - start_t))
                start_t = end_t

    final_print_str = f"Train: [Loss: {accumulator.final_loss():.5g}]"
    for name, value in accumulator.final_metrics().items():
        final_print_str += f"[{name.capitalize()}: {value:.5g}]"
    logger.info(final_print_str)
    return accumulator.final_loss(), accumulator.final_metrics()


def run_valid_epoch(epoch_id: int,
                    valid_loader: DataLoader,
                    runner: ForwardRunner,
                    viz: typing.Optional[visualization.TAPEVisualizer] = None,
                    is_master: bool = True) -> typing.Tuple[float, typing.Dict[str, float]]:
    num_batches = len(valid_loader)
    accumulator = utils.MetricsAccumulator()

    torch.set_grad_enabled(False)
    runner.eval()

    for batch in tqdm(valid_loader, desc='Running Eval', total=num_batches,
                      disable=not is_master, leave=False):
        loss, metrics = runner.forward(batch)  # type: ignore
        accumulator.update(loss, metrics)

    # Reduce loss across all processes if multiprocessing
    eval_loss = utils.reduce_scalar(accumulator.final_loss())
    metrics = {name: utils.reduce_scalar(value)
               for name, value in accumulator.final_metrics().items()}

    print_str = f"Evaluation: [Loss: {eval_loss:.5g}]"
    for name, value in metrics.items():
        print_str += f"[{name.capitalize()}: {value:.5g}]"

    metrics['loss'] = eval_loss
    if viz is not None:
        viz.log_metrics(metrics, "val", getattr(runner, 'global_step', epoch_id))

    logger.info(print_str)

    return eval_loss, metrics


def _get_outputs_to_save(batch, outputs):
    targets = batch['targets'].cpu().numpy()
    outputs = outputs.cpu().numpy()
    protein_length = batch['protein_length'].sum(1).cpu().numpy()

    reshaped_output = []
    for target, output, plength in zip(targets, outputs, protein_length):
        output_slices = tuple(slice(1, plength - 1) if dim == protein_length.max() else
                              slice(0, dim) for dim in output.shape)
        output = output[output_slices]
        target = target[output_slices]

        reshaped_output.append((target, output))
    reshaped_output


def run_eval_epoch(eval_loader: DataLoader,
                   runner: ForwardRunner,
                   is_master: bool = True) -> typing.List[typing.Dict[str, typing.Any]]:
    torch.set_grad_enabled(False)
    runner.eval()

    save_outputs = []
    preds=[]
    targs = []

    for batch in tqdm(eval_loader, desc='Evaluation', total=len(eval_loader),
                      disable=not is_master):
        loss, metrics, outputs = runner.forward(batch, return_outputs=True)  # type: ignore
        predictions = outputs[1].cpu().numpy()
        targets = batch['targets'].cpu().numpy()
        for pred, target in zip(predictions, targets):
            save_outputs.append({'prediction': pred, 'target': target})
            preds.append(pred)
            targs.append(target)

    return save_outputs, preds, targs

# NEW custom data pipeline - start

class PKLDFDataset(Dataset):
    """Creates a dataset from an pkl df file.
    Args:
        data_file (typing.Union[str, Path]): Path to pkl df file.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """

    def __init__(self,
                data_file: typing.Union[str, Path],
                in_memory: bool = False,
                split: str = 'train',
                train_ratio: float = 1,
                normalize_targets: bool = False,
                train_data_file: str = '250K_ddG_split/train_ddG.pkl',
                ):
        print("PKLDFDataset split: ", split)
        print("PKLDFDataset train_ratio: ", train_ratio)
        data_file = Path(data_file)
        print("PKLDFDataset data_file: ", data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)
        
        df = pd.read_pickle(data_file)
        
        if train_ratio != 1:
            shuffled_df = df.sort_index()
            train_num_samples = int(len(shuffled_df) * train_ratio)
#             valid_num_samples = len(shuffled_df) - train_num_samples
            if split == 'train':
                final_df = shuffled_df.iloc[:train_num_samples]
            elif split == 'valid':
                final_df = shuffled_df.iloc[train_num_samples:]
            else:
                final_df = df
        else:
            final_df = df
        
        self.df = final_df
        num_examples = len(final_df)
        self._num_examples = num_examples
        
        if in_memory:
            cache = [None] * num_examples
            self._cache = cache
            
        self._in_memory = in_memory

        # TODO: normalize target values with train stats
        self.normalize_targets = normalize_targets
        # print("self.normalize_targets: ", self.normalize_targets)
        if self.normalize_targets:
            if split not in ['train', 'valid']:
                df = pd.read_pickle(train_data_file)

            self.targets_mean = np.mean(df['ddG'])
            self.targets_std = np.std(df['ddG'])
            print("self.targets_mean: ", self.targets_mean)
            print("self.targets_std: ", self.targets_std)
        
    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        if self._in_memory and self._cache[index] is not None:
            item = self._cache[index]
        else:
            row = self.df.iloc[index]
            item = {}
            item['primary'] = row['MT_seq']
            
            if self.normalize_targets:
                item['stability_score'] = (row['ddG'] - self.targets_mean) / self.targets_std
            else:
                item['stability_score'] = row['ddG']

            item['id'] = str(index)
            if self._in_memory:
                self._cache[index] = item
            
        return item

def pad_sequences(sequences: typing.Sequence, constant_value=0, dtype=None) -> np.ndarray:
    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()

    if dtype is None:
        dtype = sequences[0].dtype

    if isinstance(sequences[0], np.ndarray):
        array = np.full(shape, constant_value, dtype=dtype)
    elif isinstance(sequences[0], torch.Tensor):
        array = torch.full(shape, constant_value, dtype=dtype)

    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array

def dataset_factory(data_file: typing.Union[str, Path], *args, **kwargs) -> Dataset:
    data_file = Path(data_file)
    if not data_file.exists():
        raise FileNotFoundError(data_file)
        
    if data_file.suffix == '.pkl':
        return PKLDFDataset(data_file, *args, **kwargs) #!
        
    if data_file.suffix == '.lmdb':
        return LMDBDataset(data_file, *args, **kwargs)
    elif data_file.suffix in {'.fasta', '.fna', '.ffn', '.faa', '.frn'}:
        return FastaDataset(data_file, *args, **kwargs)
    elif data_file.suffix == '.json':
        return JSONDataset(data_file, *args, **kwargs)
    elif data_file.is_dir():
        return NPZDataset(data_file, *args, **kwargs)
    else:
        raise ValueError(f"Unrecognized datafile type {data_file.suffix}")

class CustomStabilityDataset(Dataset):

    def __init__(self,
                data_path: typing.Union[str, Path],
                split: str,
                tokenizer: typing.Union[str, TAPETokenizer] = 'iupac',
                in_memory: bool = False,
                train_ratio: float = 1,
                normalize_targets: bool = False):

        if split not in ('train', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. "
                            f"Must be one of ['train', 'valid', 'test']")
        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        if split == 'valid':
            file_prefix = 'train'
        else:
            file_prefix = split
            
        data_path = Path(data_path)
        data_file = f'{file_prefix}_ddG.pkl' #!

        self.data = dataset_factory(data_path / data_file, in_memory, split, train_ratio, normalize_targets=normalize_targets) #! for train/valid split

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        token_ids = self.tokenizer.encode(item['primary']) #!
        input_mask = np.ones_like(token_ids)
        return token_ids, input_mask, float(item['stability_score']) #!

    def collate_fn(self, batch: typing.List[typing.Tuple[typing.Any, ...]]) -> typing.Dict[str, torch.Tensor]:
        input_ids, input_mask, stability_true_value = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        stability_true_value = torch.FloatTensor(stability_true_value)  # type: ignore
        stability_true_value = stability_true_value.unsqueeze(1)

        return {'input_ids': input_ids,
                'input_mask': input_mask,
                'targets': stability_true_value}
          
# NEW custom data pipeline - end

def run_train(model_type: str,
              task: str,
              learning_rate: float = 1e-4,
              batch_size: int = 4,
              num_train_epochs: int = 10,
              num_log_iter: int = 20,
              test_name: typing.Optional[str] = "test",
              fp16: bool = False,
              warmup_steps: int = 10000,
              gradient_accumulation_steps: int = 1,
              loss_scale: int = 0,
              dropout: float = 0.1,
              prior_decay: float = 0.0,
              weight_decay: float = 0.0,
              max_grad_norm: float = 1.0,
              exp_name: typing.Optional[str] = None,
              from_pretrained: typing.Optional[str] = None,
              log_dir: str = './logs',
              eval_freq: int = 1,
              save_freq: typing.Union[int, str] = 1,
              model_config_file: typing.Optional[str] = None,
              data_dir: str = './250K_ddG_split',
              output_dir: str = './results',
              no_cuda: bool = False,
              seed: int = 42,
              local_rank: int = -1,
              tokenizer: str = 'iupac',
              num_workers: int = 8,
              init_model: str="/export/share/bkrause/progen/progeny/t5_large_hf/",
              debug: bool = False,
              sgd: bool = False,
              average: bool = False,
              log_level: typing.Union[str, int] = logging.INFO,
              patience: int = -1,
              model_parallel: bool = False,
              resume_from_checkpoint: bool = False,
              normalize_targets: bool = False,
              predict_head: str = 'contrastive',
              ) -> None:

    # SETUP AND LOGGING CODE #
    input_args = locals()
    device, n_gpu, is_master = utils.setup_distributed(
        local_rank, no_cuda)
    n_gpu=1
    device = torch.device('cuda:0')


    # # TB code 
    # if local_rank in [-1, 0]:
    #     tb_writer = SummaryWriter(os.path.join(args.output_dir, 'train_spliceai_log'))






    exp_dir = utils.get_expname(exp_name, task, model_type)
    save_path = Path(output_dir) / exp_dir

    if is_master:
        # save all the hidden parameters.
        save_path.mkdir(parents=True, exist_ok=True)
        with (save_path / 'args.json').open('w') as f:
            json.dump(input_args, f)

    utils.barrier_if_distributed()
    utils.setup_logging(local_rank, save_path, log_level)
    utils.set_random_seeds(seed, n_gpu)

    tokenizer = TAPETokenizer(vocab="progeny")



    dataset_dict = dict()
    dataset_dict["secondary_structure"] = SecondaryStructureDataset
    dataset_dict["remote_homology"] = RemoteHomologyDataset
    dataset_dict["fluorescence"] = FluorescenceDataset
    dataset_dict["stability"] = StabilityDataset
    dataset_dict["contact_prediction"] = ProteinnetDataset

    model_dict = dict()
    model_dict["secondary_structure"] = ProgenyForSequenceToSequenceClassification
    model_dict["remote_homology"] = ProgenyForSequenceClassification
    model_dict["fluorescence"] = ProgenyForValuePrediction
    model_dict["stability"] = ProgenyForValuePrediction
    model_dict["contact_prediction"] = ProgenyForContactPrediction

    test_set_dict = dict()
    test_set_dict["secondary_structure"] = "cb513"
    test_set_dict["remote_homology"] = 'test_fold_holdout'
    test_set_dict["fluorescence"] = "test"
    test_set_dict["stability"] = "test"
    test_set_dict["contact_prediction"] = ""


    dataset_type = dataset_dict[task]
    model_type = model_dict[task]

# TODO EDIT!! - start: data loading
    tokenizer = TAPETokenizer(vocab="progeny")
    # print("dataset_type: ", dataset_type)
    train_dataset = CustomStabilityDataset(data_dir,'train',tokenizer=tokenizer, train_ratio=0.9, normalize_targets=normalize_targets)

    valid_dataset = CustomStabilityDataset(data_dir,'valid',tokenizer=tokenizer, train_ratio=0.9, normalize_targets=normalize_targets)

    train_loader = utils.setup_loader(
        train_dataset, batch_size, local_rank, n_gpu,
        gradient_accumulation_steps, 1)
    valid_loader = utils.setup_loader(
        valid_dataset, batch_size, local_rank, n_gpu,
        gradient_accumulation_steps, 1)


    # print("dataset_type: ", dataset_type)
    # train_dataset = dataset_type("tape_data",'train',tokenizer=tokenizer)

    # valid_dataset = dataset_type("tape_data",'valid',tokenizer=tokenizer)

    # train_loader = utils.setup_loader(
    #     train_dataset, batch_size, local_rank, n_gpu,
    #     gradient_accumulation_steps, 1)
    # valid_loader = utils.setup_loader(
    #     valid_dataset, batch_size, local_rank, n_gpu,
    #     gradient_accumulation_steps, 1)

# TODO EDIT!! - end


    if not test_name == "" and test_name is not None:
        # test_dataset = dataset_type("tape_data", test_name,tokenizer=tokenizer)
        test_dataset = CustomStabilityDataset(data_dir, test_name,tokenizer=tokenizer, normalize_targets=normalize_targets)
        test_loader = utils.setup_loader(
            test_dataset, batch_size, local_rank, n_gpu,
            gradient_accumulation_steps, num_workers)
    else:
        test_name=None
    num_train_optimization_steps = utils.get_num_train_optimization_steps(
        train_dataset, batch_size, num_train_epochs)


    #init_model="/export/share/bkrause/progen/progeny/t5_uniref_large_hf/"

    #init_model="/export/share/bkrause/progen/progeny/t5_base_uniref_bfd50"
#    init_model="/export/share/bkrause/progen/progeny/t5_large_hf/"
#    init_model = "/export/share/bkrause/progen/progeny/t5_11b_bfd"
#    init_model = "/export/share/bkrause/progen/progeny/t5_early_11b_bfd"

    t5config = MT5Config.from_pretrained(init_model)

    t5config.dropout_rate = dropout


    config = ProgenyConfig.from_pretrained(init_model)
    if task=="secondary_structure":
        config.num_labels=3
    if task=="remote_homology":
        config.num_labels=1195
    config.dropout_rate = dropout

    model=model_type.from_pretrained(init_model,config=config,t5config=t5config, predict_head=predict_head)
    #model=model_type(config=config,t5config=t5config)

    print("model: ", model)

    reinit_layers=0

    reinit_decoder = False
    if reinit_layers>0:
        model0=model_type(config=config,t5config=t5config)
        for j in range(1,reinit_layers+1):

            layer = model.bert.encoder.block[-j]
            layer0 = model.bert.encoder.block[-j]

            params0 = dict()
            for name,param in model0.bert.encoder.block[-j].named_parameters():
                params0[name] = param.data
            for name,param in model.bert.encoder.block[-j].named_parameters():
                if name in params0:
                    param.data = 1*params0[name]
    if reinit_decoder:
        model0=model_type(config=config,t5config=t5config)
        params0 = dict()
        for name,param in model0.bert.decoder.named_parameters():
            params0[name] = param.data
        for name,param in model.bert.decoder.named_parameters():
                if name in params0:
                    param.data = 1*params0[name]
    #model.predict.value_prediction.weight.data = 0*model.predict.value_prediction.weight.data

        #import pdb; pdb.set_trace()

        #model = ProgenyForSequenceToSequenceClassification(config, t5config)

    #    model = ProteinBertForSequenceToSequenceClassification.from_pretrained('bert-base',num_labels=3)


    #model = registry.get_task_model(model_type, task, model_config_file, from_pretrained)
    if model_parallel:

        model.bert.encoder.parallelize()
        device_id = "cuda:"+str(torch.cuda.device_count()-1)
        if task== "fluorescence" or task == "stability":
            model.pooler.to(device_id)
            model.predict.to(device_id)
        else:

            model.classify.to(device_id)
    else:
        model = model.to(device)
    optimizer = setup_optimizer(model, learning_rate, decay_rate = weight_decay, sgd=sgd)
    if prior_decay > 0:
        for param in model.bert.encoder.parameters():
            param.data0=1*param.data
    optimizer.prior_decay=prior_decay

    viz = visualization.get(log_dir, exp_dir, local_rank, debug=debug)
    viz.log_config(input_args)
    viz.log_config(model.config.to_dict())
    viz.watch(model)

    logger.info(
        f"device: {device} "
        f"n_gpu: {n_gpu}, "
        f"distributed_training: {local_rank != -1}, "
        f"16-bits training: {fp16}")

    runner = BackwardRunner(
        model, optimizer, gradient_accumulation_steps, device, n_gpu,
        fp16, local_rank, max_grad_norm, warmup_steps, num_train_optimization_steps,model_parallel=model_parallel)

    runner.initialize_fp16()
    if resume_from_checkpoint:
        assert from_pretrained is not None
        start_epoch = runner.resume_from_checkpoint(from_pretrained)
    else:
        start_epoch = 0
    runner.initialize_distributed_model()

    num_train_optimization_steps = utils.get_num_train_optimization_steps(
        train_dataset, batch_size, num_train_epochs)
    is_master = local_rank in (-1, 0)

    if isinstance(save_freq, str) and save_freq != 'improvement':
        raise ValueError(
            f"Only recongized string value for save_freq is 'improvement'"
            f", received: {save_freq}")

    if save_freq == 'improvement' and eval_freq <= 0:
        raise ValueError("Cannot set save_freq to 'improvement' and eval_freq < 0")

    num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Batch size = %d", batch_size)
    logger.info("  Num epochs = %d", num_train_epochs)
    logger.info("  Num train steps = %d", num_train_optimization_steps)
    logger.info("  Num parameters = %d", num_trainable_parameters)

    best_val_loss = float('inf')
    num_evals_no_improvement = 0


    # TODO Save model weights after training
    def do_save(epoch_id: int, num_evals_no_improvement: int) -> bool:
        # if not is_master:
        #     return False
        # if isinstance(save_freq, int):
        #     return ((epoch_id + 1) % save_freq == 0) or ((epoch_id + 1) == num_train_epochs)
        # else:
        return False #num_evals_no_improvement == 0

    utils.barrier_if_distributed()

    # ACTUAL TRAIN/EVAL LOOP #
    with utils.wrap_cuda_oom_error(local_rank, batch_size, n_gpu, gradient_accumulation_steps):
        for epoch_id in range(start_epoch, num_train_epochs):

            trigger = 5

            if average and epoch_id >= trigger:
                if epoch_id==trigger:
                    for param in runner.model.parameters():
                        param.sum_data = 1*param.data
                        param.count=0
                else:
                    for param in runner.model.parameters():
                        param.data =1*param.data0

                run_train_epoch(epoch_id, train_loader, runner,
                                viz, num_log_iter, gradient_accumulation_steps,True)
                for param in runner.model.parameters():
                    param.data0 = 1*param.data
                    param.data = param.sum_data/param.count
            else:

                run_train_epoch(epoch_id, train_loader, runner,
                                viz, num_log_iter, gradient_accumulation_steps)


            if eval_freq > 0 and (epoch_id + 1) % eval_freq == 0:
                if task == "fluorescence" or task=="stability":
                    eval_data, preds, targs = run_eval_epoch(valid_loader,runner, is_master)
                    val_loss = -1*spearmanr(targs,preds)
                    print("Valid spearman = " +str(-1*val_loss) )

                else:
                    val_loss, _ = run_valid_epoch(epoch_id, valid_loader, runner, viz, is_master)

                if test_name is not None:
                    if task == "fluorescence" or task=="stability":
                        if True:
                            eval_data, preds, targs = run_eval_epoch(test_loader,runner, is_master)
                            test_loss = -1*spearmanr(targs,preds)
                            print("Test spearman = " +str(-1*test_loss) )

                    else:
                        test_loss, _ = run_valid_epoch(epoch_id, test_loader, runner, viz, is_master)


                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    num_evals_no_improvement = 0
                    
                    if test_name is not None:
                        best_test_loss=test_loss


                else:
                    num_evals_no_improvement += 1

            # Save trained model
            if do_save(epoch_id, num_evals_no_improvement):
                logger.info("** ** * Saving trained model ** ** * ")
                # Only save the model itself
                runner.save_state(save_path, epoch_id)
                logger.info(f"Saving model checkpoint to {save_path}")

            utils.barrier_if_distributed()
            if patience > 0 and num_evals_no_improvement >= patience:
                logger.info(f"Finished training at epoch {epoch_id} because no "
                            f"improvement for {num_evals_no_improvement} epochs.")
                logger.log(35, f"Best Val Loss: {best_val_loss}")
                logger.log(35, f"Best test Loss: {best_test_loss}")
                if local_rank != -1:
                    # If you're distributed, raise this error. It sends a signal to
                    # the master process which lets it kill other processes and terminate
                    # without actually reporting an error. See utils/distributed_utils.py
                    # for the signal handling code.
                    raise errors.EarlyStopping
                else:
                    break
    logger.info(f"Finished training after {num_train_epochs} epochs.")
                
    if task == "fluorescence" or task=="stability":
        eval_data, preds, targs = run_eval_epoch(valid_loader,runner, is_master)
        val_loss = -1*spearmanr(targs,preds)
        print("Valid spearman = " +str(-1*val_loss) )

    else:
        val_loss, _ = run_valid_epoch(epoch_id, valid_loader, runner, viz, is_master)

    if test_name is not None:
        if task == "fluorescence" or task=="stability":
            if True:
                eval_data, preds, targs = run_eval_epoch(test_loader,runner, is_master)
                test_loss = -1*spearmanr(targs,preds)
                print("Test spearman = " +str(-1*test_loss) )

        else:
            test_loss, _ = run_valid_epoch(epoch_id, test_loader, runner, viz, is_master)

    logger.log(35, f"Final Val Loss: {val_loss}")
    if test_name is not None:
        logger.log(35, f"Final test Loss: {test_loss}")

    # save model weights after training
    runner.save_state(save_path, epoch_id)
    logger.info(f"Saving model checkpoint to {save_path}")
    if best_val_loss != float('inf'):
        logger.log(35, f"Best Val Loss: {best_val_loss}")
        if test_name is not None:
            logger.log(35, f"Best test Loss: {best_test_loss}")




def run_eval(model_type: str,
             task: str,
             pretrained_models: typing.Tuple[str, ...],
             split: str = 'test',
             batch_size: int = 1024,
             model_config_file: typing.Optional[str] = None,
             data_dir: str = './250K_ddG_split',
             no_cuda: bool = False,
             seed: int = 42,
             tokenizer: str = 'iupac',
             num_workers: int = 1,
             debug: bool = False,
             model_parallel: bool = False,
             metrics: typing.Tuple[str, ...] = (),
             log_level: typing.Union[str, int] = logging.INFO) -> typing.Dict[str, float]:

    local_rank = -1  # TAPE does not support torch.distributed.launch for evaluation
    device, n_gpu, is_master = utils.setup_distributed(local_rank, no_cuda)
    n_gpu=1
    device = torch.device('cuda:0')
    utils.setup_logging(local_rank, save_path=None, log_level=log_level)
    utils.set_random_seeds(seed, n_gpu)

    tokenizer = TAPETokenizer(vocab="progeny")



    dataset_dict = dict()
    dataset_dict["secondary_structure"] = SecondaryStructureDataset
    dataset_dict["remote_homology"] = RemoteHomologyDataset
    dataset_dict["fluorescence"] = FluorescenceDataset
    dataset_dict["stability"] = StabilityDataset
    dataset_dict["contact_prediction"] = ProteinnetDataset

    model_dict = dict()
    model_dict["secondary_structure"] = ProgenyForSequenceToSequenceClassification
    model_dict["remote_homology"] = ProgenyForSequenceClassification
    model_dict["fluorescence"] = ProgenyForValuePrediction
    model_dict["stability"] = ProgenyForValuePrediction
    model_dict["contact_prediction"] = ProgenyForContactPrediction

    test_set_dict = dict()
    test_set_dict["secondary_structure"] = "cb513"
    test_set_dict["remote_homology"] = 'test_fold_holdout'
    test_set_dict["fluorescence"] = "test"
    test_set_dict["stability"] = "test"
    test_set_dict["contact_prediction"] = ""


    dataset_type = dataset_dict[task]
    model_type = model_dict[task]
    test_name = test_set_dict[task]

    # train_dataset = dataset_type("tape_data",'train',tokenizer=tokenizer)

    # valid_dataset = dataset_type("tape_data",'valid',tokenizer=tokenizer)

    train_dataset = CustomStabilityDataset(data_dir,'train',tokenizer=tokenizer, train_ratio=0.9, normalize_targets=normalize_targets)

    valid_dataset = CustomStabilityDataset(data_dir,'valid',tokenizer=tokenizer, train_ratio=0.9, normalize_targets=normalize_targets)

    valid_loader = setup_loader(
        valid_dataset, batch_size, local_rank, n_gpu, 1, 1)
    if not test_name == "":
        # test_dataset = dataset_type("tape_data", test_name,tokenizer=tokenizer)
        test_dataset = CustomStabilityDataset(data_dir, test_name,tokenizer=tokenizer, normalize_targets=normalize_targets)
        test_loader = setup_loader(
            test_dataset, batch_size, local_rank, n_gpu,1,1)
    else:
        test_name=None


#    init_model="/export/share/bkrause/progen/progeny/t5_large_hf/"

    #t5config = MT5Config.from_pretrained("/export/share/bkrause/progen/progeny/t5_large_hf/")
    t5config = MT5Config.from_pretrained("/export/share/bkrause/progen/progeny/t5_11b_bfd")

    prediction = None
    targets = None

    for i in range(0,len(pretrained_models)):
        utils.set_random_seeds(seed, n_gpu)
        from_pretrained = pretrained_models[i]

        config = ProgenyConfig.from_pretrained(from_pretrained)
        #model_config = model_type.from_pretrained(from_pretrained)
        model=model_type.from_pretrained(from_pretrained,config=config,t5config=t5config)
        if model_parallel:
            #
            model.bert.encoder.parallelize()
            device_id = "cuda:"+str(torch.cuda.device_count()-1)

        #    model.pooler.to(device_id)
        #    model.predict.to(device_id)

            model.classify.to(device_id)
        else:
            model = model.to(device)
        runner = ForwardRunner(model, device, n_gpu, model_parallel=model_parallel)
        runner.initialize_distributed_model()
        # valid_dataset = utils.setup_dataset(task, data_dir, split, tokenizer)
        # valid_loader = utils.setup_loader(
        #     valid_dataset, batch_size, local_rank, n_gpu,
        #     1, num_workers)

        metric_functions = [registry.get_metric(name) for name in metrics]
        save_outputs, predictions_i, target = run_eval_epoch(valid_loader, runner, is_master)
        metrics_to_save = {name: metric(target, predictions_i)
                           for name, metric in zip(metrics, metric_functions)}
        print(metrics_to_save)

        if prediction is None:

            prediction = [x/len(pretrained_models) for x in predictions_i]
        else:
            for j in range(0,len(predictions_i)):
                prediction[j] += predictions_i[j]/len(pretrained_models)

#    import pdb; pdb.set_trace()
    #target = [el['target'] for el in save_outputs]
    #prediction = [el['prediction'] for el in save_outputs]

    metrics_to_save = {name: metric(target, prediction)
                       for name, metric in zip(metrics, metric_functions)}
    logger.info(''.join(f'{name}: {val}' for name, val in metrics_to_save.items()))

    #with (pretrained_dir / 'results.pkl').open('wb') as f:
    #    pkl.dump((metrics_to_save, save_outputs), f)
    #import pdb; pdb.set_trace()

    return metrics_to_save


def run_embed(model_type: str,
              data_file: str,
              out_file: str,
              from_pretrained: str,
              batch_size: int = 1024,
              model_config_file: typing.Optional[str] = None,
              full_sequence_embed: bool = False,
              no_cuda: bool = False,
              seed: int = 42,
              tokenizer: str = 'iupac',
              num_workers: int = 8,
              log_level: typing.Union[str, int] = logging.INFO) -> None:

    local_rank = -1  # TAPE does not support torch.distributed.launch for embedding
    device, n_gpu, is_master = utils.setup_distributed(local_rank, no_cuda)
    utils.setup_logging(local_rank, save_path=None, log_level=log_level)
    utils.set_random_seeds(seed, n_gpu)

    logger.info(
        f"device: {device} "
        f"n_gpu: {n_gpu}")

    task_spec = registry.get_task_spec('embed')
    model = registry.get_task_model(
        model_type, task_spec.name, model_config_file, from_pretrained)
    model = model.to(device)
    runner = ForwardRunner(model, device, n_gpu)
    runner.initialize_distributed_model()
    runner.eval()
    torch.set_grad_enabled(False)

    dataset = task_spec.dataset(data_file, tokenizer=tokenizer)  # type: ignore
    valid_loader = utils.setup_loader(dataset, batch_size, local_rank, n_gpu, 1, num_workers)

    with utils.IncrementalNPZ(out_file) as npzfile:
        with utils.wrap_cuda_oom_error(local_rank, batch_size, n_gpu):
            for batch in tqdm(valid_loader, total=len(valid_loader)):
                outputs = runner.forward(batch, no_loss=True)
                ids = batch['ids']
                sequence_embed = outputs[0]
                pooled_embed = outputs[1]
                sequence_lengths = batch['input_mask'].sum(1)
                sequence_embed = sequence_embed.cpu().numpy()
                pooled_embed = pooled_embed.cpu().numpy()
                sequence_lengths = sequence_lengths.cpu().numpy()

                for seqembed, poolembed, length, protein_id in zip(
                        sequence_embed, pooled_embed, sequence_lengths, ids):
                    seqembed = seqembed[:length]
                    arrays = {'pooled': poolembed}
                    if not full_sequence_embed:
                        # avgpool across the sequence
                        arrays['avg'] = seqembed.mean(0)
                    else:
                        arrays['seq'] = seqembed
                    to_save = {protein_id: arrays}
                    npzfile.savez(**to_save)
