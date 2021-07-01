'''
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''
import torch
import torch.nn.functional as F
# from transformers import MT5ForConditionalGeneration, T5Config, MT5EncoderModel, MT5Tokenizer, Trainer, TrainingArguments

from transformers_custom import T5ForConditionalGenerationWithLatentSpace, T5Tokenizer, T5Config
from progeny_tokenizer import TAPETokenizer
import numpy as np
import math
import random
import scipy
import time
import pandas as pd
from torch.utils.data import DataLoader, RandomSampler, Dataset, BatchSampler
import typing
from pathlib import Path
import argparse

from tqdm import tqdm, trange
import shutil
import os

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from tape.metrics import spearmanr

# argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--seed', action='store', type=int, default=30, help='random seed')
parser.add_argument('--data_dir', action='store', type=str, help='input df filename', default="data/sst" )
parser.add_argument('--pretrained_dir', action='store', type=str, help='dir path for pretrained progeny weights', default="t5-small" )

parser.add_argument('--output_dir', action='store', type=str, default="./debug_congen_trainer_results", help='input df filename' )
parser.add_argument('--num_train_epochs', action='store', type=int, default=12)
parser.add_argument('--per_device_train_batch_size', action='store', type=int, default=16)
parser.add_argument('--per_device_eval_batch_size', action='store', type=int, default=64)
parser.add_argument('--warmup_steps', action='store', type=int, default=500)
parser.add_argument('--weight_decay', action='store', type=float, default=0.01)
parser.add_argument('--logging_dir', action='store', type=str, default=None )
parser.add_argument('--save_total_limit', action='store', type=int, default=2)
parser.add_argument('--save_steps', action='store', type=int, default=2000)
parser.add_argument('--logging_steps', action='store', type=int, default=500)
parser.add_argument('--eval_steps', action='store', type=int, default=1000)
parser.add_argument('--num_warmup_steps', action='store', type=int, default=0)

parser.add_argument('--lr', action='store', type=float, default=5e-05, help='learning rate')
parser.add_argument('--train_ratio', action='store', type=float, default=1.0)
parser.add_argument('--train_split_name', action='store', type=str, default="train" )
parser.add_argument('--eval_split_name', action='store', type=str, default="valid" )

# latent space args
parser.add_argument('--latent_pooler', action='store', type=str, default="mean", choices=['mean', 'max', 'cls'], help='op to pool encoder hidden states' )
parser.add_argument('--pool_enc_hidden_states_for_dec', action='store_true')
parser.add_argument('--mask_non_target_z_vector', action='store_true')
parser.add_argument('--lambda_contrastive', action='store', type=float, default=1.0)
parser.add_argument('--lambda_contrastive_cyc', action='store', type=float, default=0.0)
parser.add_argument('--contrastive_cyc_start_step', action='store', type=int, default=-1, help='Step index to start contrastive_cyc loss minimization')

parser.add_argument('--lambda_contrastive_perturb_cyc', action='store', type=float, default=0.0)
parser.add_argument('--contrastive_perturb_cyc_start_step', action='store', type=int, default=-1, help='Step index to start contrastive_perturb_cyc loss minimization')
parser.add_argument('--pc_perturb', action='store', type=float, default=-0.25, help='Perturbation for contrastive_perturb_cyc loss')
parser.add_argument('--pc_perturb_type', action='store', type=str, default='std', choices=['std', 'fixed'], help='type of z perturbation for perturb cycle contrastive loss' )

parser.add_argument('--separate_targetattr_head', action='store_true')
parser.add_argument('--z_tar_vector_dim', action='store', type=int, default=1)
parser.add_argument('--do_mi', action='store_true')
parser.add_argument('--lambda_mi_head_loss', action='store', type=float, default=1.0)

# vae/wae args
parser.add_argument('--dim_target_kl', action='store', type=float, default=0.5)
parser.add_argument("--beta", type=float, default=1.0,
                    help="The weighting hyper-parameter of the KL term in VAE")
parser.add_argument("--lambda_logvar_L1", type=float, default=0.0,
                    help="Regularizing term to prevent z_logvar from being too large, recommended to be 0")
parser.add_argument("--lambda_logvar_KL", type=float, default=0.0,
                    help="Regularizing term to prevent z_logvar from diminishing, recommended to be 1e-3")
parser.add_argument("--use_beta_schedule", action='store_true', help="Use cyclical beta schedule for vae/wae.")
parser.add_argument("--beta_ratio_increase", default=0.25, type=float,
                    help="Learning schedule, the percentage for the annealing stage.") 
parser.add_argument("--beta_ratio_zero", default=0.25, type=float,
                    help="Learning schedule, the percentage for the pure auto-encoding stage.")    
parser.add_argument('--beta_start_step', action='store', type=int, default=-1, help='Step index to start z_regu_loss minimization')

parser.add_argument('--latent_space_type', action='store', type=str, default="plain", choices=['plain', 'vae', 'wae', 'adversarial'], help='type of latent space' )
parser.add_argument('--latent_size', action='store', type=int, default=None, help='use None to use pooled enc hidden state as latent vector')

parser.add_argument('--no_separate_latent_enc', action='store_false', dest='separate_latent_enc', default=True)
parser.add_argument('--no_separate_latent_dec', action='store_false', dest='separate_latent_dec', default=True)

# wae only args
parser.add_argument('--wae_z_enc_type', action='store', type=str, default=None, choices=['deterministic', 'stochastic'], help='type of wae encoder' )
parser.add_argument('--mmd_method', action='store', type=str, default="rf", choices=['rf', 'full_kernel'], help='random feature approx or full kernel for mmd computation' )
parser.add_argument("--sigma_mmd", type=float, default=None,
                    help="use None for default, RBF kernel width: ~ O( sqrt(z_dim) ), 7.0 for z_dim=100")    
parser.add_argument("--rf_dim_mmd", type=int, default=None,
                    help="Dim of random features")         

# SST5 args
parser.add_argument('--lambda_same_label_loss', action='store', type=float, default=0.0)
parser.add_argument('--train_omitted_labels', nargs='+', help='Labels to omit in training phase, labels are 0: strongly neg, 1: neg, 2: neutral, 3: pos, 4: strongly pos')
parser.add_argument('--train_reduced_labels', nargs='+', help='Labels to reduce samples in training phase, labels are 0: strongly neg, 1: neg, 2: neutral, 3: pos, 4: strongly pos')
parser.add_argument('--reduced_labels_keep_num', nargs='+', help='Number of samples to keep for reduced labels in training phase')

args = parser.parse_args()

if args.logging_dir is None:
    args.logging_dir = args.output_dir

print("args: ", args)

seed = args.seed
data_dir = args.data_dir
pretrained_dir = args.pretrained_dir
train_ratio = args.train_ratio
train_split_name = args.train_split_name
eval_split_name = args.eval_split_name

if args.train_omitted_labels is not None:
    train_omitted_labels = [int(train_omitted_label) for train_omitted_label in args.train_omitted_labels]
else:
    train_omitted_labels = None
print("train_omitted_labels: ", train_omitted_labels)

if args.train_reduced_labels is not None:
    train_reduced_labels = [int(train_omitted_label) for train_omitted_label in args.train_reduced_labels]
    reduced_labels_keep_num = [int(train_omitted_label) for train_omitted_label in args.reduced_labels_keep_num]
else:
    train_reduced_labels = None
    reduced_labels_keep_num = None
print("train_reduced_labels: ", train_reduced_labels)

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# tokenizer = TAPETokenizer(vocab="progeny")
tokenizer = T5Tokenizer.from_pretrained(pretrained_dir)

device = torch.device('cuda:0')

t5config = T5Config.from_pretrained(pretrained_dir)

latent_space_type = args.latent_space_type
wae_z_enc_type = args.wae_z_enc_type
latent_space_args = {
    'latent_pooler': args.latent_pooler,
    'pool_enc_hidden_states_for_dec': args.pool_enc_hidden_states_for_dec,
    'mask_non_target_z_vector': args.mask_non_target_z_vector,
    'separate_targetattr_head': args.separate_targetattr_head,
    'z_tar_vector_dim': args.z_tar_vector_dim,
    'do_mi': args.do_mi,
    'latent_space_type': args.latent_space_type,
    'separate_latent_enc': args.separate_latent_enc,
    'separate_latent_dec': args.separate_latent_dec,
    'wae_z_enc_type': args.wae_z_enc_type,
    'latent_size': args.latent_size,
    'dim_target_kl':  args.dim_target_kl,
    'mmd_method': args.mmd_method,
    'sigma_mmd': args.sigma_mmd,
    'rf_dim_mmd': args.rf_dim_mmd,
}

print("latent_space_args: ", latent_space_args)

# TODO: T5 model loading - start - 
model = T5ForConditionalGenerationWithLatentSpace.from_pretrained(pretrained_dir, **latent_space_args)
# TODO: T5 model loading - end - 

model.parallelize()


# TODO: add SST5 data loading pipeline - start - 
TEXT_COL, LABEL_COL = 'text', 'truth'

def read_sst5(data_dir, colnames=[LABEL_COL, TEXT_COL]):
    datasets = {}
    for t in ["train", "dev", "test"]:
        df = pd.read_csv(os.path.join(data_dir, f"sst_{t}.txt"), sep='\t', header=None, names=colnames)
        df[LABEL_COL] = df[LABEL_COL].str.replace('__label__', '')
        df[LABEL_COL] = df[LABEL_COL].astype(int)   # Categorical data type for truth labels
        df[LABEL_COL] = df[LABEL_COL] - 1  # Zero-index labels for PyTorch
        df[TEXT_COL] = df[TEXT_COL].str.replace("`", "'") # handle T5Tokenizer's inability to tokenize `, tokenizes it as <unk>
        datasets[t] = df
    return datasets

# def read_sst5(data_dir, colnames=[LABEL_COL, TEXT_COL]):
#     datasets = {}
#     for t in ["train", "dev", "test"]:
#         df = pd.read_csv(os.path.join(data_dir, f"sst_{t}.txt"), sep='\t', header=None, names=colnames)
#         df[LABEL_COL] = df[LABEL_COL].str.replace('__label__', '')
#         df[LABEL_COL] = df[LABEL_COL].astype(int)   # Categorical data type for truth labels
#         df[LABEL_COL] = df[LABEL_COL] - 1  # Zero-index labels for PyTorch
#         datasets[t] = df
#     return datasets

class TextDFDatasetForGen(Dataset):
    """Creates a dataset from an df file.
    Args:
        data_file (typing.Union[str, Path]): Path to pkl df file.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """

    def __init__(self,
                df,
                in_memory: bool = False,
                split: str = None,
                train_ratio: float = 1,
                omitted_labels=None,
                reduced_labels=None, 
                reduced_labels_keep_num=None,
                ):
        
        if omitted_labels is not None:
            df = df.loc[~df['truth'].isin(omitted_labels)]

        if reduced_labels is not None:
            assert len(reduced_labels) == len(reduced_labels_keep_num)
            df_wo_reduced_labels = df.loc[~df['truth'].isin(reduced_labels)]
            print("len(df_wo_reduced_labels): ", len(df_wo_reduced_labels))
            kept_rows = None
            df_w_keep = None
            for label_ind, label in enumerate(reduced_labels):
                keep_num = reduced_labels_keep_num[label_ind]
                label_rows =  df.loc[df['truth'] == label]
                kept_label_rows = label_rows.iloc[:keep_num]
                print("kept_label_rows: ", kept_label_rows)
                print("len(kept_label_rows): ", len(kept_label_rows))
                if df_w_keep is None:
                    df_w_keep = df_wo_reduced_labels.append(kept_label_rows)
                else:
                    df_w_keep = df_w_keep.append(kept_label_rows)

                print("len(df_w_keep): ", len(df_w_keep))

            df = df_w_keep

        if train_ratio != 1 and split != None:
            shuffled_df = df.sort_index()
            # shuffled_df = df.sample(frac=1)
            train_num_samples = int(len(shuffled_df) * train_ratio)
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
            item['sentiment_scores'] = row['truth'] 
            item['input_ids'] = row['text']
            item['labels'] = row['text']

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


class CustomTextDatasetForGenLatentSpace(Dataset):

    def __init__(self,
                df,
                tokenizer,
                split: str,
                in_memory: bool = False,
                train_ratio: float = 1,
                omitted_labels = None, # list of label to omit from dataset
                reduced_labels=None, 
                reduced_labels_keep_num=None,
                prepended_cls_token='<extra_id_0>',
                ):

        self.tokenizer = tokenizer

        if split == 'valid':
            file_prefix = 'train'
        else:
            file_prefix = split

        self.data = TextDFDatasetForGen(df, in_memory, split, train_ratio, omitted_labels=omitted_labels, reduced_labels=reduced_labels, reduced_labels_keep_num=reduced_labels_keep_num)
        self.omitted_labels = omitted_labels
        self.reduced_labels = reduced_labels
        self.reduced_labels_keep_num = reduced_labels_keep_num

        if prepended_cls_token is not None:
            self.prepended_cls_token_id = self.tokenizer.encode(prepended_cls_token)[0]
        else:
            self.prepended_cls_token_id = None

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        # decode first encoded ids to remove space before punctuations such as " ," and " ."
        # print("item['input_ids']: ", item['input_ids'])
        # print("self.tokenizer.decode(self.tokenizer.encode(item['input_ids'])): ", self.tokenizer.decode(self.tokenizer.encode(item['input_ids'])))
        input_ids = self.tokenizer.encode(self.tokenizer.decode(self.tokenizer.encode(item['input_ids'])))
        labels = self.tokenizer.encode(self.tokenizer.decode(self.tokenizer.encode(item['labels'])))
        # input_ids = self.tokenizer.encode(item['input_ids'])
        # labels = self.tokenizer.encode(item['labels'])


        if self.prepended_cls_token_id is not None:
            input_ids = [self.prepended_cls_token_id] + input_ids
            labels = [self.prepended_cls_token_id] + labels

        
        input_ids = np.array(input_ids, np.int64)
        labels = np.array(labels, np.int64)
        
        sentiment_scores = item['sentiment_scores']
#         print("__getitem__input_ids: ", input_ids)
#         print("__getitem__labels: ", labels)
#         print("__getitem__sentiment_scores: ", sentiment_scores)
        
#         print("__getitem__input_ids type: ", type(input_ids))
#         print("__getitem__labels type: ", type(labels))
#         print("__getitem__sentiment_scores : ", type(sentiment_scores))
        
#         np.array(token_ids, np.int64)
        return input_ids, labels, sentiment_scores

    
    def collate_fn(self, batch: typing.List[typing.Tuple[typing.Any, ...]]) -> typing.Dict[str, torch.Tensor]:
        input_ids, labels, sentiment_scores = tuple(zip(*batch))
        # print("input_ids: ", input_ids)
        # print("input_ids len: ", len(input_ids))
        # print("input_ids[0].shape: ", input_ids[0].shape)
        # print("input_ids[1].shape: ", input_ids[1].shape)
#         print("labels: ", labels)
#         print("sentiment_scores: ", sentiment_scores)
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        labels = torch.from_numpy(pad_sequences(labels, 0))
        sentiment_scores = torch.Tensor(sentiment_scores)

        return {'input_ids': input_ids,
                'labels': labels,
                'sentiment_scores': sentiment_scores}


# TODO: add SST5 data loading pipeline - end - 





def spearmanr(target, prediction):
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return scipy.stats.spearmanr(target_array, prediction_array).correlation
    
def evaluate(model, eval_iterator, do_mi=False, do_contrast_spearmanr=True, latent_space_type='plain', return_pred=False):
    eval_contrastive_loss_total = 0
    eval_lm_loss_total = 0
    eval_same_label_loss_total = 0
    if do_mi:
        eval_mi_head_loss_total = 0
    if latent_space_type in ['vae', 'wae']:
        eval_z_regu_loss_total = 0
    model.eval()
    num_eval_batch = 0
    
    contrast_preds=[]
    contrast_targs = []

    with torch.no_grad():
        for step, batch in enumerate(eval_iterator):
            
            input_ids = batch['input_ids'].to(model.device)
            labels = batch['labels'].to(model.device)
            contrast_targets = batch['sentiment_scores'].to(model.device)
    
            # if do_mi:
            #     model_outputs = model(input_ids, labels=labels, contrast_targets=contrast_targets)
            #     outputs, contrastive_loss, contrastive_value, mi_head_loss = model_outputs[0], model_outputs[1], model_outputs[2], model_outputs[3]
            #     eval_mi_head_loss_total = eval_mi_head_loss_total + mi_head_loss
            # else:
            #     model_outputs = model(input_ids, labels=labels, contrast_targets=contrast_targets)
            #     outputs, contrastive_loss, contrastive_value = model_outputs[0], model_outputs[1], model_outputs[2]
                        
            if do_mi:
                model_outputs = model(input_ids, labels=labels, contrast_targets=contrast_targets, mask_similar_contrast_label=True, return_same_label_loss=True)
                #!
                outputs, contrastive_loss, contrastive_value, mi_head_loss = model_outputs[0], model_outputs[1], model_outputs[2], model_outputs[4]
                # outputs, contrastive_loss, contrastive_value, mi_head_loss = model_outputs[0], model_outputs[1], model_outputs[2], model_outputs[3]
                eval_mi_head_loss_total = eval_mi_head_loss_total + mi_head_loss
            else:
                model_outputs = model(input_ids, labels=labels, contrast_targets=contrast_targets, mask_similar_contrast_label=True, return_same_label_loss=True)
                outputs, contrastive_loss, contrastive_value = model_outputs[0], model_outputs[1], model_outputs[2]
            
            same_label_loss = model_outputs[3]
            eval_same_label_loss_total = eval_same_label_loss_total + same_label_loss

            if latent_space_type in ['vae', 'wae']:
                z_regu_output = model_outputs[-1]
                if type(z_regu_output) is dict:
                    z_regu_loss = z_regu_output['z_regu_loss']
                else:
                    z_regu_loss = z_regu_output
                # z_regu_loss = model_outputs[-1]

            for pred, target in zip(contrastive_value.squeeze().cpu().numpy(), contrast_targets.cpu().numpy()):
#                 print("target: ", target)
#                 print("pred: ", pred)
                contrast_targs.append(target)
                contrast_preds.append(pred)

            lm_loss = outputs.loss
            
            eval_contrastive_loss_total = eval_contrastive_loss_total + contrastive_loss
            eval_lm_loss_total = eval_lm_loss_total + lm_loss

            if latent_space_type in ['vae', 'wae']:
                eval_z_regu_loss_total = eval_z_regu_loss_total + z_regu_loss
            
            # eval_contrastive_losses.append(contrastive_loss)
            # eval_lm_losses.append(lm_loss)

            num_eval_batch += 1

#             if step == 5:
#                 break

    # eval_contrastive_loss = torch.mean(eval_contrastive_losses)
    # eval_lm_loss = torch.mean(eval_lm_losses)
    eval_lm_loss = eval_lm_loss_total / num_eval_batch
    eval_contrastive_loss = eval_contrastive_loss_total / num_eval_batch
    eval_same_label_loss = eval_same_label_loss_total / num_eval_batch
    eval_output = {
                "lm_loss": eval_lm_loss,
                "contrastive_loss": eval_contrastive_loss,
                "same_label_loss": eval_same_label_loss,
                  }

    if do_mi:
        eval_mi_head_loss_total = eval_mi_head_loss_total / num_eval_batch
        eval_output['mi_head_loss'] = eval_mi_head_loss_total

    if latent_space_type in ['vae', 'wae']:
        eval_z_regu_loss_total = eval_z_regu_loss_total / num_eval_batch
        eval_output['z_regu_loss'] = eval_z_regu_loss_total

    if do_contrast_spearmanr:
        spearmanr_value = spearmanr(contrast_targs, contrast_preds)
        print("spearmanr_value: ", spearmanr_value)
        eval_output['spearmanr'] = spearmanr_value
    
    if return_pred:
        eval_output['contrast_preds'] = contrast_preds
        eval_output['contrast_targs'] = contrast_targs


    # print("eval_contrastive_loss: ", eval_contrastive_loss)
    # print("eval_lm_loss: ", eval_lm_loss)
    return eval_output

def frange_cycle_zero_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio_increase=0.5, ratio_zero=0.3):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio_increase) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            if i < period*ratio_zero:
                L[int(i+c*period)] = start
            else: 
                L[int(i+c*period)] = v
                v += step
            i += 1
    return L 

# TODO: add SST5 data loading pipeline - start - 
# dataset_path = "data/sst"
# datasets = read_sst5(dataset_path)
datasets = read_sst5(data_dir)

train_dataset = CustomTextDatasetForGenLatentSpace(datasets['train'], tokenizer=tokenizer, split=None, omitted_labels=train_omitted_labels, reduced_labels=train_reduced_labels, reduced_labels_keep_num=reduced_labels_keep_num)
# train_dataset = CustomTextDatasetForGenLatentSpace(datasets['train'], tokenizer=tokenizer, split=None, omitted_labels=train_omitted_labels)
eval_dataset = CustomTextDatasetForGenLatentSpace(datasets['dev'], tokenizer=tokenizer, split=None)
if train_omitted_labels != None:
    eval_dataset_w_train_omitted_labels = CustomTextDatasetForGenLatentSpace(datasets['dev'], tokenizer=tokenizer, split=None, omitted_labels=train_omitted_labels)

# train_dataset = CustomStabilityDatasetForGenLatentSpace(data_dir, train_split_name, train_ratio=train_ratio, tokenizer=tokenizer)
# eval_dataset = CustomStabilityDatasetForGenLatentSpace(data_dir, eval_split_name, train_ratio=train_ratio, tokenizer=tokenizer)

# TODO: add SST5 data loading pipeline - end - 


num_training_steps=args.num_train_epochs*len(train_dataset)//args.per_device_train_batch_size


# Train data set-up
train_loader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True, 
                        num_workers=0, collate_fn=train_dataset.collate_fn)

epoch_iterator = tqdm(train_loader)

# Eval data set-up
eval_loader = DataLoader(eval_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False, 
                        num_workers=0, collate_fn=train_dataset.collate_fn)

eval_iterator = tqdm(eval_loader)

if train_omitted_labels != None:
    eval_loader_w_train_omitted_labels = DataLoader(eval_dataset_w_train_omitted_labels, batch_size=args.per_device_eval_batch_size, shuffle=False, 
                        num_workers=0, collate_fn=train_dataset.collate_fn)

eval_iterator_w_train_omitted_labels = tqdm(eval_loader_w_train_omitted_labels)


# set up tensorboard writer
logging_dir = Path(args.logging_dir)
logging_dir.mkdir(parents=True, exist_ok=True)
tb_writer = SummaryWriter(logging_dir)

# optimizer set up
from transformers import AdamW
# optimizer = AdamW(model.parameters(), lr=1e-5)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if ('mi_head' not in n and not any(nd in n for nd in no_decay))], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if ('mi_head' not in n and any(nd in n for nd in no_decay))], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

# for n, p in model.named_parameters():
#     print("n: ", n)
#     print("'mi_head' not in n: ", 'mi_head' not in n)
    # print("p: ", p)

# lr scheduling
from transformers.optimization import Adafactor, AdamW, get_scheduler
# from transformers import get_linear_schedule_with_warmup
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)
lr_scheduler = get_scheduler(
                'linear',
                optimizer,
                num_warmup_steps=args.num_warmup_steps,
                num_training_steps=num_training_steps,
            )

if args.do_mi:
    mi_optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if ('mi_head' in n and not any(nd in n for nd in no_decay))], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if ('mi_head' in n and any(nd in n for nd in no_decay))], 'weight_decay': 0.0}
    ]

    mi_optimizer = AdamW(mi_optimizer_grouped_parameters, lr=args.lr)
    mi_lr_scheduler = get_scheduler(
                    'linear',
                    mi_optimizer,
                    num_warmup_steps=args.num_warmup_steps,
                    num_training_steps=num_training_steps,
                )

global_step = 0
n_iter = int(args.num_train_epochs * len(epoch_iterator))
# print("len(epoch_iterator): ", len(epoch_iterator))
# print("args.num_train_epochs: ", args.num_train_epochs)
print("n_iter: ", n_iter)

beta_t_list = frange_cycle_zero_linear(n_iter, start=0.0, stop=args.beta, n_cycle=10,  ratio_increase=args.beta_ratio_increase, ratio_zero=args.beta_ratio_zero)
if args.beta_start_step > 0:
    beta_t_list[:args.beta_start_step] = 0

model.train()
for epoch in trange(1, args.num_train_epochs+1):
    for step, batch in enumerate(epoch_iterator):
        # print("step: ", step)
        # print("batch: ", batch)
        # print("batch['sentiment_scores'].shape: ", batch['sentiment_scores'].shape)
        
        input_ids = batch['input_ids'].to(model.device)
        labels = batch['labels'].to(model.device)
        contrast_targets = batch['sentiment_scores'].to(model.device)

        # print("input_ids: ", input_ids)
        # print("labels: ", labels)
        # print("contrast_targets: ", contrast_targets)
        # print("input_ids.shape: ", input_ids.shape)
        # print("labels.shape: ", labels.shape)
        # print("contrast_targets.shape: ", contrast_targets.shape)

        model.zero_grad()
        # print("input_ids: ", input_ids)
        if args.do_mi:
            # train mi_head
            mi_head_loss = model(input_ids, labels=labels, contrast_targets=contrast_targets, train_mi_head_step=True, mask_similar_contrast_label=True, return_same_label_loss=True)
            mi_head_loss.backward()
            mi_optimizer.step()
            mi_lr_scheduler.step()

            model.zero_grad()

            # train generator model
            model_outputs = model(input_ids, labels=labels, contrast_targets=contrast_targets, mask_similar_contrast_label=True, return_same_label_loss=True)
            #!
            outputs, contrastive_loss, contrastive_value, mi_head_loss = model_outputs[0], model_outputs[1], model_outputs[2], model_outputs[4]
            # outputs, contrastive_loss, contrastive_value, mi_head_loss = model_outputs[0], model_outputs[1], model_outputs[2], model_outputs[3]
            lm_loss = outputs.loss
            
            total_loss = lm_loss
            if args.lambda_contrastive > 0:
                total_loss = total_loss + args.lambda_contrastive * contrastive_loss - args.lambda_mi_head_loss * mi_head_loss # generator optimized to increase mi_head_loss 
        else:
            # print("1st input_ids.shape: ", input_ids.shape)
            # print("Train Check 1")
            model_outputs = model(input_ids, labels=labels, contrast_targets=contrast_targets, mask_similar_contrast_label=True, return_same_label_loss=True)
                
            outputs, contrastive_loss, contrastive_value = model_outputs[0], model_outputs[1], model_outputs[2]
            lm_loss = outputs.loss

            total_loss = lm_loss
            if args.lambda_contrastive > 0:
                total_loss = total_loss + args.lambda_contrastive * contrastive_loss

        same_label_loss = model_outputs[3]
        if args.lambda_same_label_loss > 0:
            # print("B same_label_loss: ", same_label_loss)
            total_loss = total_loss + args.lambda_same_label_loss * same_label_loss
        
        # print("contrastive_value.shape: ", contrastive_value.shape)

        if latent_space_type in ['vae', 'wae']:
            # z_regu_loss = model_outputs[-1]
            z_regu_output = model_outputs[-1]
            if type(z_regu_output) is dict:
                z_regu_loss = z_regu_output['z_regu_loss']
            else:
                z_regu_loss = z_regu_output
                
            if args.use_beta_schedule and global_step < len(beta_t_list):
                beta_z_regu = beta_t_list[global_step]
            else:
                if args.beta_start_step > 0 and global_step < args.beta_start_step:
                    beta_z_regu = 0
                else:
                    beta_z_regu = args.beta # constant value
            # print("z_regu_loss.shape: ", z_regu_loss.shape)
            if beta_z_regu > 0:
                # print("Train NegCheck 1")
                total_loss = total_loss + beta_z_regu * z_regu_loss

            if wae_z_enc_type != 'deterministic':
                # print("Train NegCheck 2")
                z_logvar_L1 = z_regu_output['z_logvar_L1']
                if args.lambda_logvar_L1 > 0 and beta_z_regu > 0:
                    # prevent z_logvar from being too large
                    total_loss = total_loss + beta_z_regu * args.lambda_logvar_L1 * z_logvar_L1

                z_logvar_KL_penalty = z_regu_output['z_logvar_KL_penalty']
                if args.lambda_logvar_KL > 0 and beta_z_regu > 0:
                    # prevent z_logvar from diminishing
                    total_loss = total_loss + beta_z_regu * args.lambda_logvar_KL * z_logvar_KL_penalty
        

        # perturb cycle consistency loss
        if args.lambda_contrastive_perturb_cyc > 0 and global_step > args.contrastive_perturb_cyc_start_step:
        # if True:
        # if do_perturb_cycle_consistency:
            # print("PC input_ids: ", input_ids)
            # print("PC input_ids.shape: ", input_ids.shape)
            # print("PC input_ids.shape -1: ", input_ids.shape[-1])

            # pc_perturb based on std of 1st step's value_pred
            if args.pc_perturb_type == 'std':
                contrastive_value_std = torch.std(contrastive_value)
                pc_perturb = args.pc_perturb * contrastive_value_std.item()
            elif args.pc_perturb_type == 'static':
                pc_perturb = args.pc_perturb
   
            # print("pc_perturb: ", pc_perturb)
            gen_output = model.generate(input_ids, max_length=input_ids.shape[-1]+1, return_dict_in_generate=True, output_scores=True, z_tar_edit_before_dec=pc_perturb) # change z_tar_edit_before_dec
            gen_logits = torch.stack(gen_output.scores, dim=1)

            pc_gen_value_pred = model(inputs_logits=gen_logits, return_only_value_pred=True, mask_similar_contrast_label=True, return_same_label_loss=True)
            # print("pc_gen_value_pred.shape: ", pc_gen_value_pred.shape) # torch.Size([16, 1])
            # contrastive_pc_loss = model_outputs_pc_forward[1]
            # total_loss = total_loss + args.lambda_contrastive_pc * contrastive_pc_loss

            # print("PC gen_output.sequences: ", gen_output.sequences)
            # # print("PC gen_output.scores: ", gen_output.scores)
            # print("PC gen_output.sequences shape: ", gen_output.sequences.shape) # 86 as sequences is dec's input_ids which includes a pad token at the start 
            # print("PC gen_output.scores len: ", len(gen_output.scores)) # len # 85
            # print("PC gen_output.scores[0].shape: ", gen_output.scores[0].shape) # torch.Size([16, 30080])
            # print("PC torch.stack(gen_output.scores, dim=1).shape: ", torch.stack(gen_output.scores, dim=1).shape) # torch.Size([16, 30080])
            # print("PC gen_output.scores shape: ", gen_output.scores.shape)

            # contrastive loss for contrastive_value vs pc_gen_value_pred
            if len(contrast_targets.shape) != 2:
                contrast_targets = torch.unsqueeze(contrast_targets, dim=-1)
            pc_gen_contrast_targets = contrast_targets + pc_perturb
            # print("pc_perturb: ", pc_perturb)
            # print("contrast_targets: ", contrast_targets)
            pc_contrast_labels = torch.sign(contrast_targets-pc_gen_contrast_targets)*0.5 + 0.5
            # print("pc_gen_contrast_targets: ", pc_gen_contrast_targets)
            # print("pc_contrast_labels: ", pc_contrast_labels)
            # print("pc_gen_contrast_targets.shape: ", pc_gen_contrast_targets.shape) # torch.Size([16])
            # print("pc_contrast_labels.shape: ", pc_contrast_labels.shape) # torch.Size([16])
            contrastive_preds = F.logsigmoid(contrastive_value-pc_gen_value_pred)   
            # print("contrastive_preds.shape: ", contrastive_preds.shape) 
            inverse_preds = F.logsigmoid(-1*(contrastive_value-pc_gen_value_pred))   
            # print("inverse_preds.shape: ", inverse_preds.shape) # torch.Size([16, 1])
            pc_losses = -pc_contrast_labels*contrastive_preds - (1-pc_contrast_labels)*inverse_preds
            # print("pc_losses.shape: ", pc_losses.shape) # torch.Size([16, 1])
            contrastive_perturb_cyc_loss = pc_losses.mean()

            total_loss = total_loss + args.lambda_contrastive_perturb_cyc * contrastive_perturb_cyc_loss

        # print("outputs.logits: ", outputs.logits)
        # print("outputs.logits.shape: ", outputs.logits.shape) # torch.Size([16, 85, 30080])
        if args.lambda_contrastive_cyc > 0 and global_step > args.contrastive_cyc_start_step:
            # print("model_outputs_2nd_forward")
            model_outputs_2nd_forward = model(inputs_logits=outputs.logits, labels=labels, contrast_targets=contrast_targets, mask_similar_contrast_label=True, return_same_label_loss=True)
            contrastive_cyc_loss = model_outputs_2nd_forward[1]
            total_loss = total_loss + args.lambda_contrastive_cyc * contrastive_cyc_loss

        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        # print("contrastive_loss: ", contrastive_loss)
        # print("lm_loss: ", lm_loss)

        global_step += 1
        
        if global_step % args.logging_steps == 0:
            tb_writer.add_scalar("TRAIN/lr", lr_scheduler.get_last_lr()[0], global_step)
            # tb_writer.add_scalar("TRAIN/lr", lr_scheduler.get_lr()[0], global_step)
            tb_writer.add_scalar("TRAIN/contrastive_loss", contrastive_loss, global_step)
            tb_writer.add_scalar("TRAIN/lm_loss", lm_loss, global_step)
            tb_writer.add_scalar("TRAIN/same_label_loss", same_label_loss, global_step)

            if args.do_mi:
                tb_writer.add_scalar("TRAIN/mi_head_loss", mi_head_loss, global_step)
            if latent_space_type in ['vae', 'wae']:
                tb_writer.add_scalar("TRAIN/z_regu_loss", z_regu_loss, global_step)
                tb_writer.add_scalar("TRAIN/beta_z_regu", beta_z_regu, global_step)
                if wae_z_enc_type != 'deterministic':
                    tb_writer.add_scalar("TRAIN/z_logvar_L1", z_logvar_L1, global_step)
                    tb_writer.add_scalar("TRAIN/z_logvar_KL_penalty", z_logvar_KL_penalty, global_step)

            if args.lambda_contrastive_cyc > 0 and global_step > args.contrastive_cyc_start_step:
                tb_writer.add_scalar("TRAIN/contrastive_cyc_loss", contrastive_cyc_loss, global_step)

            if args.lambda_contrastive_perturb_cyc > 0 and global_step > args.contrastive_perturb_cyc_start_step:
                tb_writer.add_scalar("TRAIN/contrastive_perturb_cyc_loss", contrastive_perturb_cyc_loss, global_step)

                

        if global_step % args.eval_steps == 0:
            model.eval()
            eval_output = evaluate(model, eval_iterator, do_mi=args.do_mi, latent_space_type=latent_space_type)
            eval_lm_loss, eval_contrastive_loss, eval_spearmanr = eval_output['lm_loss'], eval_output['contrastive_loss'], eval_output['spearmanr']
            tb_writer.add_scalar("EVAL/lm_loss", eval_lm_loss, global_step)
            tb_writer.add_scalar("EVAL/contrastive_loss", eval_contrastive_loss, global_step)
            
            eval_same_label_loss = eval_output['same_label_loss']
            tb_writer.add_scalar("EVAL/same_label_loss", eval_same_label_loss, global_step)

            tb_writer.add_scalar("EVAL/spearmanr", eval_spearmanr, global_step)
            if args.do_mi:
                eval_mi_head_loss = eval_output['mi_head_loss']
                tb_writer.add_scalar("EVAL/mi_head_loss", eval_mi_head_loss, global_step)
            if latent_space_type in ['vae', 'wae']:
                eval_z_regu_loss = eval_output['z_regu_loss']
                tb_writer.add_scalar("EVAL/z_regu_loss", eval_z_regu_loss, global_step)

            if train_omitted_labels != None:
                eval_output_w_train_omitted_labels = evaluate(model, eval_iterator_w_train_omitted_labels, do_mi=args.do_mi, latent_space_type=latent_space_type)
                eval_contrastive_loss_w_train_omitted_labels, eval_spearmanr_w_train_omitted_labels = eval_output_w_train_omitted_labels['contrastive_loss'], eval_output_w_train_omitted_labels['spearmanr']

                tb_writer.add_scalar("EVAL/contrastive_loss_w_train_omitted_labels", eval_contrastive_loss_w_train_omitted_labels, global_step)
                tb_writer.add_scalar("EVAL/spearmanr_w_train_omitted_labels", eval_spearmanr_w_train_omitted_labels, global_step)
            
            model.train()

        # if global_step % args.save_steps == 0:
        #     weights_name = "pytorch_model.bin"
        #     saved_weights_file = os.path.join(output_dir, output_dir, weights_name)
        #     torch.save(con2_block.state_dict(), saved_weights_file)


# Final log step
tb_writer.add_scalar("TRAIN/lr", lr_scheduler.get_last_lr()[0], global_step)
tb_writer.add_scalar("TRAIN/contrastive_loss", contrastive_loss, global_step)
tb_writer.add_scalar("TRAIN/same_label_loss", same_label_loss, global_step)
tb_writer.add_scalar("TRAIN/lm_loss", lm_loss, global_step)
if args.do_mi:
    tb_writer.add_scalar("TRAIN/mi_head_loss", mi_head_loss, global_step)

if latent_space_type in ['vae', 'wae']:
    tb_writer.add_scalar("TRAIN/z_regu_loss", z_regu_loss, global_step)
    tb_writer.add_scalar("TRAIN/beta_z_regu", beta_z_regu, global_step)
    if wae_z_enc_type != 'deterministic':
        tb_writer.add_scalar("TRAIN/z_logvar_L1", z_logvar_L1, global_step)
        tb_writer.add_scalar("TRAIN/z_logvar_KL_penalty", z_logvar_KL_penalty, global_step)

if args.lambda_contrastive_cyc > 0 and global_step > args.contrastive_cyc_start_step:
    tb_writer.add_scalar("TRAIN/contrastive_cyc_loss", contrastive_cyc_loss, global_step)

if args.lambda_contrastive_perturb_cyc > 0 and global_step > args.contrastive_perturb_cyc_start_step:
    tb_writer.add_scalar("TRAIN/contrastive_perturb_cyc_loss", contrastive_perturb_cyc_loss, global_step)

# Final evaluation
model.eval()
eval_output = evaluate(model, eval_iterator, do_mi=args.do_mi, latent_space_type=latent_space_type)
eval_lm_loss, eval_contrastive_loss, eval_spearmanr = eval_output['lm_loss'], eval_output['contrastive_loss'], eval_output['spearmanr']
tb_writer.add_scalar("EVAL/lm_loss", eval_lm_loss, global_step)
tb_writer.add_scalar("EVAL/contrastive_loss", eval_contrastive_loss, global_step)
tb_writer.add_scalar("EVAL/spearmanr", eval_spearmanr, global_step)
if args.do_mi:
    eval_mi_head_loss = eval_output['mi_head_loss']
    tb_writer.add_scalar("EVAL/mi_head_loss", eval_mi_head_loss, global_step)
if latent_space_type in ['vae', 'wae']:
    eval_z_regu_loss = eval_output['z_regu_loss']
    tb_writer.add_scalar("EVAL/z_regu_loss", eval_z_regu_loss, global_step)

if train_omitted_labels != None:
    eval_output_w_train_omitted_labels = evaluate(model, eval_iterator_w_train_omitted_labels, do_mi=args.do_mi, latent_space_type=latent_space_type)
    eval_contrastive_loss_w_train_omitted_labels, eval_spearmanr_w_train_omitted_labels = eval_output_w_train_omitted_labels['contrastive_loss'], eval_output_w_train_omitted_labels['spearmanr']

    tb_writer.add_scalar("EVAL/contrastive_loss_w_train_omitted_labels", eval_contrastive_loss_w_train_omitted_labels, global_step)
    tb_writer.add_scalar("EVAL/spearmanr_w_train_omitted_labels", eval_spearmanr_w_train_omitted_labels, global_step)


# evaluate on full training set
eval_output = evaluate(model, epoch_iterator, do_mi=args.do_mi, latent_space_type=latent_space_type)
eval_lm_loss, eval_contrastive_loss, eval_spearmanr = eval_output['lm_loss'], eval_output['contrastive_loss'], eval_output['spearmanr']
tb_writer.add_scalar("EVAL/train_lm_loss", eval_lm_loss, global_step)
tb_writer.add_scalar("EVAL/train_contrastive_loss", eval_contrastive_loss, global_step)
tb_writer.add_scalar("EVAL/train_spearmanr", eval_spearmanr, global_step)


results_txt_name = "eval_results.txt"
results_path = output_dir / results_txt_name
with open(results_path, "w") as writer:
    for key in sorted(eval_output.keys()):
        writer.write("%s = %s\n" % (key, str(eval_output[key])))

weights_name = "pytorch_model.bin"
saved_weights_file = output_dir / weights_name
torch.save(model.state_dict(), saved_weights_file)

torch.save(args, output_dir / "training_args.bin")
torch.save(optimizer.state_dict(), output_dir / "optimizer.pt")
torch.save(lr_scheduler.state_dict(), output_dir / "scheduler.pt")

model.save_pretrained(save_directory=output_dir)
# copy config.jsonl from pretrained_dir to output_dir
# src_json = os.path.join(pretrained_dir, 'config.json')
# shutil.copy(src_json, output_dir)


