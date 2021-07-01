'''
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''
import torch
import torch.nn.functional as F
# from transformers import MT5ForConditionalGeneration, T5Config, MT5EncoderModel, MT5Tokenizer, Trainer, TrainingArguments

from transformers_custom import T5Discriminator, T5Tokenizer, T5Config
# from progeny_tokenizer import TAPETokenizer
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

parser.add_argument('--output_dir', action='store', type=str, default="./SST5_disc_results", help='input df filename' )
parser.add_argument('--num_train_epochs', action='store', type=int, default=12)
parser.add_argument('--per_device_train_batch_size', action='store', type=int, default=16)
parser.add_argument('--per_device_eval_batch_size', action='store', type=int, default=64)
parser.add_argument('--warmup_steps', action='store', type=int, default=500)
parser.add_argument('--weight_decay', action='store', type=float, default=0.01)
parser.add_argument('--logging_dir', action='store', type=str, default=None )
parser.add_argument('--save_steps', action='store', type=int, default=2000)
parser.add_argument('--logging_steps', action='store', type=int, default=500)
parser.add_argument('--eval_steps', action='store', type=int, default=500)
parser.add_argument('--num_warmup_steps', action='store', type=int, default=0)

parser.add_argument('--lr', action='store', type=float, default=5e-05, help='learning rate')
parser.add_argument('--train_ratio', action='store', type=float, default=1.0)
parser.add_argument('--train_split_name', action='store', type=str, default="train" )
parser.add_argument('--eval_split_name', action='store', type=str, default="valid" )

parser.add_argument('--latent_pooler', action='store', type=str, default="mean", choices=['mean', 'max', 'cls'], help='op to pool encoder hidden states' )

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

tokenizer = T5Tokenizer.from_pretrained(pretrained_dir)


# print("tokenizer.cls_token_id: ", tokenizer.cls_token_id)
# print("tokenizer._extra_ids: ", tokenizer._extra_ids)
# print("tokenizer.encode('<extra_id_99>') ", tokenizer.encode('<extra_id_99>'))
# print("tokenizer.encode('<extra_id_0>') ", tokenizer.encode('<extra_id_0>'))
# print("tokenizer.additional_special_tokens: ", tokenizer.additional_special_tokens)
# print("tokenizer.get_vocab: ", tokenizer.get_vocab())


device = torch.device('cuda:0')

t5config = T5Config.from_pretrained(pretrained_dir)

disc_args = {
    'latent_pooler': args.latent_pooler,
}

# latent_space_type = args.latent_space_type
# wae_z_enc_type = args.wae_z_enc_type
# latent_space_args = {
#     'latent_pooler': args.latent_pooler,
#     'pool_enc_hidden_states_for_dec': args.pool_enc_hidden_states_for_dec,
#     'mask_non_target_z_vector': args.mask_non_target_z_vector,
#     'separate_targetattr_head': args.separate_targetattr_head,
#     'z_tar_vector_dim': args.z_tar_vector_dim,
#     'do_mi': args.do_mi,
#     'latent_space_type': args.latent_space_type,
#     'separate_latent_enc': args.separate_latent_enc,
#     'separate_latent_dec': args.separate_latent_dec,
#     'wae_z_enc_type': args.wae_z_enc_type,
#     'latent_size': args.latent_size,
#     'dim_target_kl':  args.dim_target_kl,
#     'mmd_method': args.mmd_method,
#     'sigma_mmd': args.sigma_mmd,
#     'rf_dim_mmd': args.rf_dim_mmd,
# }

# print("latent_space_args: ", latent_space_args)

# TODO: T5 model loading - start - 
model = T5Discriminator.from_pretrained(pretrained_dir, **disc_args)
# model = T5ForConditionalGenerationWithLatentSpace.from_pretrained(pretrained_dir, **latent_space_args)
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


class TextDFDatasetForDisc(Dataset):
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


class CustomTextDatasetForDisc(Dataset):

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

        self.data = TextDFDatasetForDisc(df, in_memory, split, train_ratio, omitted_labels=omitted_labels, reduced_labels=reduced_labels, reduced_labels_keep_num=reduced_labels_keep_num)
        # self.data = TextDFDatasetForDisc(df, in_memory, split, train_ratio, omitted_labels=omitted_labels)
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
        input_ids = self.tokenizer.encode(item['input_ids'])


        if self.prepended_cls_token_id is not None:
            # print("A input_ids: ", input_ids)
            input_ids = [self.prepended_cls_token_id] + input_ids
            # print("B input_ids: ", input_ids)

        input_ids = np.array(input_ids, np.int64)

        sentiment_scores = item['sentiment_scores']

        return input_ids, sentiment_scores

    
    def collate_fn(self, batch: typing.List[typing.Tuple[typing.Any, ...]]) -> typing.Dict[str, torch.Tensor]:
        input_ids, sentiment_scores = tuple(zip(*batch))
#         print("input_ids: ", input_ids)
#         print("labels: ", labels)
#         print("sentiment_scores: ", sentiment_scores)
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        sentiment_scores = torch.Tensor(sentiment_scores).type(dtype=torch.long)

        # print("input_ids: ", input_ids)
        # print("sentiment_scores: ", sentiment_scores)
        # print("sentiment_scores type: ", type(sentiment_scores))
        # print("sentiment_scores.dtype: ", sentiment_scores.dtype)

        return {'input_ids': input_ids,
                'labels': sentiment_scores}

# TODO: add SST5 data loading pipeline - end - 


def spearmanr(target, prediction):
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return scipy.stats.spearmanr(target_array, prediction_array).correlation
    
def evaluate(model, eval_iterator, do_mi=False, do_contrast_spearmanr=True, latent_space_type='plain', return_pred=False):
    eval_contrastive_loss_total = 0
    eval_same_label_loss_total = 0
    
    model.eval()
    num_eval_batch = 0
    
    contrast_preds=[]
    contrast_targs = []

    with torch.no_grad():
        for step, batch in enumerate(eval_iterator):
            
            input_ids = batch['input_ids'].to(model.device)
            contrast_targets = batch['labels'].to(model.device)
        
            model_outputs = model(input_ids, contrast_targets=contrast_targets, mask_similar_contrast_label=True, return_same_label_loss=True)
            contrastive_loss, contrastive_value = model_outputs[0], model_outputs[1]
            
            same_label_loss = model_outputs[2]
            eval_same_label_loss_total = eval_same_label_loss_total + same_label_loss

            for pred, target in zip(contrastive_value.squeeze().cpu().numpy(), contrast_targets.cpu().numpy()):
                contrast_targs.append(target)
                contrast_preds.append(pred)
            
            eval_contrastive_loss_total = eval_contrastive_loss_total + contrastive_loss

            num_eval_batch += 1

    eval_contrastive_loss = eval_contrastive_loss_total / num_eval_batch
    eval_same_label_loss = eval_same_label_loss_total / num_eval_batch
    eval_output = {
                "contrastive_loss": eval_contrastive_loss,
                "same_label_loss": eval_same_label_loss,
                  }

    if do_contrast_spearmanr:
        spearmanr_value = spearmanr(contrast_targs, contrast_preds)
        print("spearmanr_value: ", spearmanr_value)
        eval_output['spearmanr'] = spearmanr_value
    
    if return_pred:
        eval_output['contrast_preds'] = contrast_preds
        eval_output['contrast_targs'] = contrast_targs

    return eval_output

# TODO: add SST5 data loading pipeline - start - 
# dataset_path = "data/sst"
# datasets = read_sst5(dataset_path)
datasets = read_sst5(data_dir)

train_dataset = CustomTextDatasetForDisc(datasets['train'], tokenizer=tokenizer, split=None, omitted_labels=train_omitted_labels, reduced_labels=train_reduced_labels, reduced_labels_keep_num=reduced_labels_keep_num)
# train_dataset = CustomTextDatasetForDisc(datasets['train'], tokenizer=tokenizer, split=None, omitted_labels=train_omitted_labels)
eval_dataset = CustomTextDatasetForDisc(datasets['dev'], tokenizer=tokenizer, split=None)
if train_omitted_labels != None:
    eval_dataset_w_train_omitted_labels = CustomTextDatasetForDisc(datasets['dev'], tokenizer=tokenizer, split=None, omitted_labels=train_omitted_labels)
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
    {'params': [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay))], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay))], 'weight_decay': 0.0}
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

global_step = 0
n_iter = int(args.num_train_epochs * len(epoch_iterator))
# print("len(epoch_iterator): ", len(epoch_iterator))
# print("args.num_train_epochs: ", args.num_train_epochs)
print("n_iter: ", n_iter)

model.train()
for epoch in trange(1, args.num_train_epochs+1):
    for step, batch in enumerate(epoch_iterator):
        input_ids = batch['input_ids'].to(model.device)
        # print("input_ids.shape: ", input_ids.shape)
        # print("input_ids[:, 0]: ", input_ids[:, 0])
        # print("input_ids: ", input_ids)
        contrast_targets = batch['labels'].to(model.device)

        model.zero_grad()

        print("input_ids: ", input_ids)
        model_outputs = model(input_ids, contrast_targets=contrast_targets, mask_similar_contrast_label=True, return_same_label_loss=True)
            
        contrastive_loss, contrastive_value = model_outputs[0], model_outputs[1]

        total_loss = contrastive_loss

        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        global_step += 1
        
        if global_step % args.logging_steps == 0:
            tb_writer.add_scalar("TRAIN/lr", lr_scheduler.get_last_lr()[0], global_step)
            tb_writer.add_scalar("TRAIN/contrastive_loss", contrastive_loss, global_step)
                
        if global_step % args.eval_steps == 0:
            model.eval()
            eval_output = evaluate(model, eval_iterator)
            eval_contrastive_loss, eval_spearmanr = eval_output['contrastive_loss'], eval_output['spearmanr']

            tb_writer.add_scalar("EVAL/contrastive_loss", eval_contrastive_loss, global_step)
            tb_writer.add_scalar("EVAL/spearmanr", eval_spearmanr, global_step)

            if train_omitted_labels != None:
                eval_output_w_train_omitted_labels = evaluate(model, eval_iterator_w_train_omitted_labels)
                eval_contrastive_loss_w_train_omitted_labels, eval_spearmanr_w_train_omitted_labels = eval_output_w_train_omitted_labels['contrastive_loss'], eval_output_w_train_omitted_labels['spearmanr']

                tb_writer.add_scalar("EVAL/contrastive_loss_w_train_omitted_labels", eval_contrastive_loss_w_train_omitted_labels, global_step)
                tb_writer.add_scalar("EVAL/spearmanr_w_train_omitted_labels", eval_spearmanr_w_train_omitted_labels, global_step)
            
            model.train()

# Final log step
tb_writer.add_scalar("TRAIN/lr", lr_scheduler.get_last_lr()[0], global_step)
tb_writer.add_scalar("TRAIN/contrastive_loss", contrastive_loss, global_step)

# Final evaluation
model.eval()
eval_output = evaluate(model, eval_iterator)
eval_contrastive_loss, eval_spearmanr = eval_output['contrastive_loss'], eval_output['spearmanr']

tb_writer.add_scalar("EVAL/contrastive_loss", eval_contrastive_loss, global_step)
tb_writer.add_scalar("EVAL/spearmanr", eval_spearmanr, global_step)

if train_omitted_labels != None:
    eval_output_w_train_omitted_labels = evaluate(model, eval_iterator_w_train_omitted_labels)
    eval_contrastive_loss_w_train_omitted_labels, eval_spearmanr_w_train_omitted_labels = eval_output_w_train_omitted_labels['contrastive_loss'], eval_output_w_train_omitted_labels['spearmanr']

    tb_writer.add_scalar("EVAL/contrastive_loss_w_train_omitted_labels", eval_contrastive_loss_w_train_omitted_labels, global_step)
    tb_writer.add_scalar("EVAL/spearmanr_w_train_omitted_labels", eval_spearmanr_w_train_omitted_labels, global_step)

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
