import torch

# from transformers import Trainer, TrainingArguments
from transformers_custom import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

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
import os

from datasets import load_metric

# argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--seed', action='store', type=int, default=30, help='random seed')
parser.add_argument('--data_dir', action='store', type=str, help='input df filename', default="data/sst" )
parser.add_argument('--pretrained_dir', action='store', type=str, help='name or dir path for pretrained LM weights', default="bert-base-uncased" )

parser.add_argument('--output_dir', action='store', type=str, default="./sst5_disc_results", help='input df filename' )
parser.add_argument('--num_train_epochs', action='store', type=int, default=3)
parser.add_argument('--per_device_train_batch_size', action='store', type=int, default=16)
parser.add_argument('--per_device_eval_batch_size', action='store', type=int, default=64)
parser.add_argument('--warmup_steps', action='store', type=int, default=500)
parser.add_argument('--weight_decay', action='store', type=float, default=0.01)
parser.add_argument('--logging_dir', action='store', type=str, default=None )
parser.add_argument('--save_total_limit', action='store', type=int, default=1)
parser.add_argument('--save_steps', action='store', type=int, default=2000)

parser.add_argument('--lr', action='store', type=float, default=5e-05, help='learning rate')
parser.add_argument('--train_ratio', action='store', type=float, default=1.0)
parser.add_argument('--train_split_name', action='store', type=str, default="train" )
parser.add_argument('--eval_split_name', action='store', type=str, default="valid" )

# SST5 args
parser.add_argument('--train_omitted_labels', nargs='+', help='Labels to omit in training phase, labels are 0: strongly neg, 1: neg, 2: neutral, 3: pos, 4: strongly pos')

args = parser.parse_args()

print("args: ", args)

seed = args.seed
data_dir = args.data_dir
pretrained_dir = args.pretrained_dir
train_ratio = args.train_ratio
train_split_name = args.train_split_name
eval_split_name = args.eval_split_name

if args.logging_dir is None:
    args.logging_dir = args.output_dir

if args.train_omitted_labels is not None:
    train_omitted_labels = [int(train_omitted_label) for train_omitted_label in args.train_omitted_labels]
    num_labels = 5 - len(train_omitted_labels)
else:
    train_omitted_labels = None
    num_labels = 5
    
print("train_omitted_labels: ", train_omitted_labels)

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

tokenizer = BertTokenizer.from_pretrained(pretrained_dir)

device = torch.device('cuda:0')

# TODO: model setup
print("num_labels: ", num_labels)
model = BertForSequenceClassification.from_pretrained(pretrained_dir, num_labels=num_labels)

# model.parallelize()


# TODO: data loading setup - start -
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
                ):
        
        if omitted_labels is not None:
            df = df.loc[~df['truth'].isin(omitted_labels)]

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
                omitted_labels = None # list of label to omit from dataset
                ):

        self.tokenizer = tokenizer

        if split == 'valid':
            file_prefix = 'train'
        else:
            file_prefix = split

        self.data = TextDFDatasetForDisc(df, in_memory, split, train_ratio, omitted_labels=omitted_labels)
        self.omitted_labels = omitted_labels

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        input_ids = self.tokenizer.encode(item['input_ids'])
        input_ids = np.array(input_ids, np.int64)

        sentiment_scores = item['sentiment_scores']

        return input_ids, sentiment_scores

    
    def collate_fn(self, batch: typing.List[typing.Tuple[typing.Any, ...]]) -> typing.Dict[str, torch.Tensor]:
        input_ids, sentiment_scores = tuple(zip(*batch))
        # print("input_ids: ", input_ids)
#         print("labels: ", labels)
        # print("sentiment_scores: ", sentiment_scores)
        # print("input_ids len: ", len(input_ids))
        # print("sentiment_scores len: ", len(sentiment_scores))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        sentiment_scores = torch.Tensor(sentiment_scores).type(dtype=torch.long)

        # print("input_ids: ", input_ids)
        # print("sentiment_scores: ", sentiment_scores)
        # print("sentiment_scores type: ", type(sentiment_scores))
        # print("sentiment_scores.dtype: ", sentiment_scores.dtype)

        return {'input_ids': input_ids,
                'labels': sentiment_scores}

# TODO: data loading setup - end -

# TODO: Eval code
def spearmanr(target, prediction):
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    # print("target_array[:10]: ", target_array[:10])
    # print("prediction_array[:10]: ", prediction_array[:10])
    # print("target_array.shape: ", target_array.shape)
    # print("prediction_array.shape: ", prediction_array.shape)
    return scipy.stats.spearmanr(target_array, prediction_array).correlation

metric = load_metric('glue', 'sst2')
def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # compute pos_neg_acc
    predictions_tensor = torch.from_numpy(predictions)
    pred_probs = torch.nn.functional.softmax(predictions_tensor, dim=1)

    neg_probs = torch.sum(pred_probs[:, [0,1]], dim=1)
    pos_probs = torch.sum(pred_probs[:, [3,4]], dim=1)

    pred_2class = (pos_probs > neg_probs).int()
    labels_tensor = torch.from_numpy(labels)
    labels_2class = (labels_tensor > 2).int()
    not_neutral = (labels_tensor != 2).int()
    correct_pred = (pred_2class == labels_2class).int()

    not_neutral_n_correct = not_neutral * correct_pred

    num_neg_pos = torch.sum(not_neutral)
    neg_pos_acc = torch.sum(not_neutral_n_correct) / torch.sum(not_neutral)

    predictions = np.argmax(predictions, axis=1)

    metric_dict = metric.compute(predictions=predictions, references=labels)
    spearmanr_value = spearmanr(labels, predictions)

    # compute neutral acc

    prediction_class_tensor = torch.argmax(predictions_tensor, dim=1)
    print("prediction_class_tensor == 2: ", prediction_class_tensor == 2)
    pred_neutral = (prediction_class_tensor == 2).int()
    is_neutral = (labels_tensor == 2).int()
    is_neutral_n_correct = is_neutral * pred_neutral
    neutral_acc = torch.sum(is_neutral_n_correct) / torch.sum(is_neutral)
    print("torch.sum(is_neutral): ", torch.sum(is_neutral))
    print("torch.sum(is_neutral_n_correct): ", torch.sum(is_neutral_n_correct))
    
    metric_dict = {**metric_dict, "spearmanr": spearmanr_value, "neg_pos_acc": neg_pos_acc, "neutral_acc": neutral_acc}

    return metric_dict


datasets = read_sst5(data_dir)

train_dataset = CustomTextDatasetForDisc(datasets['train'], tokenizer=tokenizer, split=None, omitted_labels=train_omitted_labels)
eval_dataset = CustomTextDatasetForDisc(datasets['dev'], tokenizer=tokenizer, split=None)

# train_dataset = CustomStabilityDatasetForGen(data_dir, train_split_name, train_ratio=train_ratio, tokenizer=tokenizer)
# eval_dataset = CustomStabilityDatasetForGen(data_dir, eval_split_name, train_ratio=train_ratio, tokenizer=tokenizer)


training_args = TrainingArguments(
    output_dir=args.output_dir ,          # output directory
    num_train_epochs=args.num_train_epochs ,              # total # of training epochs
    per_device_train_batch_size=args.per_device_train_batch_size ,  # batch size per device during training
    per_device_eval_batch_size=args.per_device_eval_batch_size ,   # batch size for evaluation
    warmup_steps=args.warmup_steps ,                # number of warmup steps for learning rate scheduler
    weight_decay=args.weight_decay ,               # strength of weight decay
    logging_dir=args.logging_dir ,            # directory for storing logs
    save_total_limit=args.save_total_limit ,
    save_steps=args.save_steps,
    learning_rate=args.lr,
    evaluation_strategy='epoch' 
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=eval_dataset,            # evaluation dataset
    data_collator=train_dataset.collate_fn,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.save_model()

trainer.evaluate()