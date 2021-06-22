import torch
from transformers import MT5ForConditionalGeneration, MT5Config, MT5EncoderModel, MT5Tokenizer, Trainer, TrainingArguments
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



# argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--seed', action='store', type=int, default=30, help='random seed')
parser.add_argument('--data_dir', action='store', type=str, help='input df filename', default="/export/share/alvinchan/data/ACE/data/gen_train_data/top_half_ddG" )
parser.add_argument('--pretrained_dir', action='store', type=str, help='dir path for pretrained progeny weights', default="/export/share/bkrause/progen/progeny/t5_base_uniref_bfd50/" )

parser.add_argument('--output_dir', action='store', type=str, default="./trainer_results", help='input df filename' )
parser.add_argument('--num_train_epochs', action='store', type=int, default=3)
parser.add_argument('--per_device_train_batch_size', action='store', type=int, default=16)
parser.add_argument('--per_device_eval_batch_size', action='store', type=int, default=64)
parser.add_argument('--warmup_steps', action='store', type=int, default=500)
parser.add_argument('--weight_decay', action='store', type=float, default=0.01)
parser.add_argument('--logging_dir', action='store', type=str, default="./trainer_logs" )
parser.add_argument('--save_total_limit', action='store', type=int, default=2)
parser.add_argument('--save_steps', action='store', type=int, default=2000)

parser.add_argument('--lr', action='store', type=float, default=5e-05, help='learning rate')
parser.add_argument('--train_ratio', action='store', type=float, default=1.0)
parser.add_argument('--train_split_name', action='store', type=str, default="train" )
parser.add_argument('--eval_split_name', action='store', type=str, default="valid" )

args = parser.parse_args()

print("args: ", args)

seed = args.seed
data_dir = args.data_dir
pretrained_dir = args.pretrained_dir
train_ratio = args.train_ratio
train_split_name = args.train_split_name
eval_split_name = args.eval_split_name

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

tokenizer = TAPETokenizer(vocab="progeny")

device = torch.device('cuda:0')

t5config = MT5Config.from_pretrained(pretrained_dir)
model = MT5ForConditionalGeneration.from_pretrained(pretrained_dir)

model.parallelize()


class PKLDFDatasetForGen(Dataset):
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
                train_data_file: str = '250K_ddG_split/train_ddG.pkl',
                ):

        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)
        
        df = pd.read_pickle(data_file)
        
        if train_ratio != 1:
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
            item['input_ids'] = ""
            item['labels'] = row['MT_seq']

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

class CustomStabilityDatasetForGen(Dataset):

    def __init__(self,
                data_path: typing.Union[str, Path],
                split: str,
                tokenizer: typing.Union[str, TAPETokenizer] = 'iupac',
                in_memory: bool = False,
                train_ratio: float = 1,
                normalize_targets: bool = False):

        # if split not in ('train', 'valid', 'test'):
        #     raise ValueError(f"Unrecognized split: {split}. "
        #                     f"Must be one of ['train', 'valid', 'test']")
        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        if split == 'valid':
            file_prefix = 'train'
        else:
            file_prefix = split
            
        data_path = Path(data_path)
        data_file = f'{file_prefix}_ddG.pkl' 

        self.data = PKLDFDatasetForGen(data_path / data_file, in_memory)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        input_ids = self.tokenizer.encode(item['input_ids']) 
        labels = self.tokenizer.encode(item['labels']) 
        return input_ids, labels 

    
    def collate_fn(self, batch: typing.List[typing.Tuple[typing.Any, ...]]) -> typing.Dict[str, torch.Tensor]:
        input_ids, labels = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        labels = torch.from_numpy(pad_sequences(labels, 0))

        return {'input_ids': input_ids,
                'labels': labels}

train_dataset = CustomStabilityDatasetForGen(data_dir, train_split_name, train_ratio=train_ratio, tokenizer=tokenizer)
eval_dataset = CustomStabilityDatasetForGen(data_dir, eval_split_name, train_ratio=train_ratio, tokenizer=tokenizer)


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
    data_collator=train_dataset.collate_fn
)

trainer.train()

trainer.save_model()

trainer.evaluate()