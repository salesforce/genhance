import torch
# from transformers import MT5ForConditionalGeneration, MT5Config, MT5EncoderModel, MT5Tokenizer, Trainer, TrainingArguments
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
from collections import OrderedDict
import os
import pickle
from tqdm import tqdm

# from modeling_progeny import ProgenyForSequenceToSequenceClassification, ProgenyForValuePrediction, ProgenyForSequenceClassification, ProgenyForContactPrediction, ProgenyConfig
from transformers_custom import T5ForConditionalGeneration, T5ForConditionalGenerationWithLatentSpace, T5Discriminator, T5Tokenizer, T5Config, BertTokenizer, BertForSequenceClassification, GPT2LMHeadModel, GPT2TokenizerFast

# argparse 
parser = argparse.ArgumentParser()

parser.add_argument('--seed', action='store', type=int, default=30, help='random seed')
parser.add_argument('--num_generations', action='store', type=int, default=20000, help='(min) number of generation')
parser.add_argument('--generation_output_dir', action='store', type=str, default="generated_seqs/" )
parser.add_argument('--prepend_output_name', action='store', type=str, default="" )
parser.add_argument('--gen_pretrained_dir', action='store', type=str, default="gen/tophalf_12ep/results/checkpoint-92000" )
parser.add_argument('--tokenizer_pretrained_dir', action='store', type=str, default="t5-small" )
parser.add_argument('--input_seq', action='store', type=str, default="" )
parser.add_argument('--temperature_init', action='store', type=float, default=1.0)
parser.add_argument('--temperature_multiple', action='store', type=float, default=1.2)
parser.add_argument('--patience', action='store', type=int, default=50, help='number of repeats before increasing temperature values for gen decoding')
parser.add_argument('--batch_repeat_threshold', action='store', type=int, default=4)
parser.add_argument('--gen_batch_size', action='store', type=int, default=800)
parser.add_argument('--gen_save_interval', action='store', type=int, default=1000, help='interval to save generations')
parser.add_argument('--train_data_dir', action='store', type=str, default="data/sst", help='data for generator input seqs' )
parser.add_argument('--skip_gen', action='store_true')

# discriminator args
parser.add_argument('--disc_batch_size', action='store', type=int, default=1000)
parser.add_argument('--disc_save_interval', action='store', type=int, default=30)
parser.add_argument('--disc_pretrained_dir', action='store', type=str, default="/export/share/alvinchan/models/SST5/disc/SST5_discT5base_lre-04_25ep" )
parser.add_argument('--disc_latent_pooler', action='store', type=str, default="mean", choices=['mean', 'max', 'cls'], help='op to pool encoder hidden states' )

# GT model args
parser.add_argument('--gt_batch_size', action='store', type=int, default=1000)
parser.add_argument('--gt_tokenizer_pretrained_dir', action='store', type=str, default="bert-large-uncased" )
parser.add_argument('--gt_pretrained_dir', action='store', type=str, default="/export/share/alvinchan/models/SST5/disc/SST5_clsBERTlarge_lre-05_30ep_bs32" )
parser.add_argument('--gt_save_interval', action='store', type=int, default=30, help='interval to save generations')

# PPL model args
parser.add_argument('--ppl_model_id', action='store', type=str, default="gpt2-large" )

# SST5 args
# parser.add_argument('--gen_input_labels', nargs='+', help='Labels of samples to use for generation input seqs, labels are 0: strongly neg, 1: neg, 2: neutral, 3: pos, 4: strongly pos')
parser.add_argument('--prepended_cls_token', action='store', type=str, default="<extra_id_0>" )

args = parser.parse_args()

print("args: ", args)

seed = args.seed
num_generations = args.num_generations
gen_save_interval = args.gen_save_interval
generation_output_dir = args.generation_output_dir
prepend_output_name = args.prepend_output_name
gen_pretrained_dir = args.gen_pretrained_dir
tokenizer_pretrained_dir = args.tokenizer_pretrained_dir

tokenizer_pretrained_dir = args.tokenizer_pretrained_dir
# gen_input_labels = args.gen_input_labels
prepended_cls_token = args.prepended_cls_token

input_seq = args.input_seq
temperature_init = args.temperature_init
temperature_multiple = args.temperature_multiple
patience = args.patience
batch_repeat_threshold = args.batch_repeat_threshold
gen_batch_size = args.gen_batch_size
disc_batch_size = args.disc_batch_size
disc_save_interval = args.disc_save_interval
disc_pretrained_dir = args.disc_pretrained_dir
train_data_dir = args.train_data_dir

gt_batch_size = args.gt_batch_size
gt_tokenizer_pretrained_dir = args.gt_tokenizer_pretrained_dir
gt_pretrained_dir = args.gt_pretrained_dir
gt_save_interval = args.gt_save_interval

ppl_model_id = args.ppl_model_id

os.makedirs(generation_output_dir, exist_ok = True)

# wt_seq = 'STIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITEENVQNMNNAGDKWSAFLKEQSTLAQMYPLQEIQNLTVKLQLQALQ'
# constant_region = 'NTNITEEN'

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)


tokenizer = T5Tokenizer.from_pretrained(tokenizer_pretrained_dir)
# tokenizer = TAPETokenizer(vocab="progeny")

device = torch.device('cuda:0')

# t5config = MT5Config.from_pretrained(gen_pretrained_dir)
# gen_model = MT5ForConditionalGeneration.from_pretrained(gen_pretrained_dir)
gen_model = T5ForConditionalGeneration.from_pretrained(gen_pretrained_dir)

gen_model.parallelize()

input_ids = tokenizer.encode(input_seq)
# print("A input_ids: ", input_ids)
input_ids = np.array(input_ids, np.int64)
# print("B input_ids: ", input_ids)
input_ids = torch.from_numpy(input_ids).to(gen_model.device).unsqueeze(0)
# print("C input_ids: ", input_ids)

batch_input_ids = torch.cat([input_ids for i in range(gen_batch_size)], dim=0)

# Set up train data - start -
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

datasets = read_sst5(train_data_dir)
input_data_df = datasets['train']
train_seq_list = input_data_df['text'].tolist()

if prepended_cls_token is not None:
    prepended_cls_token_id = tokenizer.encode(prepended_cls_token)[0]
else:
    prepended_cls_token_id = None
    

# new gen
output_seq_list = []
output_tensor_list = []
temperature = temperature_init

start_time = time.time()
prev_save_path = None

repeat_event_count = 0
prev_log_interval_ind = 0
num_generated = 0

if not args.skip_gen:
    gen_model.eval()
    with torch.no_grad():
        while num_generated < num_generations:   
            if repeat_event_count > patience:
                temperature = float(temperature * temperature_multiple)
                print("Increased temperature to: ", temperature)
                repeat_event_count = 0

            gen_output = gen_model.generate(batch_input_ids, max_length=85+1, do_sample=True, temperature=temperature)

            batch_repeat_count = 0
            for seq_ind, gen_seq in enumerate(gen_output.cpu().numpy()):
                str_token_seq = tokenizer.decode(gen_seq.tolist(), skip_special_tokens=True )

                if str_token_seq in output_seq_list or str_token_seq in train_seq_list:
                # if str_token_seq in output_seq_list:
                    batch_repeat_count += 1
                else:
                    seq_tensor = gen_output[seq_ind].detach().cpu()
                    output_tensor_list.append(seq_tensor)
                    output_seq_list.append(str_token_seq)

            if batch_repeat_count >= batch_repeat_threshold:
                repeat_event_count += 1

            num_generated = len(output_seq_list)
            log_interval_ind = num_generated // gen_save_interval

            if log_interval_ind > prev_log_interval_ind:
                save_path = os.path.join(generation_output_dir, "{}gen_seqs{}-{}.pkl".format(prepend_output_name, num_generated, num_generations))
                saved_dict = {'output_seq_list': output_seq_list, "output_tensor_list": output_tensor_list, 'temperature': temperature}
                with open(save_path, 'wb') as f:
                    pickle.dump(saved_dict, f)
                print("generated #", num_generated)
                cur_time = time.time()
                print("Time taken so far:", cur_time - start_time)

                if prev_save_path is not None:
                    os.remove(prev_save_path)
                prev_save_path = save_path
                prev_log_interval_ind = log_interval_ind

    save_path = os.path.join(generation_output_dir, "{}gen_seqs_full{}.pkl".format(prepend_output_name, num_generations))
    saved_dict = {'output_seq_list': output_seq_list, "output_tensor_list": output_tensor_list, 'temperature': temperature}
    with open(save_path, 'wb') as f:
        pickle.dump(saved_dict, f)

else:
    print("Skipping generation step and loading from saved pkl")
    save_path = os.path.join(generation_output_dir, "{}gen_seqs_full{}.pkl".format(prepend_output_name, num_generations))
    with open(save_path, 'rb') as f:
        saved_dict = pickle.load(f)
    output_seq_list = saved_dict['output_seq_list']
    output_tensor_list = saved_dict['output_tensor_list']
    temperature = saved_dict['temperature']

if prev_save_path is not None:
    os.remove(prev_save_path)

gen_tensors = output_tensor_list
# gen_tensors = torch.stack(output_tensor_list, dim=0)
# new gen


# Discriminator inference
# TODO: Set up discriminator model - start -
t5config = T5Config.from_pretrained(disc_pretrained_dir)
disc_args = {
    'latent_pooler': args.disc_latent_pooler,
}

disc_model = T5Discriminator.from_pretrained(disc_pretrained_dir, **disc_args)

disc_model.eval()

disc_model = disc_model.to(gen_model.device)

# t5config = MT5Config.from_pretrained(disc_pretrained_dir)
# config = ProgenyConfig.from_pretrained(disc_pretrained_dir)

# disc_model = ProgenyForValuePrediction.from_pretrained(disc_pretrained_dir, config=config, t5config=t5config, predict_head='contrastive')
# disc_model.eval()

# disc_model = disc_model.to(gen_model.device)
# TODO: Set up discriminator model - end -


# new disc
# more positive values mean less stable, more negative values mean more stable
disc_pred_list = []
prev_save_path = None

num_disc_batch = len(gen_tensors) // disc_batch_size
if len(gen_tensors) % disc_batch_size != 0:
    num_disc_batch += 1

start_time = time.time()
with torch.no_grad():
    for batch_ind in tqdm(range(num_disc_batch)):
        gen_tensor_batch = gen_tensors[batch_ind*disc_batch_size : (batch_ind+1)*disc_batch_size]
         
        gen_tensor_batch = torch.nn.utils.rnn.pad_sequence(gen_tensor_batch, batch_first=True, padding_value=0)
        # print("A gen_tensor_batch: ", gen_tensor_batch)
        gen_tensor_batch = gen_tensor_batch[:, 1:]
        
        gen_tensor_batch = gen_tensor_batch.to(gen_model.device)
        # print("B gen_tensor_batch: ", gen_tensor_batch)
        # print("B gen_tensor_batch.shape: ", gen_tensor_batch.shape)

        # Add cls (32099) token at the front before inference!
        if prepended_cls_token_id is not None:
            cls_tensor = torch.full(size=[gen_tensor_batch.shape[0], 1], fill_value=prepended_cls_token_id, dtype=gen_tensor_batch.dtype, device=gen_tensor_batch.device)
            disc_input_batch = torch.cat([ cls_tensor, gen_tensor_batch ], dim=1)
            # print("disc_input_batch: ", disc_input_batch)
            # print("disc_input_batch.shape: ", disc_input_batch.shape)
        else:
            disc_input_batch = gen_tensor_batch

        disc_output = disc_model(disc_input_batch)
        disc_pred_list.append(disc_output[0].cpu().numpy())
        
        if batch_ind % disc_save_interval == 0:
            print("inferred #", (batch_ind+1)*disc_batch_size)
            cur_time = time.time()

            save_path = os.path.join(generation_output_dir, "{}disc_{}-{}.pkl".format(prepend_output_name, (batch_ind+1)*disc_batch_size, num_generations))
            with open(save_path, 'wb') as f:
                pickle.dump(disc_pred_list, f)
            cur_time = time.time()
            print("Time taken so far:", cur_time - start_time)

            if prev_save_path is not None:
                os.remove(prev_save_path)
            prev_save_path = save_path

disc_pred_list = np.concatenate(disc_pred_list, axis=None).tolist()

save_path = os.path.join(generation_output_dir, "{}disc_full{}.pkl".format(prepend_output_name, num_generations))
with open(save_path, 'wb') as f:
    pickle.dump(disc_pred_list, f)

if prev_save_path is not None:
    os.remove(prev_save_path)
# new disc



# TODO: new args: gt_save_interval
# gt_batch_size, gt_tokenizer_pretrained_dir, gt_pretrained_dir

# TODO: Ground-Truth classifier inference - start -
# Ground-Truth model set up - Start -
gt_tokenizer = BertTokenizer.from_pretrained(gt_tokenizer_pretrained_dir)
gt_model = BertForSequenceClassification.from_pretrained(gt_pretrained_dir, num_labels=5)

gt_model.eval()

gt_model = gt_model.to(gen_model.device)

# free up GPU memory
del gen_model
del disc_model
# Ground-Truth model set up - End -

# Ground-Truth model inference
gt_pred_list = []
gt_class_pred_list = []
gt_highest_prob_list = []
gt_neg_prob_list = []
gt_pos_prob_list = []
gt_2class_pred_list = []
prev_save_path = None

num_gt_batch = len(output_seq_list) // gt_batch_size
if len(output_seq_list) % gt_batch_size != 0:
    num_gt_batch += 1

start_time = time.time()

with torch.no_grad():
    for batch_ind in tqdm(range(num_gt_batch)):

        # TODO: Process input batch - start -
        gen_seq_batch = output_seq_list[batch_ind*gt_batch_size : (batch_ind+1)*gt_batch_size]

        batch_input_ids = []
        # tokenize
        for seq in gen_seq_batch:
            # print("seq: ", seq)
            input_ids = gt_tokenizer.encode(seq)
            input_ids = np.array(input_ids, np.int64)
            batch_input_ids.append(input_ids)

        # collate
        batch_input_ids = torch.from_numpy(pad_sequences(batch_input_ids, 0)).to(gt_model.device)
            
        # TODO: Process input batch - end -
        gt_output = gt_model(input_ids=batch_input_ids)
        gt_pred_list.append(gt_output.logits.cpu().numpy())
        # gt_class_pred = torch.argmax(gt_output.logits, dim=1)
        gt_class_probs = torch.nn.functional.softmax(gt_output.logits, dim=1)
        gt_highest_prob, gt_class_pred = torch.max(gt_class_probs, dim=1)
        gt_neg_prob = torch.sum(gt_class_probs[:, [0,1]], dim=1)
        gt_pos_prob = torch.sum(gt_class_probs[:, [3,4]], dim=1)
        gt_2class_pred = (gt_pos_prob > gt_neg_prob).int()

        gt_class_pred_list.append(gt_class_pred.cpu().numpy())
        gt_highest_prob_list.append(gt_highest_prob.cpu().numpy())
        gt_neg_prob_list.append(gt_neg_prob.cpu().numpy())
        gt_pos_prob_list.append(gt_pos_prob.cpu().numpy())
        gt_2class_pred_list.append(gt_2class_pred.cpu().numpy())
        
        if batch_ind % gt_save_interval == 0:
            print("inferred #", (batch_ind+1)*gt_batch_size)
            cur_time = time.time()

            save_path = os.path.join(generation_output_dir, "{}gt_{}-{}.pkl".format(prepend_output_name, (batch_ind+1)*gt_batch_size, num_generations))
            with open(save_path, 'wb') as f:
                pickle.dump(gt_pred_list, f)
            cur_time = time.time()
            print("Time taken so far:", cur_time - start_time)

            if prev_save_path is not None:
                os.remove(prev_save_path)
            prev_save_path = save_path

gt_pred_list = np.concatenate(gt_pred_list, axis=0)
gt_class_pred_list = np.concatenate(gt_class_pred_list, axis=None).tolist()
gt_highest_prob_list = np.concatenate(gt_highest_prob_list, axis=None).tolist()
gt_neg_prob_list = np.concatenate(gt_neg_prob_list, axis=None).tolist()
gt_pos_prob_list = np.concatenate(gt_pos_prob_list, axis=None).tolist()
gt_2class_pred_list = np.concatenate(gt_2class_pred_list, axis=None).tolist()

save_path = os.path.join(generation_output_dir, "{}gt_pred_full{}.pkl".format(prepend_output_name, num_generations))
with open(save_path, 'wb') as f:
    pickle.dump(gt_pred_list, f)

if prev_save_path is not None:
    os.remove(prev_save_path)


# TODO:Ground-Truth classifier inference - end -



# PPL computation with GPT-2 - start -
ppl_batch_size = 1 # only works with batch size 1 now
ppl_model = GPT2LMHeadModel.from_pretrained(ppl_model_id).to(gt_model.device)
ppl_tokenizer = GPT2TokenizerFast.from_pretrained(ppl_model_id)
gen_seq_ppl_list = []

del gt_model

num_ppl_batch = len(output_seq_list) // ppl_batch_size
if len(output_seq_list) % ppl_batch_size != 0:
    num_ppl_batch += 1

start_time = time.time()

print("PPL compute for generated sequences")
with torch.no_grad():
    for batch_ind in tqdm(range(num_ppl_batch)):

        # TODO: Process input batch - start -
        gen_seq_batch = output_seq_list[batch_ind*ppl_batch_size : (batch_ind+1)*ppl_batch_size]

        batch_input_ids = []
        # tokenize
        for seq in gen_seq_batch:
            input_ids = ppl_tokenizer.encode(seq)
            input_ids = np.array(input_ids, np.int64)
            batch_input_ids.append(input_ids)

        # collate
        batch_input_ids = torch.from_numpy(pad_sequences(batch_input_ids, 0)).to(ppl_model.device)
        
        if batch_input_ids.shape[1] == 0:
            gen_seq_ppl_list.append(None)
        else:
            ppl_output = ppl_model(input_ids=batch_input_ids, labels=batch_input_ids)
            log_likelihood = ppl_output[0]
            seq_ppl = torch.exp(log_likelihood)
            gen_seq_ppl_list.append(seq_ppl.cpu().numpy())

gen_seq_ppl_list = np.concatenate(gen_seq_ppl_list, axis=None).tolist()

# PPL computation with GPT-2 - end - 


# Save generated samples into TSV file
# PDB, Chain, Start_index, WT_seq, MT_seq
# PDB = 'template2.pdb'
# Chain = 'A'
# Start_index = 19
# WT_seq = 'STIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITEENVQNMNNAGDKWSAFLKEQSTLAQMYPLQEIQNLTVKLQLQALQ'

df = pd.DataFrame()

df['disc_pred'] = disc_pred_list
df['gt_class_pred'] = gt_class_pred_list
df['gt_highest_prob'] = gt_highest_prob_list

df['gt_2class_pred'] = gt_2class_pred_list
df['gt_neg_prob'] = gt_neg_prob_list
df['gt_pos_prob'] = gt_pos_prob_list

df['generated_seq_ppl'] = gen_seq_ppl_list
df['generated_seq'] = output_seq_list

# Disc-predicted most stable ones first
df = df.sort_values(by='disc_pred', ascending=False)

tsv_name = os.path.join(generation_output_dir, "{}basegen_seqs{}.tsv".format(prepend_output_name, num_generations))

df.to_csv(tsv_name, sep="\t", index=False)