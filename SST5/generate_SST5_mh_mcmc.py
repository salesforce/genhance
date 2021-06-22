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
import shutil
import pickle
from tqdm import tqdm

# from modeling_progeny import ProgenyForSequenceToSequenceClassification, ProgenyForValuePrediction, ProgenyForSequenceClassification, ProgenyForContactPrediction, ProgenyConfig
# from transformers_custom import MT5ForConditionalGenerationWithLatentSpace
from transformers_custom import T5ForConditionalGenerationWithLatentSpace, T5ForConditionalGeneration, T5Discriminator, T5Tokenizer, T5Config, BertTokenizer, BertForSequenceClassification, GPT2LMHeadModel, GPT2TokenizerFast


# argparse 
parser = argparse.ArgumentParser()

parser.add_argument('--seed', action='store', type=int, default=30, help='random seed')
# parser.add_argument('--num_generations', action='store', type=int, default=None, help='(min) number of generation')
parser.add_argument('--generation_output_dir', action='store', type=str, default="generated_seqs/mcmc_SST5/top12500input20iter" )
parser.add_argument('--prepend_output_name', action='store', type=str, default="" )
parser.add_argument('--unique_gen', action='store_true')

parser.add_argument('--tokenizer_pretrained_dir', action='store', type=str, default="t5-small" )

# new controlled gen args
parser.add_argument('--input_data_dir', action='store', type=str, default="data/sst", help='data for generator input seqs' )
parser.add_argument('--input_data_subset', action='store', type=str, default="train", help='data subset for generator input seqs', choices=["train", "dev", "test"] )
# parser.add_argument('--input_data_dir', action='store', type=str, default="data/gen_train_data/top_half_ddG", help='data for generator input seqs' )
# parser.add_argument('--topk_as_input', action='store', type=int, default=12500, help='first K sequences to use input for generation')
parser.add_argument('--num_gen_inputs', action='store', type=int, default=None, help='first K sequences to use input for generation')

# discriminator args
parser.add_argument('--disc_batch_size', action='store', type=int, default=1000)
parser.add_argument('--disc_save_interval', action='store', type=int, default=30)
parser.add_argument('--disc_pretrained_dir', action='store', type=str, default="/export/share/alvinchan/models/SST5/disc/SST5_discT5base_lre-04_25ep" )
parser.add_argument('--disc_latent_pooler', action='store', type=str, default="mean", choices=['mean', 'max', 'cls'], help='op to pool encoder hidden states' )


# MCMC args
parser.add_argument('--boltz_constant', action='store', type=float, default=1.0)
parser.add_argument('--temperature', action='store', type=float, default=1.0)
parser.add_argument('--num_gen_rounds', action='store', type=int, default=1, help='how many rounds of evolutionary generation across the whole set of gen_input_df')
parser.add_argument('--trust_radius', action='store', type=float, default=0.3, help='as a ratio of original length')
# parser.add_argument('--trust_radius', action='store', type=int, default=18)
parser.add_argument('--num_evo_iters', action='store', type=int, default=20)
parser.add_argument('--num_last_iters_to_keep', action='store', type=int, default=None)

# SST5 args
parser.add_argument('--gen_input_labels', nargs='+', help='Labels of samples to use for generation input seqs, labels are 0: strongly neg, 1: neg, 2: neutral, 3: pos, 4: strongly pos')
parser.add_argument('--num_mut_token_per_iter', action='store', type=int, default=1)
parser.add_argument('--prepended_cls_token', action='store', type=str, default="<extra_id_0>" )

# GT model args
parser.add_argument('--gt_batch_size', action='store', type=int, default=1000)
parser.add_argument('--gt_tokenizer_pretrained_dir', action='store', type=str, default="bert-large-uncased" )
parser.add_argument('--gt_pretrained_dir', action='store', type=str, default="/export/share/alvinchan/models/SST5/disc/SST5_clsBERTlarge_lre-05_30ep_bs32" )
parser.add_argument('--gt_save_interval', action='store', type=int, default=30, help='interval to save generations')

# PPL model args
parser.add_argument('--ppl_model_id', action='store', type=str, default="gpt2-large" )

# T5 mutation args
parser.add_argument('--mut_type', action='store', type=str, default="random", choices=['random', 't5'] )
parser.add_argument('--t5_model_type', action='store', type=str, default="t5-base", choices=['t5-small', 't5-base'] )
parser.add_argument('--max_masked_span_len', action='store', type=int, default=1, help="how many tokens to mask and replace, use > 1 to include deletion")
parser.add_argument('--t5_gen_temp', action='store', type=float, default=1.0)
parser.add_argument('--t5_gen_top_k', action='store', type=int, default=50)


args = parser.parse_args()

print("args: ", args)

seed = args.seed
generation_output_dir = args.generation_output_dir
prepend_output_name = args.prepend_output_name
disc_pretrained_dir = args.disc_pretrained_dir
disc_batch_size = args.disc_batch_size
input_data_dir = args.input_data_dir
# topk_as_input = args.topk_as_input

input_data_subset = args.input_data_subset
num_gen_inputs = args.num_gen_inputs
tokenizer_pretrained_dir = args.tokenizer_pretrained_dir
prepended_cls_token = args.prepended_cls_token
gen_input_labels = args.gen_input_labels
num_mut_token_per_iter = args.num_mut_token_per_iter

# T5 args
mut_type = args.mut_type
t5_model_type = args.t5_model_type
max_masked_span_len = args.max_masked_span_len 
t5_gen_temp = args.t5_gen_temp
t5_gen_top_k = args.t5_gen_top_k

gt_batch_size = args.gt_batch_size
gt_tokenizer_pretrained_dir = args.gt_tokenizer_pretrained_dir
gt_pretrained_dir = args.gt_pretrained_dir
gt_save_interval = args.gt_save_interval

ppl_model_id = args.ppl_model_id


num_evo_iters = args.num_evo_iters
num_last_iters_to_keep = args.num_last_iters_to_keep
if num_last_iters_to_keep is not None and num_last_iters_to_keep > num_evo_iters:
    raise ValueError(f"num_last_iters_to_keep must be smaller than num_evo_iters but is {num_last_iters_to_keep}.")
num_gen_rounds = args.num_gen_rounds
k = args.boltz_constant
T = args.temperature
trust_radius = args.trust_radius

output_dir = Path(generation_output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)


# Discriminator inference
# Set up discriminator model - start -
disc_tokenizer = T5Tokenizer.from_pretrained(tokenizer_pretrained_dir)

t5config = T5Config.from_pretrained(disc_pretrained_dir)
disc_args = {
    'latent_pooler': args.disc_latent_pooler,
}
disc_model = T5Discriminator.from_pretrained(disc_pretrained_dir, **disc_args)

disc_model.eval()
disc_model.to("cuda")

if mut_type == 't5':
    t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_type)
    t5_model.eval()
    t5_model.to('cuda')

# disc_model = disc_model.to(gen_model.device)

# tokenizer = TAPETokenizer(vocab="progeny")

# t5config = MT5Config.from_pretrained(disc_pretrained_dir)
# config = ProgenyConfig.from_pretrained(disc_pretrained_dir)

# disc_model = ProgenyForValuePrediction.from_pretrained(disc_pretrained_dir, config=config, t5config=t5config, predict_head='contrastive')
# disc_model.eval()




# Set up input data - start -
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

datasets = read_sst5(input_data_dir)
input_data_df = datasets[input_data_subset]
# input_data_df = datasets['train']
print("gen_input_labels: ", gen_input_labels)
if gen_input_labels != None:
    gen_input_labels = [int(gen_input_label) for gen_input_label in gen_input_labels]
    gen_input_df = input_data_df.loc[input_data_df['truth'].isin(gen_input_labels)]
else:
    gen_input_df = input_data_df
print("gen_input_df len: ", len(gen_input_df))
if num_gen_inputs is None:
    num_gen_inputs = len(gen_input_df)
gen_input_df = gen_input_df.iloc[:num_gen_inputs]
train_seq_list = input_data_df['text'].tolist()


if prepended_cls_token is not None:
    prepended_cls_token_id = disc_tokenizer.encode(prepended_cls_token)[0]
else:
    prepended_cls_token_id = None
# Set up input data - end -

# # Set up input data
# input_data_path = Path(input_data_dir)
# input_data_file = f'train_ddG.pkl' 
# input_data_file = input_data_path / input_data_file
# input_data_df = pd.read_pickle(input_data_file)

# train_seq_list = input_data_df['MT_seq'].tolist()

# print("ddG stats of input data")
# print("min: ", np.min(input_data_df['ddG']))
# print("mean: ", np.mean(input_data_df['ddG']))
# print("median: ", np.median(input_data_df['ddG']))
# print("max: ", np.max(input_data_df['ddG']))

# ddG_sorted_input_df = input_data_df.sort_values(by='ddG', ascending=True)

# gen_input_df = ddG_sorted_input_df.iloc[:topk_as_input]



# mutation params for SST5
"""
first 3 ids are special tokens to avoid during substitution
{'<pad>': 0,
 '</s>': 1,
 '<unk>': 2,
 '▁': 3,
 'X': 4,
 '.': 5,
 ',': 6,
 """
special_token_ids = [0,1,2,5,6]
possible_letters = [i for i in range(disc_tokenizer.vocab_size) if i not in special_token_ids]




# wt_seq = "STIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITEENVQNMNNAGDKWSAFLKEQSTLAQMYPLQEIQNLTVKLQLQALQ"
# reserved_span = "NTNITEEN"
# reserved_span_start = 32
# constant_indexes = [i for i in range(reserved_span_start, reserved_span_start+len(reserved_span))]

# # possible indexes to mutate
# mutatable_indexes = [i for i in range(len(wt_seq)) if i not in constant_indexes]

# omitted_letters = ["C", "X", "Z", "J", "B", "O", "U"] # for ACE
# # omitted_letters = ["X", "Z", "J", "B", "O", "U"] # for SH3
# possible_letters = [chr(i).upper() for i in range(ord('a'),ord('z')+1) if chr(i).upper() not in omitted_letters]

# mutatable_len = len(wt_seq) - len(reserved_span)
# print("mutatable_len: ", mutatable_len)

# def levenshtein_dist(str1, str2):
#     i = 0
#     count = 0
    
#     if len(str1) != len(str2):
#         print("disc_tokenizer.decode(str1[1:]): ", disc_tokenizer.decode(str1[1:]))
#         print("disc_tokenizer.decode(str2[1:]): ", disc_tokenizer.decode(str2[1:]))
#     while(i < len(str1)):
#         if(str1[i] != str2[i]):
#             count += 1
#         i += 1
#     return count

def levenshtein_dist(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

# Start MCMC generations
num_batch = len(gen_input_df) // disc_batch_size
if len(gen_input_df) % disc_batch_size != 0:
    num_batch += 1

init_seq_fitness_list = []
full_trajectory_fitness_scores = []
full_trajectory_s_star_seqs = []
full_accept_or_not_arr = []
full_s_star_levenshtein_dists = []

with torch.no_grad():
    for batch_ind in tqdm(range(num_batch)):
        batch_seqs = gen_input_df[batch_ind*disc_batch_size : (batch_ind+1)*disc_batch_size]['text']
        batch_classes = gen_input_df[batch_ind*disc_batch_size : (batch_ind+1)*disc_batch_size]['truth'].tolist()
        # batch_seqs = gen_input_df[batch_ind*disc_batch_size : (batch_ind+1)*disc_batch_size]['MT_seq']

        batch_init_input_ids = []
        batch_input_seqs = []
        batch_num_trust_radius = []

        batch_s = []
        # encode seqs
        for seq in batch_seqs:
            batch_input_seqs = batch_input_seqs + [seq]
            input_ids = disc_tokenizer.encode(disc_tokenizer.decode(disc_tokenizer.encode(seq))) # decode first encoded ids to remove space before punctuations such as " ," and " ."
            # input_ids = tokenizer.encode(seq)

            seq_trust_radius = max(1, int(trust_radius*len(input_ids)))
            batch_num_trust_radius.append(seq_trust_radius)

            s_decoded = disc_tokenizer.decode(input_ids, skip_special_tokens=True )
            # s_decoded = disc_tokenizer.decode(input_ids)
            batch_s.append(s_decoded)
            
            if prepended_cls_token_id is not None:
                input_ids = [prepended_cls_token_id] + input_ids
                
            input_ids = np.array(input_ids, np.int64)
            batch_init_input_ids.append(input_ids)
            # input_ids = torch.from_numpy(input_ids).unsqueeze(0)
        
        batch_input_ids = torch.from_numpy(pad_sequences(batch_init_input_ids, 0))
        # batch_input_ids = torch.cat(batch_input_ids, dim=0)
        batch_input_ids = batch_input_ids.cuda()

        # if prepended_cls_token_id is not None:
        #     cls_tensor = torch.full(size=[batch_input_ids.shape[0], 1], fill_value=prepended_cls_token_id, dtype=batch_input_ids.dtype, device=batch_input_ids.device)
        #     disc_input_batch = torch.cat([ cls_tensor, batch_input_ids ], dim=1)
        #     print("disc_input_batch: ", disc_input_batch)
        #     print("disc_input_batch.shape: ", disc_input_batch.shape)
        # else:
        #     disc_input_batch = batch_input_ids

        # disc_output = disc_model(disc_input_batch)
        # print("batch_input_ids: ", batch_input_ids)
        disc_output = disc_model(batch_input_ids)
        init_seq_fitness = disc_output[0].cpu().numpy() # shape: [N, 1]
        init_seq_fitness_list.append(init_seq_fitness)

        # initialize trajectory fitness scores
        batch_trajectory_fitness_scores = np.copy(init_seq_fitness)

        # TODO: handle saving of variable length seqs
        batch_trajectory_s_star_seqs = np.expand_dims(np.array(batch_seqs), axis=-1)

        batch_accept_or_not_arr = None
        batch_s_star_levenshtein_dists = None
        s_fitness = np.squeeze(np.copy(init_seq_fitness), axis=1) # shape: [N]

        # compute evolutionary trajectory

        
        # batch_s = batch_seqs[:]
        # batch_mu_mut = np.random.uniform(low=1, high=2.5, size=len(batch_s))
        for iter_ind in tqdm(range(num_evo_iters)):
            
            # mutate s to get s_star
            batch_s_star = []
            batch_s_star_input_ids = []
            for s_ind, s in enumerate(batch_s):
                num_mutation = num_mut_token_per_iter # 1 to prevent overwriting
                # num_mutation = np.random.poisson(batch_mu_mut[s_ind]-1) + 1

                # s_input_ids = disc_tokenizer.encode(s) # decode first encoded ids to remove space before punctuations such as " ," and " ."
                s_input_ids = disc_tokenizer.encode(disc_tokenizer.decode(disc_tokenizer.encode(s))) # decode first encoded ids to remove space before punctuations such as " ," and " ."
                
                mutatable_indexes = [ i for i in range(len(s_input_ids)) if s_input_ids[i] not in special_token_ids]
                # print("disc_tokenizer.encode(disc_tokenizer.decode(disc_tokenizer.encode(''))): ", disc_tokenizer.encode(disc_tokenizer.decode(disc_tokenizer.encode(''))))
                # print("mutatable_indexes: ", mutatable_indexes)
                # print("s_input_ids: ", s_input_ids)
                # print("s_input_ids len: ", len(s_input_ids))
                indexes_to_mutate = random.sample(mutatable_indexes, num_mutation)


                if mut_type == 't5':
                    # TODO: Mutate target token with T5 masked generations
                    masked_span_length = random.randint(1, max_masked_span_len)
                    sentinent_token_ids = [i for i in range(32099, 32099-100, -1)] # [32099, 32098, ..]

                    # change mutation sites to sentinent tokens
                    mut_t5_input_ids = s_input_ids[:]
                    sentinent_token_ids_to_use = sorted(sentinent_token_ids[:len(indexes_to_mutate)]) # small id first ... 32098, 32099 to fill from the back
                    prev_i_mut = len(mut_t5_input_ids)
                    indexes_to_mutate_des = sorted(indexes_to_mutate, reverse=True)
                    actual_masked_span_len_list = []
                    for enu_ind, i_mut in enumerate(indexes_to_mutate_des): # enumerate through the larger indexes first
                        sentinent_token_id = sentinent_token_ids_to_use[enu_ind]
                        # print("sentinent_token_id: ", sentinent_token_id)
                        # print("A mut_t5_input_ids: ", mut_t5_input_ids)


                        actual_masked_span_len = min(masked_span_length, prev_i_mut-i_mut) # so that mask doesn't spill over to the end of seq or the masked position behind
                        # print("actual_masked_span_len: ", actual_masked_span_len)
                        actual_masked_span_len_list.append(actual_masked_span_len)
                        mut_t5_input_ids = mut_t5_input_ids[:i_mut] + [sentinent_token_id] + mut_t5_input_ids[i_mut+actual_masked_span_len:]
                        # print("B mut_t5_input_ids: ", mut_t5_input_ids)
                        # mut_t5_input_ids[i_mut] = sentinent_token_id
                        prev_i_mut = i_mut

                    # print("actual_masked_span_len_list: ", actual_masked_span_len_list)
                    mut_t5_input_ids = np.array(mut_t5_input_ids, np.int64)
                    mut_t5_input_ids = torch.from_numpy(mut_t5_input_ids).unsqueeze(0)
                    mut_t5_input_ids = mut_t5_input_ids.cuda()
                    
                    mut_outputs = t5_model.generate(mut_t5_input_ids, do_sample=True, temperature=t5_gen_temp, top_k=t5_gen_top_k)

                    # process T5 generated mutation, add generated span to s_star
                    s_star_input_ids = s_input_ids[:]
                    mut_outputs_list = mut_outputs.cpu().numpy()[0].tolist()
                    for enu_ind, sentinent_token_id in enumerate(sentinent_token_ids_to_use):
                        i_mut = indexes_to_mutate_des[enu_ind]
                        actual_masked_span_len = actual_masked_span_len_list[enu_ind]
                        # print("mut_outputs_list: ", mut_outputs_list)
                        if sentinent_token_id in mut_outputs_list:
                            cur_sentinent_output_seq_ind = mut_outputs_list.index(sentinent_token_id) # 32099
                        else:
                            continue

                        if sentinent_token_id-1 in mut_outputs_list:
                            next_sentinent_output_seq_ind = mut_outputs_list.index(sentinent_token_id-1) # 32098
                        else:
                            continue

                        # print("cur_sentinent_output_seq_ind: ", cur_sentinent_output_seq_ind)
                        # print("next_sentinent_output_seq_ind: ", next_sentinent_output_seq_ind)
                        s_mut_ids = mut_outputs_list[cur_sentinent_output_seq_ind+1:next_sentinent_output_seq_ind] # generated mut span
                        # print("s_mut_ids: ", s_mut_ids)
                        # s_mut_ids = mut_outputs_list[mut_outputs_list.index(32099)+1:mut_outputs_list.index(32098)]
                        # print("A s_star_input_ids: ", s_star_input_ids)
                        s_star_input_ids = s_star_input_ids[:i_mut] + s_mut_ids + s_star_input_ids[i_mut+actual_masked_span_len:]
                        # print("s_star_input_ids[:i_mut]: ", s_star_input_ids[:i_mut])
                        # print("s_star_input_ids[i_mut+actual_masked_span_len:]: ", s_star_input_ids[i_mut+actual_masked_span_len:])
                        # print("B s_star_input_ids: ", s_star_input_ids)

                elif mut_type == 'random':
                    # Mutate (random substitution) target token
                    s_star_input_ids = s_input_ids[:]
                    for i_mut in indexes_to_mutate:
                        current_token_id = s_star_input_ids[i_mut]

                        # Random substitution with a different token from the vocab
                        sub_token_id = current_token_id
                        while sub_token_id == current_token_id:
                            sub_token_id = random.sample(possible_letters, 1)[0]

                        s_star_input_ids[i_mut] = sub_token_id

                mut_seq = disc_tokenizer.decode(s_star_input_ids, skip_special_tokens=True )
                # print("mut_seq: ", mut_seq)
                # mut_seq = disc_tokenizer.decode(s_star_input_ids)
                batch_s_star.append(mut_seq)

                # prepend cls token for disc inference
                if prepended_cls_token_id is not None:
                    s_star_input_ids = [prepended_cls_token_id] + s_star_input_ids

                s_star_input_ids = np.array(s_star_input_ids, np.int64)
                batch_s_star_input_ids.append(s_star_input_ids)
            
            batch_input_ids_np = pad_sequences(batch_s_star_input_ids, 0)
            batch_input_ids_tensor = torch.from_numpy(batch_input_ids_np)
            batch_input_ids_tensor = batch_input_ids_tensor.cuda()

                # indexes_to_mutate = random.sample(mutatable_indexes, num_mutation)

                # mut_seq = list(s)
                # for i_mut in indexes_to_mutate:
                #     current_aa = s[i_mut]

                #     sub_aa = current_aa
                #     while sub_aa == current_aa:
                #         # sample an AA to subsitute current AA
                #         sub_aa = random.sample(possible_letters, 1)[0]

                #     mut_seq[i_mut] = sub_aa

                # mut_seq = "".join(mut_seq)
                # batch_s_star.append(mut_seq)

            # # encode s_star
            # batch_input_ids = []
            # for s_star in batch_s_star:
            #     input_ids = tokenizer.encode(s_star)
            #     input_ids = torch.from_numpy(input_ids).unsqueeze(0)
            #     batch_input_ids.append(input_ids)

            # batch_input_ids = torch.cat(batch_input_ids, dim=0)
            # batch_input_ids = batch_input_ids.cuda()

            # infer fitness of s_star
            # print("batch_input_ids_tensor: ", batch_input_ids_tensor)
            disc_output = disc_model(batch_input_ids_tensor)
            fitness_tensor = disc_output[0]
            s_star_fitness_arr = fitness_tensor.cpu().numpy() # shape: [N, 1]
            s_star_fitness = np.squeeze(s_star_fitness_arr, axis=1) # shape: [N]



            # add s_star's fitness and seq to trajectory fitness array
            batch_trajectory_fitness_scores = np.concatenate([batch_trajectory_fitness_scores, s_star_fitness_arr], axis=1)
            # print("batch_trajectory_fitness_scores.shape: ", batch_trajectory_fitness_scores.shape)


            # TODO: handle saving of variable length seqs
            iter_s_star_seqs = np.expand_dims(np.array(batch_s_star), axis=-1) # [N, 1]
            batch_trajectory_s_star_seqs = np.concatenate([batch_trajectory_s_star_seqs, iter_s_star_seqs], axis=1)
            # print("batch_trajectory_s_star_seqs.shape: ", batch_trajectory_s_star_seqs.shape)

            # compute s_star accept probability
            accept_prob = np.exp(-1*(s_star_fitness - s_fitness)/(k*T)) # more negative is more fit
            accept_prob[accept_prob > 1] = 1
            # print("B accept_prob: ", accept_prob)
            # print("B accept_prob.shape: ", accept_prob.shape)
            # print("np.random.rand(len(accept_prob)) < accept_prob: ", np.random.rand(len(accept_prob)) < accept_prob)
            accept_or_not = (np.random.rand(len(accept_prob)) < accept_prob).astype(int)
            # print("accept_or_not: ", accept_or_not)
            # print("accept_or_not.shape: ", accept_or_not.shape)

            # assign 0 probability if s_star is outside trust radius
            # print("batch_s_star_input_ids: ", batch_s_star_input_ids)
            # print("batch_init_input_ids: ", batch_init_input_ids)
            # print("batch_s: ", batch_s)
            s_star_levenshtein_dists = np.array(list(map(levenshtein_dist, batch_s_star_input_ids, batch_init_input_ids)))
            # s_star_levenshtein_dists = np.array(list(map(levenshtein_dist, batch_s_star)))
            if batch_s_star_levenshtein_dists is None:
                batch_s_star_levenshtein_dists = np.expand_dims(s_star_levenshtein_dists, axis=-1)
            else:
                batch_s_star_levenshtein_dists = np.concatenate([batch_s_star_levenshtein_dists, np.expand_dims(s_star_levenshtein_dists, axis=-1)], axis=1)

            # print("s_star_levenshtein_dists: ", s_star_levenshtein_dists)
            # TODO: new trust radius check for insertions/deletions
            num_trust_radius = np.array(batch_num_trust_radius, dtype=s_star_levenshtein_dists.dtype)
            within_trust_radius = (s_star_levenshtein_dists <= num_trust_radius).astype(int)
            accept_or_not = accept_or_not * within_trust_radius
            # print("within_trust_radius: ", within_trust_radius)
            # print("within_trust_radius.shape: ", within_trust_radius.shape)
            
            # log accept or not
            if batch_accept_or_not_arr is None:
                batch_accept_or_not_arr = np.expand_dims(accept_or_not, axis=-1)
            else:
                batch_accept_or_not_arr = np.concatenate([batch_accept_or_not_arr, np.expand_dims(accept_or_not, axis=-1)], axis=1)
            # print("batch_accept_or_not_arr.shape: ", batch_accept_or_not_arr.shape)
                    
            # replace s with s_star if applicable
            accept_or_not_boolean = accept_or_not == 1
            reject_or_not_boolean = accept_or_not != 1
            # print("accept_or_not: ", accept_or_not)
            # print("accept_or_not_boolean: ", accept_or_not_boolean)

            # s_fitness, s_star_fitness
            s_fitness_new = np.select(condlist=[accept_or_not_boolean, reject_or_not_boolean], choicelist=[s_star_fitness, s_fitness])
            # print("s_star_fitness: ", s_star_fitness)
            # print("s_fitness: ", s_fitness)
            # print("s_fitness_new: ", s_fitness_new)
            s_fitness = s_fitness_new

            # batch_s, batch_s_star
            batch_s_new = np.select(condlist=[accept_or_not_boolean, reject_or_not_boolean], choicelist=[np.array(batch_s_star), np.array(batch_s)])
            batch_s = batch_s_new

        
        full_trajectory_fitness_scores.append(batch_trajectory_fitness_scores)
        full_trajectory_s_star_seqs.append(batch_trajectory_s_star_seqs)
        full_accept_or_not_arr.append(batch_accept_or_not_arr)
        full_s_star_levenshtein_dists.append(batch_s_star_levenshtein_dists)

full_trajectory_fitness_scores = np.concatenate(full_trajectory_fitness_scores, axis=0) # shape: [topk_as_input, num_evo_iters+1], +1 to include initial seqs
full_trajectory_s_star_seqs = np.concatenate(full_trajectory_s_star_seqs, axis=0) # shape: [topk_as_input, num_evo_iters+1], +1 to include initial seqs
full_accept_or_not_arr = np.concatenate(full_accept_or_not_arr, axis=0) # shape: [topk_as_input, num_evo_iters]
full_s_star_levenshtein_dists = np.concatenate(full_s_star_levenshtein_dists, axis=0) # shape: [topk_as_input, num_evo_iters]


print("full_trajectory_fitness_scores.shape: ", full_trajectory_fitness_scores.shape)
print("full_trajectory_s_star_seqs.shape: ", full_trajectory_s_star_seqs.shape)
print("full_accept_or_not_arr.shape: ", full_accept_or_not_arr.shape)
print("full_s_star_levenshtein_dists.shape: ", full_s_star_levenshtein_dists.shape)
if num_last_iters_to_keep is not None:
    full_trajectory_fitness_scores = full_trajectory_fitness_scores[:, -1*num_last_iters_to_keep:]
    full_trajectory_s_star_seqs = full_trajectory_s_star_seqs[:, -1*num_last_iters_to_keep:]
    full_accept_or_not_arr = full_accept_or_not_arr[:, -1*num_last_iters_to_keep:]
    full_s_star_levenshtein_dists = full_s_star_levenshtein_dists[:, -1*num_last_iters_to_keep:]
    print("Truncating mcmc generations to last {} iters".format(num_last_iters_to_keep))
    print("full_trajectory_fitness_scores.shape: ", full_trajectory_fitness_scores.shape)
    print("full_trajectory_s_star_seqs.shape: ", full_trajectory_s_star_seqs.shape)
    print("full_accept_or_not_arr.shape: ", full_accept_or_not_arr.shape)
    print("full_s_star_levenshtein_dists.shape: ", full_s_star_levenshtein_dists.shape)

# init_seq_fitness_list = np.concatenate(init_seq_fitness_list, axis=None).tolist()

save_path = os.path.join(generation_output_dir, "{}mcmc_gen_dict.pkl".format(prepend_output_name))
saved_dict = {
                "full_trajectory_fitness_scores": full_trajectory_fitness_scores, 
                "full_trajectory_s_star_seqs": full_trajectory_s_star_seqs, 
                "full_accept_or_not_arr": full_accept_or_not_arr, 
                "full_s_star_levenshtein_dists": full_s_star_levenshtein_dists
                }

with open(save_path, 'wb') as f:
    pickle.dump(saved_dict, f)
    
if num_last_iters_to_keep is None:
    print("Check A1")
    all_mt_fitness_scores_list = full_trajectory_fitness_scores[:, 1:].flatten().tolist()
    all_mt_seqs_list = full_trajectory_s_star_seqs[:, 1:].flatten().tolist()
else:
    print("Check A2")
    all_mt_fitness_scores_list = full_trajectory_fitness_scores.flatten().tolist()
    all_mt_seqs_list = full_trajectory_s_star_seqs.flatten().tolist()

all_mt_accept_list = full_accept_or_not_arr.flatten().tolist()
all_mt_levenshtein_dists_list = full_s_star_levenshtein_dists.flatten().tolist()




# TODO: Ground-Truth classifier inference - start -
# Ground-Truth model set up - Start -
# print("GT all_mt_seqs_list[:5]: ", all_mt_seqs_list[:5])
output_seq_list = all_mt_seqs_list
gt_tokenizer = BertTokenizer.from_pretrained(gt_tokenizer_pretrained_dir)
gt_model = BertForSequenceClassification.from_pretrained(gt_pretrained_dir, num_labels=5)

gt_model.eval()

gt_model = gt_model.to("cuda")

# free up GPU memory
# del gen_model
del disc_model
if mut_type == 't5':
    del t5_model
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

        # Process input batch - start -
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
            
        # Process input batch - end -
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
        
        # if batch_ind % gt_save_interval == 0:
        #     print("inferred #", (batch_ind+1)*gt_batch_size)
        #     cur_time = time.time()

        #     save_path = os.path.join(generation_output_dir, "{}gt_{}-{}.pkl".format(prepend_output_name, (batch_ind+1)*gt_batch_size, num_generations))
        #     with open(save_path, 'wb') as f:
        #         pickle.dump(gt_pred_list, f)
        #     cur_time = time.time()
        #     print("Time taken so far:", cur_time - start_time)

        #     if prev_save_path is not None:
        #         os.remove(prev_save_path)
        #     prev_save_path = save_path

gt_pred_list = np.concatenate(gt_pred_list, axis=0)
gt_class_pred_list = np.concatenate(gt_class_pred_list, axis=None).tolist()
gt_highest_prob_list = np.concatenate(gt_highest_prob_list, axis=None).tolist()
gt_neg_prob_list = np.concatenate(gt_neg_prob_list, axis=None).tolist()
gt_pos_prob_list = np.concatenate(gt_pos_prob_list, axis=None).tolist()
gt_2class_pred_list = np.concatenate(gt_2class_pred_list, axis=None).tolist()



# save_path = os.path.join(generation_output_dir, "{}gt_pred_full{}.pkl".format(prepend_output_name, num_generations))
# with open(save_path, 'wb') as f:
#     pickle.dump(gt_pred_list, f)

# if prev_save_path is not None:
#     os.remove(prev_save_path)

# TODO:Ground-Truth classifier inference - end -





# TODO: PPL computation with GPT-2 - start - 
# print("PPL all_mt_seqs_list[:5]: ", all_mt_seqs_list[:5])
output_seq_list = all_mt_seqs_list
ppl_batch_size = 1 # only works with batch size 1 now
ppl_model = GPT2LMHeadModel.from_pretrained(ppl_model_id).to(gt_model.device)
ppl_tokenizer = GPT2TokenizerFast.from_pretrained(ppl_model_id)
gen_seq_ppl_list = []
input_seq_ppl_list = []


num_ppl_batch = len(output_seq_list) // ppl_batch_size
if len(output_seq_list) % ppl_batch_size != 0:
    num_ppl_batch += 1

start_time = time.time()

print("PPL compute for generated sequences")
with torch.no_grad():
    for batch_ind in tqdm(range(num_ppl_batch)):

        # TODO: Process input batch - start -
        # decoded string sequences
        gen_seq_batch = output_seq_list[batch_ind*ppl_batch_size : (batch_ind+1)*ppl_batch_size]

        batch_input_ids = []
        # tokenize
        for seq in gen_seq_batch:
            # print("seq: ", seq)
            input_ids = ppl_tokenizer.encode(seq)
            # print("input_ids: ", input_ids)
            input_ids = np.array(input_ids, np.int64)
            batch_input_ids.append(input_ids)

        # collate
        batch_input_ids = torch.from_numpy(pad_sequences(batch_input_ids, 0)).to(ppl_model.device)

        # print("batch_input_ids: ", batch_input_ids)
        # print("batch_input_ids.shape: ", batch_input_ids.shape)
        if batch_input_ids.shape[1] == 0:
            gen_seq_ppl_list.append(None)
        else:
            ppl_output = ppl_model(input_ids=batch_input_ids, labels=batch_input_ids)
            log_likelihood = ppl_output[0]
            # print("B log_likelihood: ", log_likelihood)
            seq_ppl = torch.exp(log_likelihood)
            # print("B seq_ppl: ", seq_ppl)
            # print("B seq_ppl.shape: ", seq_ppl.shape)
            gen_seq_ppl_list.append(seq_ppl.cpu().numpy())

gen_seq_ppl_list = np.concatenate(gen_seq_ppl_list, axis=None).tolist()

# print("PPL compute for input sequences")
# # infer input_seq ppl
# with torch.no_grad():
#     for batch_ind in tqdm(range(num_ppl_batch)):

#         # TODO: Process input batch - start -
#         input_seq_batch = input_seq_list[batch_ind*ppl_batch_size : (batch_ind+1)*ppl_batch_size]

#         batch_input_ids = []
#         # tokenize
#         for seq in input_seq_batch:
#             fixed_seq = ppl_tokenizer.decode(ppl_tokenizer.encode(seq)) # hack to remove space before punctuations (e.g. ' .' , ' ,') which inflates ppl value
#             input_ids = ppl_tokenizer.encode(fixed_seq)
#             input_ids = np.array(input_ids, np.int64)
#             batch_input_ids.append(input_ids)

#         # collate
#         batch_input_ids = torch.from_numpy(pad_sequences(batch_input_ids, 0)).to(ppl_model.device)
            
#         # TODO: Process input batch - end -
#         ppl_output = ppl_model(input_ids=batch_input_ids, labels=batch_input_ids)
#         log_likelihood = ppl_output[0]
#         seq_ppl = torch.exp(log_likelihood)
#         input_seq_ppl_list.append(seq_ppl.cpu().numpy())

# input_seq_ppl_list = np.concatenate(input_seq_ppl_list, axis=None).tolist()
# TODO: PPL computation with GPT-2 - end - 



# Save generated samples into TSV file
# PDB, Chain, Start_index, WT_seq, MT_seq
# PDB = 'template2.pdb'
# Chain = 'A'
# Start_index = 19
# WT_seq = 'STIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITEENVQNMNNAGDKWSAFLKEQSTLAQMYPLQEIQNLTVKLQLQALQ'

df = pd.DataFrame()
df['disc_pred'] = all_mt_fitness_scores_list
# df['latent_head_pred'] = latent_head_pred_list
# df['MT_seq'] = all_mt_seqs_list

# df['PDB'] = PDB
# df['Chain'] = Chain
# df['Start_index'] = Start_index
# df['WT_seq'] = WT_seq


# added
# df['gen_input_seq'] = input_seq_list
# df['input_seq_ppl'] = input_seq_ppl_list
df['gt_class_pred'] = gt_class_pred_list
df['MT_edit_dist_vs_WT'] = all_mt_levenshtein_dists_list
df['generated_seq_ppl'] = gen_seq_ppl_list
df['gt_highest_prob'] = gt_highest_prob_list
df['gt_2class_pred'] = gt_2class_pred_list
df['gt_neg_prob'] = gt_neg_prob_list
df['gt_pos_prob'] = gt_pos_prob_list
df['generated_seq'] = all_mt_seqs_list
df['accepted'] = all_mt_accept_list



# Disc-predicted most positive ones first
df = df.sort_values(by='disc_pred', ascending=False)

tsv_name = os.path.join(generation_output_dir, "{}mcmc_seqs.tsv".format(prepend_output_name))

df.to_csv(tsv_name, sep="\t", index=False)
print("output tsv file: ", tsv_name)










