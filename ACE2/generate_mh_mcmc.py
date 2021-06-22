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
from collections import OrderedDict
import os
import shutil
import pickle
from tqdm import tqdm

from modeling_progeny import ProgenyForSequenceToSequenceClassification, ProgenyForValuePrediction, ProgenyForSequenceClassification, ProgenyForContactPrediction, ProgenyConfig
from transformers_custom import MT5ForConditionalGenerationWithLatentSpace


# argparse 
parser = argparse.ArgumentParser()

parser.add_argument('--seed', action='store', type=int, default=30, help='random seed')
# parser.add_argument('--num_generations', action='store', type=int, default=None, help='(min) number of generation')
parser.add_argument('--generation_output_dir', action='store', type=str, default="generated_seqs/mcmc_ACE/top12500input20iter" )
parser.add_argument('--prepend_output_name', action='store', type=str, default="" )
parser.add_argument('--unique_gen', action='store_true')

parser.add_argument('--input_data_dir', action='store', type=str, default="data/gen_train_data/top_half_ddG", help='data for generator input seqs' )
parser.add_argument('--topk_as_input', action='store', type=int, default=12500, help='top K most stable sequences to use input for generation')

# discriminator args
parser.add_argument('--disc_batch_size', action='store', type=int, default=1000)
parser.add_argument('--disc_save_interval', action='store', type=int, default=30)
parser.add_argument('--disc_pretrained_dir', action='store', type=str, default="/export/share/alvinchan/models/ACE/basegen/discriminator/stability_transformer_21-03-08-00-37-29_932314" )


# MCMC args
parser.add_argument('--boltz_constant', action='store', type=float, default=1.0)
parser.add_argument('--temperature', action='store', type=float, default=1.0)
parser.add_argument('--num_gen_rounds', action='store', type=int, default=1, help='how many rounds of evolutionary generation across the whole set of gen_input_df')
parser.add_argument('--trust_radius', action='store', type=int, default=18)
parser.add_argument('--num_evo_iters', action='store', type=int, default=20)
parser.add_argument('--num_last_iters_to_keep', action='store', type=int, default=None)



args = parser.parse_args()

print("args: ", args)

seed = args.seed
generation_output_dir = args.generation_output_dir
prepend_output_name = args.prepend_output_name
disc_pretrained_dir = args.disc_pretrained_dir
disc_batch_size = args.disc_batch_size
input_data_dir = args.input_data_dir
topk_as_input = args.topk_as_input

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

tokenizer = TAPETokenizer(vocab="progeny")

t5config = MT5Config.from_pretrained(disc_pretrained_dir)
config = ProgenyConfig.from_pretrained(disc_pretrained_dir)

disc_model = ProgenyForValuePrediction.from_pretrained(disc_pretrained_dir, config=config, t5config=t5config, predict_head='contrastive')
disc_model.eval()

disc_model.to("cuda")

# Set up input data
input_data_path = Path(input_data_dir)
input_data_file = f'train_ddG.pkl' 
input_data_file = input_data_path / input_data_file
input_data_df = pd.read_pickle(input_data_file)

train_seq_list = input_data_df['MT_seq'].tolist()

print("ddG stats of input data")
print("min: ", np.min(input_data_df['ddG']))
print("mean: ", np.mean(input_data_df['ddG']))
print("median: ", np.median(input_data_df['ddG']))
print("max: ", np.max(input_data_df['ddG']))

ddG_sorted_input_df = input_data_df.sort_values(by='ddG', ascending=True)

gen_input_df = ddG_sorted_input_df.iloc[:topk_as_input]




# mutation params
wt_seq = "STIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITEENVQNMNNAGDKWSAFLKEQSTLAQMYPLQEIQNLTVKLQLQALQ"
reserved_span = "NTNITEEN"
reserved_span_start = 32
constant_indexes = [i for i in range(reserved_span_start, reserved_span_start+len(reserved_span))]

# possible indexes to mutate
mutatable_indexes = [i for i in range(len(wt_seq)) if i not in constant_indexes]

omitted_letters = ["C", "X", "Z", "J", "B", "O", "U"] # for ACE
# omitted_letters = ["X", "Z", "J", "B", "O", "U"] # for SH3
possible_letters = [chr(i).upper() for i in range(ord('a'),ord('z')+1) if chr(i).upper() not in omitted_letters]

mutatable_len = len(wt_seq) - len(reserved_span)
print("mutatable_len: ", mutatable_len)

def hamming_dist(str1, str2=wt_seq):
    i = 0
    count = 0
 
    while(i < len(str1)):
        if(str1[i] != str2[i]):
            count += 1
        i += 1
    return count


# Start MCMC generations
num_batch = len(gen_input_df) // disc_batch_size
if len(gen_input_df) % disc_batch_size != 0:
    num_batch += 1

init_seq_fitness_list = []
full_trajectory_fitness_scores = []
full_trajectory_s_star_seqs = []
full_accept_or_not_arr = []
full_s_star_hamming_dists = []

with torch.no_grad():
    for batch_ind in tqdm(range(num_batch)):
        batch_seqs = gen_input_df[batch_ind*disc_batch_size : (batch_ind+1)*disc_batch_size]['MT_seq']

        batch_input_ids = []
        batch_input_seqs = []

        # encode seqs
        for seq in batch_seqs:
            batch_input_seqs = batch_input_seqs + [seq]
            input_ids = tokenizer.encode(seq)
            input_ids = torch.from_numpy(input_ids).unsqueeze(0)
            batch_input_ids.append(input_ids)
            
        batch_input_ids = torch.cat(batch_input_ids, dim=0)
        batch_input_ids = batch_input_ids.cuda()

        disc_output = disc_model(batch_input_ids)
        init_seq_fitness = disc_output[0].cpu().numpy() # shape: [N, 1]
        init_seq_fitness_list.append(init_seq_fitness)

        # initialize trajectory fitness scores
        batch_trajectory_fitness_scores = np.copy(init_seq_fitness)
        batch_trajectory_s_star_seqs = np.expand_dims(np.array(batch_seqs), axis=-1)
        batch_accept_or_not_arr = None
        batch_s_star_hamming_dists = None
        s_fitness = np.squeeze(np.copy(init_seq_fitness), axis=1) # shape: [N]

        # compute evolutionary trajectory
        batch_s = batch_seqs[:]
        batch_mu_mut = np.random.uniform(low=1, high=2.5, size=len(batch_s))
        for iter_ind in tqdm(range(num_evo_iters)):
            
            # mutate s to get s_star
            batch_s_star = []
            for s_ind, s in enumerate(batch_s):
                num_mutation = np.random.poisson(batch_mu_mut[s_ind]-1) + 1
                indexes_to_mutate = random.sample(mutatable_indexes, num_mutation)

                mut_seq = list(s)
                for i_mut in indexes_to_mutate:
                    current_aa = s[i_mut]

                    sub_aa = current_aa
                    while sub_aa == current_aa:
                        # sample an AA to subsitute current AA
                        sub_aa = random.sample(possible_letters, 1)[0]

                    mut_seq[i_mut] = sub_aa

                mut_seq = "".join(mut_seq)
                batch_s_star.append(mut_seq)

            # encode s_star
            batch_input_ids = []
            for s_star in batch_s_star:
                input_ids = tokenizer.encode(s_star)
                input_ids = torch.from_numpy(input_ids).unsqueeze(0)
                batch_input_ids.append(input_ids)

            batch_input_ids = torch.cat(batch_input_ids, dim=0)
            batch_input_ids = batch_input_ids.cuda()

            # infer fitness of s_star
            disc_output = disc_model(batch_input_ids)
            fitness_tensor = disc_output[0]
            s_star_fitness_arr = fitness_tensor.cpu().numpy()
            # print("A s_star_fitness_arr.shape: ", s_star_fitness_arr.shape)
            s_star_fitness = np.squeeze(s_star_fitness_arr, axis=1) # shape: [N]
            # print("B s_star_fitness.shape: ", s_star_fitness.shape)



            # add s_star's fitness and seq to trajectory fitness array
            batch_trajectory_fitness_scores = np.concatenate([batch_trajectory_fitness_scores, s_star_fitness_arr], axis=1)
            # print("C batch_trajectory_fitness_scores.shape: ", batch_trajectory_fitness_scores.shape)

            iter_s_star_seqs = np.expand_dims(np.array(batch_s_star), axis=-1)
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
            s_star_hamming_dists = np.array(list(map(hamming_dist, batch_s_star)))
            if batch_s_star_hamming_dists is None:
                batch_s_star_hamming_dists = np.expand_dims(s_star_hamming_dists, axis=-1)
            else:
                batch_s_star_hamming_dists = np.concatenate([batch_s_star_hamming_dists, np.expand_dims(s_star_hamming_dists, axis=-1)], axis=1)

            # print("s_star_hamming_dists: ", s_star_hamming_dists)
            within_trust_radius = (s_star_hamming_dists <= trust_radius).astype(int)
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
        full_s_star_hamming_dists.append(batch_s_star_hamming_dists)

full_trajectory_fitness_scores = np.concatenate(full_trajectory_fitness_scores, axis=0) # shape: [topk_as_input, num_evo_iters+1], +1 to include initial seqs
full_trajectory_s_star_seqs = np.concatenate(full_trajectory_s_star_seqs, axis=0) # shape: [topk_as_input, num_evo_iters+1], +1 to include initial seqs
full_accept_or_not_arr = np.concatenate(full_accept_or_not_arr, axis=0) # shape: [topk_as_input, num_evo_iters]
full_s_star_hamming_dists = np.concatenate(full_s_star_hamming_dists, axis=0) # shape: [topk_as_input, num_evo_iters]


print("full_trajectory_fitness_scores.shape: ", full_trajectory_fitness_scores.shape)
print("full_trajectory_s_star_seqs.shape: ", full_trajectory_s_star_seqs.shape)
print("full_accept_or_not_arr.shape: ", full_accept_or_not_arr.shape)
print("full_s_star_hamming_dists.shape: ", full_s_star_hamming_dists.shape)
if num_last_iters_to_keep is not None:
    full_trajectory_fitness_scores = full_trajectory_fitness_scores[:, -1*num_last_iters_to_keep:]
    full_trajectory_s_star_seqs = full_trajectory_s_star_seqs[:, -1*num_last_iters_to_keep:]
    full_accept_or_not_arr = full_accept_or_not_arr[:, -1*num_last_iters_to_keep:]
    full_s_star_hamming_dists = full_s_star_hamming_dists[:, -1*num_last_iters_to_keep:]
    print("Truncating mcmc generations to last {} iters".format(num_last_iters_to_keep))
    print("full_trajectory_fitness_scores.shape: ", full_trajectory_fitness_scores.shape)
    print("full_trajectory_s_star_seqs.shape: ", full_trajectory_s_star_seqs.shape)
    print("full_accept_or_not_arr.shape: ", full_accept_or_not_arr.shape)
    print("full_s_star_hamming_dists.shape: ", full_s_star_hamming_dists.shape)

# init_seq_fitness_list = np.concatenate(init_seq_fitness_list, axis=None).tolist()

save_path = os.path.join(generation_output_dir, "{}mcmc_gen_dict.pkl".format(prepend_output_name))
saved_dict = {
                "full_trajectory_fitness_scores": full_trajectory_fitness_scores, 
                "full_trajectory_s_star_seqs": full_trajectory_s_star_seqs, 
                "full_accept_or_not_arr": full_accept_or_not_arr, 
                "full_s_star_hamming_dists": full_s_star_hamming_dists
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
all_mt_hamming_dists_list = full_s_star_hamming_dists.flatten().tolist()

# Save generated samples into TSV file
# PDB, Chain, Start_index, WT_seq, MT_seq
PDB = 'template2.pdb'
Chain = 'A'
Start_index = 19
WT_seq = 'STIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITEENVQNMNNAGDKWSAFLKEQSTLAQMYPLQEIQNLTVKLQLQALQ'

df = pd.DataFrame()
df['disc_pred'] = all_mt_fitness_scores_list
# df['latent_head_pred'] = latent_head_pred_list
df['MT_seq'] = all_mt_seqs_list

df['PDB'] = PDB
df['Chain'] = Chain
df['Start_index'] = Start_index
df['WT_seq'] = WT_seq

df['MT_edit_dist_vs_WT'] = all_mt_hamming_dists_list
df['accepted'] = all_mt_accept_list


# Disc-predicted most stable ones first
df = df.sort_values(by='disc_pred', ascending=True)

tsv_name = os.path.join(generation_output_dir, "{}mcmc_seqs.tsv".format(prepend_output_name))

df.to_csv(tsv_name, sep="\t", index=False)










