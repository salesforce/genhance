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
parser.add_argument('--num_generations', action='store', type=int, default=None, help='(min) number of generation')
parser.add_argument('--generation_output_dir', action='store', type=str, default="controlled_generated_seqs_debug/" )
parser.add_argument('--prepend_output_name', action='store', type=str, default="" )
parser.add_argument('--unique_gen', action='store_true')

# generator args
parser.add_argument('--gen_pretrained_dir', action='store', type=str, default="congen/v1/meanpool_sephead256dim_domi/" )
# parser.add_argument('--input_seq', action='store', type=str, default="" )
parser.add_argument('--temperature_init', action='store', type=float, default=1.0)
parser.add_argument('--temperature_multiple', action='store', type=float, default=1.2)
parser.add_argument('--patience', action='store', type=int, default=50, help='number of repeats before increasing temperature values for gen decoding')
parser.add_argument('--batch_repeat_threshold', action='store', type=int, default=4)
parser.add_argument('--gen_batch_size', action='store', type=int, default=200)
parser.add_argument('--gen_save_interval', action='store', type=int, default=100, help='interval to save generations')
parser.add_argument('--skip_gen', action='store_true')

parser.add_argument('--gen_token_len', action='store', type=int, default=86, help='len to check for generated tokens')

# new controlled gen args
parser.add_argument('--input_data_dir', action='store', type=str, default="data/gen_train_data/top_half_ddG", help='data for generator input seqs' )
parser.add_argument('--src_config_json', action='store', type=str, default="/export/share/bkrause/progen/progeny/t5_base_uniref_bfd50/config.json" )
parser.add_argument('--topk_as_input', action='store', type=int, default=12500, help='top K most stable sequences to use input for generation')
parser.add_argument('--num_gen_samples_per_input', action='store', type=int, default=20, help='number of generation per input sequence')

# latent space args
parser.add_argument('--latent_pooler', action='store', type=str, default="mean", choices=['mean', 'max', 'cls'], help='op to pool encoder hidden states' )
parser.add_argument('--pool_enc_hidden_states_for_dec', action='store_true')
parser.add_argument('--mask_non_target_z_vector', action='store_true')
parser.add_argument('--separate_targetattr_head', action='store_true')
parser.add_argument('--z_tar_vector_dim', action='store', type=int, default=1)
parser.add_argument('--do_mi', action='store_true')
parser.add_argument('--z_tar_edit_before_dec', action='store', type=float, default=None, help='perturbation to latent vector z_tar')

# vae/wae args
parser.add_argument('--latent_space_type', action='store', type=str, default="plain", choices=['plain', 'vae', 'wae', 'adversarial'], help='type of latent space' )
parser.add_argument('--latent_size', action='store', type=int, default=None, help='use None to use pooled enc hidden state as latent vector')

parser.add_argument('--no_separate_latent_enc', action='store_false', dest='separate_latent_enc', default=True)
parser.add_argument('--no_separate_latent_dec', action='store_false', dest='separate_latent_dec', default=True)

# wae only args
parser.add_argument('--wae_z_enc_type', action='store', type=str, default=None, choices=['deterministic', 'stochastic'], help='type of wae encoder' )


# discriminator args
parser.add_argument('--disc_batch_size', action='store', type=int, default=1000)
parser.add_argument('--disc_save_interval', action='store', type=int, default=30)
parser.add_argument('--disc_pretrained_dir', action='store', type=str, default="/export/share/alvinchan/models/ACE/basegen/discriminator/stability_transformer_21-03-08-00-37-29_932314" )

args = parser.parse_args()

print("args: ", args)

seed = args.seed
num_generations = args.num_generations
gen_save_interval = args.gen_save_interval
generation_output_dir = args.generation_output_dir
prepend_output_name = args.prepend_output_name
gen_pretrained_dir = args.gen_pretrained_dir
# input_seq = args.input_seq
temperature_init = args.temperature_init
temperature_multiple = args.temperature_multiple
patience = args.patience
batch_repeat_threshold = args.batch_repeat_threshold
gen_batch_size = args.gen_batch_size
disc_batch_size = args.disc_batch_size
disc_save_interval = args.disc_save_interval
disc_pretrained_dir = args.disc_pretrained_dir
unique_gen = args.unique_gen

# new controlled gen args
src_config_json = args.src_config_json
input_data_dir = args.input_data_dir
topk_as_input = args.topk_as_input
num_gen_samples_per_input = args.num_gen_samples_per_input
z_tar_edit_before_dec = args.z_tar_edit_before_dec
latent_space_args = {
    'latent_pooler': args.latent_pooler,
    'pool_enc_hidden_states_for_dec': args.pool_enc_hidden_states_for_dec,
    'mask_non_target_z_vector': args.mask_non_target_z_vector,
    'separate_targetattr_head': args.separate_targetattr_head,
    'z_tar_vector_dim': args.z_tar_vector_dim,
    'do_mi': args.do_mi,
    'latent_space_type': args.latent_space_type,
    'latent_size': args.latent_size,    
    'separate_latent_enc': args.separate_latent_enc,
    'separate_latent_dec': args.separate_latent_dec,
    'wae_z_enc_type': args.wae_z_enc_type,
}

if not os.path.isfile(os.path.join(gen_pretrained_dir, 'config.json')):
    shutil.copy(src_config_json, gen_pretrained_dir)

output_dir = Path(generation_output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
# os.makedirs(generation_output_dir, exist_ok = True)

wt_seq = 'STIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITEENVQNMNNAGDKWSAFLKEQSTLAQMYPLQEIQNLTVKLQLQALQ'
constant_region = 'NTNITEEN'

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

# Set up generator model
tokenizer = TAPETokenizer(vocab="progeny")
device = torch.device('cuda:0')

gen_model = MT5ForConditionalGenerationWithLatentSpace.from_pretrained(gen_pretrained_dir, **latent_space_args)
# t5config = MT5Config.from_pretrained(gen_pretrained_dir)
# gen_model = MT5ForConditionalGeneration.from_pretrained(gen_pretrained_dir)

gen_model.parallelize()


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


# gen code - start
if num_generations is None:
    num_generations = topk_as_input * num_gen_samples_per_input
num_unique_seqs_per_batch = gen_batch_size // num_gen_samples_per_input
num_batch = len(gen_input_df) // num_unique_seqs_per_batch
if len(gen_input_df) % num_unique_seqs_per_batch != 0:
    num_batch += 1
output_seq_list = []
input_seq_list = []
output_tensor_list = []
repeat_list = []
in_train_data_list = []
unique_n_notrain_list = []
start_time = time.time()
prev_save_path = None
repeat_seq_count = 0
in_train_count = 0
temperature = temperature_init
generation_rounds_done = 0

# TODO loading code for batching input for generations
if not args.skip_gen:
    gen_model.eval()
    while unique_gen and np.sum(unique_n_notrain_list) < num_generations:
        if generation_rounds_done > 0:
            temperature = temperature * temperature_multiple
            print("New generation round, temperature: ", temperature)

        for batch_ind in tqdm(range(num_batch)):
            batch_seqs = gen_input_df[batch_ind*num_unique_seqs_per_batch : (batch_ind+1)*num_unique_seqs_per_batch]['MT_seq']
            batch_input_ids = []
            batch_input_seqs = []
            for seq in batch_seqs:
                batch_input_seqs = batch_input_seqs + [seq]* num_gen_samples_per_input
                input_ids = tokenizer.encode(seq)
                input_ids = torch.from_numpy(input_ids).to(gen_model.device).unsqueeze(0)
                repeated_input_ids = input_ids.repeat((num_gen_samples_per_input, 1)) 
                batch_input_ids.append(repeated_input_ids)
            # print("batch_input_seqs: ", batch_input_seqs)

            batch_input_ids = torch.cat(batch_input_ids, dim=0)

            # print("batch_input_ids: ", batch_input_ids)
            # print("batch_input_ids.shape: ", batch_input_ids.shape)
            gen_output = gen_model.generate(batch_input_ids, max_length=85+1, do_sample=True, temperature=temperature, z_tar_edit_before_dec=z_tar_edit_before_dec)
            # print("gen_output.shape: ", gen_output.shape)

            for seq_ind, gen_seq in enumerate(gen_output.cpu().numpy()):
                unique_n_notrain = True
                repeat = False
                in_train_data = False

                tokens = tokenizer.convert_ids_to_tokens(gen_seq.tolist())
                # print("len(tokens): ", len(tokens))
                if tokens == None or len(tokens) != args.gen_token_len:
                    continue
                # print("tokens[:2]: ", tokens[:2]) # ['<pad>', '<cls>']
                # print("gen_seq[:2]: ", gen_seq[:2]) # [0 2]
                str_token_seq = "".join(tokens[2:-1])
                if str_token_seq in output_seq_list:
                    repeat_seq_count += 1
                    repeat = True
                    unique_n_notrain = False

                if str_token_seq in train_seq_list:
                    in_train_count += 1
                    in_train_data = True
                    unique_n_notrain = False

                if unique_gen and not unique_n_notrain:
                    continue
                        
                unique_n_notrain_list.append(unique_n_notrain)
                repeat_list.append(repeat)
                in_train_data_list.append(in_train_data)
                
                input_seq_str = batch_input_seqs[seq_ind]
                input_seq_list.append(input_seq_str)
                output_seq_list.append(str_token_seq)

                seq_tensor = gen_output[seq_ind].detach().cpu()
                output_tensor_list.append(seq_tensor)

            if batch_ind % gen_save_interval == 0 and batch_ind != 0:
                save_path = os.path.join(generation_output_dir, "{}gen_seqs{}-{}.pkl".format(prepend_output_name, len(output_seq_list), num_generations))
                saved_dict = {'output_seq_list': output_seq_list, "input_seq_list": input_seq_list, "output_tensor_list": output_tensor_list, 'repeat_list': repeat_list, 'in_train_data_list': in_train_data_list,  'temperature': temperature}
                with open(save_path, 'wb') as f:
                    pickle.dump(saved_dict, f)
                print("generated #", len(output_seq_list))
                cur_time = time.time()
                print("Time taken so far:", cur_time - start_time)

                if prev_save_path is not None:
                    os.remove(prev_save_path)
                prev_save_path = save_path

            if unique_gen and np.sum(unique_n_notrain_list) > num_generations:
                break
        generation_rounds_done += 1
            


    save_path = os.path.join(generation_output_dir, "{}gen_seqs_full{}.pkl".format(prepend_output_name, num_generations))
    saved_dict = {'output_seq_list': output_seq_list, "input_seq_list": input_seq_list, "output_tensor_list": output_tensor_list, 'repeat_list': repeat_list, 'in_train_data_list': in_train_data_list, 'temperature': temperature}
    with open(save_path, 'wb') as f:
        pickle.dump(saved_dict, f)

else:
    print("Skipping generation step and loading from saved pkl")
    save_path = os.path.join(generation_output_dir, "{}gen_seqs_full{}.pkl".format(prepend_output_name, num_generations))
    with open(save_path, 'rb') as f:
        saved_dict = pickle.load(f)
    output_seq_list = saved_dict['output_seq_list']
    input_seq_list = saved_dict['input_seq_list']
    output_tensor_list = saved_dict['output_tensor_list']
    repeat_list = saved_dict['repeat_list']
    in_train_data_list = saved_dict['in_train_data_list']
    temperature = saved_dict['temperature']

if prev_save_path is not None:
    os.remove(prev_save_path)
    
gen_tensors = torch.stack(output_tensor_list, dim=0)
# new gen


# Latent Head inference - start
latent_head_pred_list = []
prev_save_path = None

num_disc_batch = len(gen_tensors) // gen_batch_size
if len(gen_tensors) % gen_batch_size != 0:
    num_disc_batch += 1

start_time = time.time()
gen_model.eval()
with torch.no_grad():
    for batch_ind in tqdm(range(num_disc_batch)):
        gen_tensor_batch = gen_tensors[batch_ind*gen_batch_size : (batch_ind+1)*gen_batch_size, 1:]
        gen_tensor_batch = gen_tensor_batch.to(gen_model.device)
        print("A gen_tensor_batch: ", gen_tensor_batch)
        print("A gen_tensor_batch.shape: ", gen_tensor_batch.shape)
        model_outputs = gen_model(gen_tensor_batch, labels=gen_tensor_batch)
        contrastive_value = model_outputs[1]
        latent_head_pred_list.append(contrastive_value.squeeze().cpu().numpy())

        if batch_ind % disc_save_interval == 0:
            print("latent head inferred #", (batch_ind+1)*gen_batch_size)
            cur_time = time.time()

            save_path = os.path.join(generation_output_dir, "{}latent_head_{}-{}.pkl".format(prepend_output_name, (batch_ind+1)*gen_batch_size, num_generations))
            with open(save_path, 'wb') as f:
                pickle.dump(latent_head_pred_list, f)
            cur_time = time.time()
            print("Time taken so far:", cur_time - start_time)

            if prev_save_path is not None:
                os.remove(prev_save_path)
            prev_save_path = save_path

latent_head_pred_list = np.concatenate(latent_head_pred_list, axis=None).tolist()

save_path = os.path.join(generation_output_dir, "{}latent_head_full{}.pkl".format(prepend_output_name, num_generations))
with open(save_path, 'wb') as f:
    pickle.dump(latent_head_pred_list, f)

if prev_save_path is not None:
    os.remove(prev_save_path)

# Latent Head inference - end



# Discriminator inference
t5config = MT5Config.from_pretrained(disc_pretrained_dir)
config = ProgenyConfig.from_pretrained(disc_pretrained_dir)

disc_model = ProgenyForValuePrediction.from_pretrained(disc_pretrained_dir, config=config, t5config=t5config, predict_head='contrastive')
disc_model.eval()

disc_model = disc_model.to(gen_model.device)

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
        gen_tensor_batch = gen_tensors[batch_ind*disc_batch_size : (batch_ind+1)*disc_batch_size, 1:]
        gen_tensor_batch = gen_tensor_batch.to(gen_model.device)
        # print("B gen_tensor_batch: ", gen_tensor_batch)
        # print("B gen_tensor_batch.shape: ", gen_tensor_batch.shape)
        disc_output = disc_model(gen_tensor_batch)
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

# Save generated samples into TSV file
# PDB, Chain, Start_index, WT_seq, MT_seq
PDB = 'template2.pdb'
Chain = 'A'
Start_index = 19
WT_seq = 'STIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITEENVQNMNNAGDKWSAFLKEQSTLAQMYPLQEIQNLTVKLQLQALQ'

df = pd.DataFrame()
df['disc_pred'] = disc_pred_list
df['latent_head_pred'] = latent_head_pred_list
df['MT_seq'] = output_seq_list
df['gen_input_seq'] = input_seq_list

df['PDB'] = PDB
df['Chain'] = Chain
df['Start_index'] = Start_index
df['WT_seq'] = WT_seq

df['repeated_gen'] = repeat_list
df['in_train_data_gen'] = in_train_data_list


# Latent head-predicted most stable ones first
df = df.sort_values(by='latent_head_pred', ascending=True)

tsv_name = os.path.join(generation_output_dir, "{}congen_seqs{}.tsv".format(prepend_output_name, num_generations))

df.to_csv(tsv_name, sep="\t", index=False)