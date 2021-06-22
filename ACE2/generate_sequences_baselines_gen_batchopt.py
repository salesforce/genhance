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
import pickle
from tqdm import tqdm

from modeling_progeny import ProgenyForSequenceToSequenceClassification, ProgenyForValuePrediction, ProgenyForSequenceClassification, ProgenyForContactPrediction, ProgenyConfig
from transformers_custom import MT5ForConditionalGenerationWithLatentSpace

# argparse 
parser = argparse.ArgumentParser()

parser.add_argument('--seed', action='store', type=int, default=30, help='random seed')
parser.add_argument('--num_generations', action='store', type=int, default=20000, help='(min) number of generation')
parser.add_argument('--generation_output_dir', action='store', type=str, default="generated_seqs/" )
parser.add_argument('--prepend_output_name', action='store', type=str, default="" )
parser.add_argument('--gen_pretrained_dir', action='store', type=str, default="gen/tophalf_12ep/results/checkpoint-92000" )
parser.add_argument('--input_seq', action='store', type=str, default="" )
parser.add_argument('--temperature_init', action='store', type=float, default=1.0)
parser.add_argument('--temperature_multiple', action='store', type=float, default=1.2)
parser.add_argument('--patience', action='store', type=int, default=50, help='number of repeats before increasing temperature values for gen decoding')
parser.add_argument('--batch_repeat_threshold', action='store', type=int, default=4)
parser.add_argument('--gen_batch_size', action='store', type=int, default=800)
parser.add_argument('--gen_save_interval', action='store', type=int, default=1000, help='interval to save generations')
parser.add_argument('--disc_batch_size', action='store', type=int, default=1000)
parser.add_argument('--disc_save_interval', action='store', type=int, default=30)
parser.add_argument('--disc_pretrained_dir', action='store', type=str, default="/export/share/alvinchan/models/ACE/basegen/discriminator/stability_transformer_21-03-08-00-37-29_932314" )

# latent head args
parser.add_argument('--do_latenthead_eval', action='store_true')
parser.add_argument('--latenthead_batch_size', action='store', type=int, default=200)
parser.add_argument('--latenthead_pretrained_dir', action='store', type=str, default="congen/v1/clspool_waeDeterencStart84kstep1024dim_cyccon1Start84kstep_lre-04_24ep" )
parser.add_argument('--latent_pooler', action='store', type=str, default="mean", choices=['mean', 'max', 'cls'], help='op to pool encoder hidden states' )
parser.add_argument('--pool_enc_hidden_states_for_dec', action='store_true')
parser.add_argument('--mask_non_target_z_vector', action='store_true')
parser.add_argument('--separate_targetattr_head', action='store_true')
parser.add_argument('--z_tar_vector_dim', action='store', type=int, default=1)
parser.add_argument('--do_mi', action='store_true')
parser.add_argument('--latent_space_type', action='store', type=str, default="plain", choices=['plain', 'vae', 'wae', 'adversarial'], help='type of latent space' )
parser.add_argument('--latent_size', action='store', type=int, default=None, help='use None to use pooled enc hidden state as latent vector')
parser.add_argument('--no_separate_latent_enc', action='store_false', dest='separate_latent_enc', default=True)
parser.add_argument('--no_separate_latent_dec', action='store_false', dest='separate_latent_dec', default=True)
parser.add_argument('--wae_z_enc_type', action='store', type=str, default=None, choices=['deterministic', 'stochastic'], help='type of wae encoder' )

parser.add_argument('--skip_gen', action='store_true')

args = parser.parse_args()

print("args: ", args)

seed = args.seed
num_generations = args.num_generations
gen_save_interval = args.gen_save_interval
generation_output_dir = args.generation_output_dir
prepend_output_name = args.prepend_output_name
gen_pretrained_dir = args.gen_pretrained_dir
input_seq = args.input_seq
temperature_init = args.temperature_init
temperature_multiple = args.temperature_multiple
patience = args.patience
batch_repeat_threshold = args.batch_repeat_threshold
gen_batch_size = args.gen_batch_size
disc_batch_size = args.disc_batch_size
disc_save_interval = args.disc_save_interval
disc_pretrained_dir = args.disc_pretrained_dir

os.makedirs(generation_output_dir, exist_ok = True)

wt_seq = 'STIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITEENVQNMNNAGDKWSAFLKEQSTLAQMYPLQEIQNLTVKLQLQALQ'
constant_region = 'NTNITEEN'

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

tokenizer = TAPETokenizer(vocab="progeny")

device = torch.device('cuda:0')

t5config = MT5Config.from_pretrained(gen_pretrained_dir)
gen_model = MT5ForConditionalGeneration.from_pretrained(gen_pretrained_dir)

gen_model.parallelize()

input_ids = tokenizer.encode(input_seq)
input_ids = torch.from_numpy(input_ids).to(gen_model.device).unsqueeze(0)
print("input_ids: ", input_ids)

batch_input_ids = torch.cat([input_ids for i in range(gen_batch_size)], dim=0)

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
                tokens = tokenizer.convert_ids_to_tokens(gen_seq.tolist())
                if tokens == None:
                    continue
                str_token_seq = "".join(tokens[2:-1])

                if str_token_seq in output_seq_list:
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
    
gen_tensors = torch.stack(output_tensor_list, dim=0)
# new gen


# Discriminator inference
t5config = MT5Config.from_pretrained(disc_pretrained_dir)
config = ProgenyConfig.from_pretrained(disc_pretrained_dir)

disc_model = ProgenyForValuePrediction.from_pretrained(disc_pretrained_dir, config=config, t5config=t5config, predict_head='contrastive')
disc_model.eval()

disc_model = disc_model.to(gen_model.device)

# disc: more positive values mean less stable, more negative values mean more stable
disc_pred_list = []
prev_save_path = None

num_disc_batch = len(gen_tensors) // disc_batch_size
if len(gen_tensors) % disc_batch_size != 0:
    num_disc_batch += 1

start_time = time.time()
with torch.no_grad():
    for batch_ind in range(num_disc_batch):
        gen_tensor_batch = gen_tensors[batch_ind*disc_batch_size : (batch_ind+1)*disc_batch_size, 1:]
        gen_tensor_batch = gen_tensor_batch.to(gen_model.device)
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


# TODO: add latent head inference
if args.do_latenthead_eval:
    latenthead_batch_size = args.latenthead_batch_size
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
    genhance_model = MT5ForConditionalGenerationWithLatentSpace.from_pretrained(args.latenthead_pretrained_dir, **latent_space_args)

    genhance_model.parallelize()

    # Latent Head inference - start
    latent_head_pred_list = []
    prev_save_path = None

    num_latenthead_batch = len(gen_tensors) // latenthead_batch_size
    if len(gen_tensors) % latenthead_batch_size != 0:
        num_latenthead_batch += 1

    start_time = time.time()
    del disc_model
    del gen_model
    genhance_model.eval()
    with torch.no_grad():
        for batch_ind in tqdm(range(num_latenthead_batch)):
            gen_tensor_batch = gen_tensors[batch_ind*latenthead_batch_size : (batch_ind+1)*latenthead_batch_size, 1:]
            gen_tensor_batch = gen_tensor_batch.to(genhance_model.device)
            print("gen_tensor_batch: ", gen_tensor_batch)
            print("gen_tensor_batch.shape: ", gen_tensor_batch.shape)
            model_outputs = genhance_model(gen_tensor_batch, labels=gen_tensor_batch)
            contrastive_value = model_outputs[1]
            latent_head_pred_list.append(contrastive_value.squeeze().cpu().numpy())

            if batch_ind % disc_save_interval == 0:
                print("latent head inferred #", (batch_ind+1)*latenthead_batch_size)
                cur_time = time.time()

                save_path = os.path.join(generation_output_dir, "{}latent_head_{}-{}.pkl".format(prepend_output_name, (batch_ind+1)*latenthead_batch_size, num_generations))
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




# Save generated samples into TSV file
# PDB, Chain, Start_index, WT_seq, MT_seq
PDB = 'template2.pdb'
Chain = 'A'
Start_index = 19
WT_seq = 'STIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITEENVQNMNNAGDKWSAFLKEQSTLAQMYPLQEIQNLTVKLQLQALQ'

df = pd.DataFrame()
df['disc_pred'] = disc_pred_list
if args.do_latenthead_eval:
    df['latent_head_pred'] = latent_head_pred_list
df['MT_seq'] = output_seq_list

df['PDB'] = PDB
df['Chain'] = Chain
df['Start_index'] = Start_index
df['WT_seq'] = WT_seq

# Disc-predicted most stable ones first
df = df.sort_values(by='disc_pred', ascending=True)

tsv_name = os.path.join(generation_output_dir, "{}basegen_seqs{}.tsv".format(prepend_output_name, num_generations))

df.to_csv(tsv_name, sep="\t", index=False)