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
from transformers_custom import T5ForConditionalGenerationWithLatentSpace, T5Discriminator, T5Tokenizer, T5Config, BertTokenizer, BertForSequenceClassification, GPT2LMHeadModel, GPT2TokenizerFast


# argparse 
parser = argparse.ArgumentParser()

parser.add_argument('--seed', action='store', type=int, default=30, help='random seed')
parser.add_argument('--num_generations', action='store', type=int, default=None, help='(min) number of generation')
parser.add_argument('--generation_output_dir', action='store', type=str, default="controlled_generated_seqs_debug/" )
parser.add_argument('--prepend_output_name', action='store', type=str, default="" )
parser.add_argument('--unique_gen', action='store_true')
parser.add_argument('--no_repeat_input_seq', action='store_true')

# generator args
parser.add_argument('--gen_pretrained_dir', action='store', type=str, default="congen/v1/meanpool_sephead256dim_domi/" )
parser.add_argument('--tokenizer_pretrained_dir', action='store', type=str, default="t5-small" )

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
parser.add_argument('--input_data_dir', action='store', type=str, default="data/sst", help='data for generator input seqs' )
parser.add_argument('--input_data_subset', action='store', type=str, default="train", help='data subset for generator input seqs', choices=["train", "dev", "test"] )
parser.add_argument('--src_config_json', action='store', type=str, default=None )
parser.add_argument('--num_gen_inputs', action='store', type=int, default=None, help='top K most stable sequences to use input for generation')
parser.add_argument('--num_gen_samples_per_input', action='store', type=int, default=10, help='number of generation per input sequence')

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
parser.add_argument('--gen_input_labels', nargs='+', help='Labels of samples to use for generation input seqs, labels are 0: strongly neg, 1: neg, 2: neutral, 3: pos, 4: strongly pos')
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
gen_input_labels = args.gen_input_labels
prepended_cls_token = args.prepended_cls_token
temperature_init = args.temperature_init
temperature_multiple = args.temperature_multiple
patience = args.patience
batch_repeat_threshold = args.batch_repeat_threshold
gen_batch_size = args.gen_batch_size
unique_gen = args.unique_gen
no_repeat_input_seq =args.no_repeat_input_seq

disc_batch_size = args.disc_batch_size
disc_save_interval = args.disc_save_interval
disc_pretrained_dir = args.disc_pretrained_dir
disc_latent_pooler = args.disc_latent_pooler

gt_batch_size = args.gt_batch_size
gt_tokenizer_pretrained_dir = args.gt_tokenizer_pretrained_dir
gt_pretrained_dir = args.gt_pretrained_dir
gt_save_interval = args.gt_save_interval

ppl_model_id = args.ppl_model_id

# new controlled gen args
src_config_json = args.src_config_json
input_data_dir = args.input_data_dir
input_data_subset = args.input_data_subset
num_gen_inputs = args.num_gen_inputs
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

# wt_seq = 'STIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITEENVQNMNNAGDKWSAFLKEQSTLAQMYPLQEIQNLTVKLQLQALQ'
# constant_region = 'NTNITEEN'

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

# Set up generator model
tokenizer = T5Tokenizer.from_pretrained(tokenizer_pretrained_dir)
# tokenizer = TAPETokenizer(vocab="progeny")
device = torch.device('cuda:0')

gen_model = T5ForConditionalGenerationWithLatentSpace.from_pretrained(gen_pretrained_dir, **latent_space_args)

gen_model.parallelize()


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

# Set up input data - end -

# gen code - start
if num_generations is None:
    num_generations = num_gen_inputs * num_gen_samples_per_input
print("num_generations: ", num_generations)
num_unique_seqs_per_batch = gen_batch_size // num_gen_samples_per_input
num_batch = len(gen_input_df) // num_unique_seqs_per_batch
if len(gen_input_df) % num_unique_seqs_per_batch != 0:
    num_batch += 1
output_seq_list = []
input_seq_list = []
input_seq_class_list = []
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

    
# loading code for batching input for generations
if not args.skip_gen:
    gen_model.eval()

    # get prepended_cls_token_id
    if prepended_cls_token is not None:
        prepended_cls_token_id = tokenizer.encode(prepended_cls_token)[0]
    else:
        prepended_cls_token_id = None
    while len(output_seq_list) < num_generations:
    # while unique_gen and np.sum(unique_n_notrain_list) < num_generations:
        if generation_rounds_done > 0:
            temperature = temperature * temperature_multiple
            print("New generation round, temperature: ", temperature)

        # print("num_batch: ", num_batch)
        for batch_ind in tqdm(range(num_batch)):
            # print("batch_ind: ", batch_ind)
            batch_seqs = gen_input_df[batch_ind*num_unique_seqs_per_batch : (batch_ind+1)*num_unique_seqs_per_batch]['text']
            batch_classes = gen_input_df[batch_ind*num_unique_seqs_per_batch : (batch_ind+1)*num_unique_seqs_per_batch]['truth'].tolist()
            # print("batch_seqs: ", batch_seqs)
            batch_input_ids = []
            batch_input_seqs = []
            batch_input_seq_classes = []

            for seq_ind, seq in enumerate(batch_seqs):
                batch_input_seqs = batch_input_seqs + [seq]* num_gen_samples_per_input
                # print("batch_classes: ", batch_classes)
                seq_class = batch_classes[seq_ind]
                # print("seq_class: ", seq_class)
                batch_input_seq_classes = batch_input_seq_classes + [seq_class]* num_gen_samples_per_input
                input_ids = tokenizer.encode(tokenizer.decode(tokenizer.encode(seq))) # decode first encoded ids to remove space before punctuations such as " ," and " ."
                # input_ids = tokenizer.encode(seq)
                # prepend cls token to input seqs
                if prepended_cls_token_id is not None:
                    input_ids = [prepended_cls_token_id] + input_ids

                input_ids = np.array(input_ids, np.int64)
                batch_input_ids.extend([input_ids] * num_gen_samples_per_input)
                # batch_input_ids.append(input_ids)

            batch_input_ids = torch.from_numpy(pad_sequences(batch_input_ids, 0)).to(gen_model.device)
            # print("batch_input_ids.shape: ", batch_input_ids.shape)
            # print("A batch_input_ids: ", batch_input_ids)

            gen_output = gen_model.generate(batch_input_ids, max_length=85+1, do_sample=True, temperature=temperature, z_tar_edit_before_dec=z_tar_edit_before_dec)
            # print("gen_output.shape: ", gen_output.shape)
            # print("B gen_output: ", gen_output)

            for seq_ind, gen_seq in enumerate(gen_output.cpu().numpy()):
                unique_n_notrain = True
                repeat = False
                in_train_data = False

                str_token_seq = tokenizer.decode(gen_seq.tolist(), skip_special_tokens=True )

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
                input_seq_class = batch_input_seq_classes[seq_ind]
                input_seq_class_list.append(input_seq_class)
                output_seq_list.append(str_token_seq)

                seq_tensor = gen_output[seq_ind].detach().cpu()
                output_tensor_list.append(seq_tensor)

            if batch_ind % gen_save_interval == 0 and batch_ind != 0:
                save_path = os.path.join(generation_output_dir, "{}gen_seqs{}-{}.pkl".format(prepend_output_name, len(output_seq_list), num_generations))
                saved_dict = {'output_seq_list': output_seq_list, "input_seq_list": input_seq_list, "input_seq_class_list": input_seq_class_list, "output_tensor_list": output_tensor_list, 'repeat_list': repeat_list, 'in_train_data_list': in_train_data_list,  'temperature': temperature}
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

        if no_repeat_input_seq:
            break

    save_path = os.path.join(generation_output_dir, "{}gen_seqs_full{}.pkl".format(prepend_output_name, num_generations))
    saved_dict = {'output_seq_list': output_seq_list, "input_seq_list": input_seq_list, "input_seq_class_list": input_seq_class_list, "output_tensor_list": output_tensor_list, 'repeat_list': repeat_list, 'in_train_data_list': in_train_data_list, 'temperature': temperature}
    with open(save_path, 'wb') as f:
        pickle.dump(saved_dict, f)

else:
    print("Skipping generation step and loading from saved pkl")
    save_path = os.path.join(generation_output_dir, "{}gen_seqs_full{}.pkl".format(prepend_output_name, num_generations))
    with open(save_path, 'rb') as f:
        saved_dict = pickle.load(f)
    output_seq_list = saved_dict['output_seq_list']
    input_seq_list = saved_dict['input_seq_list']
    input_seq_class_list = saved_dict['input_seq_class_list']
    output_tensor_list = saved_dict['output_tensor_list']
    repeat_list = saved_dict['repeat_list']
    in_train_data_list = saved_dict['in_train_data_list']
    temperature = saved_dict['temperature']

if prev_save_path is not None:
    os.remove(prev_save_path)

# print("output_tensor_list: ", output_tensor_list)
print("output_tensor_list len: ", len(output_tensor_list))
# print("output_tensor_list shape: ", output_tensor_list.shape)
gen_tensors = output_tensor_list
# gen_tensors = torch.stack(output_tensor_list, dim=0)
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
        gen_tensor_batch = gen_tensors[batch_ind*gen_batch_size : (batch_ind+1)*gen_batch_size]
        # gen_tensor_batch = gen_tensors[batch_ind*gen_batch_size : (batch_ind+1)*gen_batch_size, 1:]
        
        gen_tensor_batch = torch.nn.utils.rnn.pad_sequence(gen_tensor_batch, batch_first=True, padding_value=0)
        gen_tensor_batch = gen_tensor_batch[:, 1:]


        gen_tensor_batch = gen_tensor_batch.to(gen_model.device)
        # print("C gen_tensor_batch: ", gen_tensor_batch)
        # gen_tensor_batch already has <cls> token (32099) at the front due to reconstruction objective
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
# Disc model set up - Start -

t5config = T5Config.from_pretrained(disc_pretrained_dir)
disc_args = {
    'latent_pooler': args.disc_latent_pooler,
}
disc_model = T5Discriminator.from_pretrained(disc_pretrained_dir, **disc_args)

disc_model.eval()

disc_model = disc_model.to(gen_model.device)
# Disc model set up - End -

# new disc
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
        gen_tensor_batch = gen_tensor_batch[:, 1:]

        gen_tensor_batch = gen_tensor_batch.to(gen_model.device)
        # print("D gen_tensor_batch: ", gen_tensor_batch)
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
        # print("batch_input_ids: ", batch_input_ids)
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





# TODO: PPL computation with GPT-2 - start - 
ppl_batch_size = 1 # only works with batch size 1 now
ppl_model = GPT2LMHeadModel.from_pretrained(ppl_model_id).to(gt_model.device)
ppl_tokenizer = GPT2TokenizerFast.from_pretrained(ppl_model_id)
gen_seq_ppl_list = []
input_seq_ppl_list = []

del gt_model

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

print("PPL compute for input sequences")
# infer input_seq ppl
with torch.no_grad():
    for batch_ind in tqdm(range(num_ppl_batch)):

        # TODO: Process input batch - start -
        input_seq_batch = input_seq_list[batch_ind*ppl_batch_size : (batch_ind+1)*ppl_batch_size]

        batch_input_ids = []
        # tokenize
        for seq in input_seq_batch:
            fixed_seq = ppl_tokenizer.decode(ppl_tokenizer.encode(seq)) # hack to remove space before punctuations (e.g. ' .' , ' ,') which inflates ppl value
            input_ids = ppl_tokenizer.encode(fixed_seq)
            input_ids = np.array(input_ids, np.int64)
            batch_input_ids.append(input_ids)

        # collate
        batch_input_ids = torch.from_numpy(pad_sequences(batch_input_ids, 0)).to(ppl_model.device)
            
        # TODO: Process input batch - end -
        ppl_output = ppl_model(input_ids=batch_input_ids, labels=batch_input_ids)
        log_likelihood = ppl_output[0]
        seq_ppl = torch.exp(log_likelihood)
        input_seq_ppl_list.append(seq_ppl.cpu().numpy())

input_seq_ppl_list = np.concatenate(input_seq_ppl_list, axis=None).tolist()
# TODO: PPL computation with GPT-2 - end - 



df = pd.DataFrame()
df['disc_pred'] = disc_pred_list
df['latent_head_pred'] = latent_head_pred_list
df['gen_input_seq_class'] = input_seq_class_list
df['gt_class_pred'] = gt_class_pred_list
df['gt_highest_prob'] = gt_highest_prob_list
df['gt_2class_pred'] = gt_2class_pred_list
df['gt_neg_prob'] = gt_neg_prob_list
df['gt_pos_prob'] = gt_pos_prob_list
df['generated_seq_ppl'] = gen_seq_ppl_list
df['input_seq_ppl'] = input_seq_ppl_list
df['generated_seq'] = output_seq_list
df['gen_input_seq'] = input_seq_list

df['repeated_gen'] = repeat_list
df['in_train_data_gen'] = in_train_data_list


# Latent head-predicted most postive ones first
df = df.sort_values(by='latent_head_pred', ascending=False)

tsv_name = os.path.join(generation_output_dir, "{}congen_seqs{}.tsv".format(prepend_output_name, num_generations))

df.to_csv(tsv_name, sep="\t", index=False)