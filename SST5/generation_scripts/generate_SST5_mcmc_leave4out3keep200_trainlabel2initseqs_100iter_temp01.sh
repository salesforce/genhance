#!bin/bash

python generate_SST5_mh_mcmc.py \
    --tokenizer_pretrained_dir=t5-base \
    --disc_pretrained_dir=disc/SST5/SST5_discT5base_leave4out3keep200_lre-04_25ep \
    --disc_latent_pooler=cls \
    --generation_output_dir=generated_seqs/mcmc_SST5/SST5_mcmc_trainlabel2initseqs_100iter_temp01 \
    --prepend_output_name=100iter_temp01- \
    --num_evo_iters=100 \
    --num_last_iters_to_keep=16 \
    --temperature=0.1 \
    --gen_input_labels 2
    