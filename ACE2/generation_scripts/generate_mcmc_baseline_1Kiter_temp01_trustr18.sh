#!bin/sh

python generate_mh_mcmc.py \
    --generation_output_dir=generated_seqs/mcmc_ACE/top12500input1Kiter_temp01_trustr18 \
    --trust_radius=18 \
    --num_evo_iters=1000 \
    --num_last_iters_to_keep=20 \
    --disc_batch_size=1000 \
    --temperature=0.1 \
    --prepend_output_name=top12500input1Kiter_temp01_trustr18-
