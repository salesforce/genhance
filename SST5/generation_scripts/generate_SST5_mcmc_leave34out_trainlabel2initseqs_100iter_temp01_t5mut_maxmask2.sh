#!bin/bash

python generate_SST5_mh_mcmc.py \
    --tokenizer_pretrained_dir=t5-base \
    --disc_pretrained_dir=disc/SST5/SST5_discT5base_leave34out_lre-04_25ep \
    --disc_latent_pooler=cls \
    --generation_output_dir=generated_seqs/mcmc_SST5/SST5_mcmc_leave34out_trainlabel2initseqs_100iter_temp01_t5mut_maxmask2 \
    --prepend_output_name=100iter_temp01_t5mut_maxmask2- \
    --num_evo_iters=100 \
    --num_last_iters_to_keep=16 \
    --temperature=0.1 \
    --gen_input_labels 2 \
    --mut_type=t5 \
    --max_masked_span_len=2 \
    --t5_gen_temp=1.0 \
    --t5_gen_top_k=50

    