#!bin/bash

python generate_sequences_baselines_gen_batchopt.py \
    --generation_output_dir=/export/share/alvinchan/results/generated_seqs/ACE_basegen \
    --prepend_output_name=tophalf_12ep_250K- \
    --gen_pretrained_dir=/export/share/alvinchan/models/ACE/basegen/tophalf_12ep/results/checkpoint-92000 \
    --num_generations=260000 \
    --do_latenthead_eval \
    --latenthead_pretrained_dir=/export/share/alvinchan/models/ACE/congen/clspool_waeDeterencStart84kstep1024dim_cyccon1Start84kstep_lre-04_24ep \
    --latent_pooler=cls \
    --pool_enc_hidden_states_for_dec \
    --latent_space_type=wae \
    --latent_size=1024 \
    --wae_z_enc_type=deterministic \
    --no_separate_latent_enc \
    --no_separate_latent_dec