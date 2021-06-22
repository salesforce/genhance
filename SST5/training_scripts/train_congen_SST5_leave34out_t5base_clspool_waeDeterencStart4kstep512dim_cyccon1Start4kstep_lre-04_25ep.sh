#!bin/sh

python train_controlled_generator_sst5.py \
    --pretrained_dir=t5-base \
    --output_dir=./congen/SST5/SST5_leave34out_t5base_clspool_waeDeterencStart4kstep512dim_cyccon1Start4kstep_lre-04_25ep \
    --train_omitted_labels 3 4 \
    --lr=1e-04 \
    --num_train_epochs=25 \
    --lambda_contrastive=1.0 \
    --latent_pooler=cls \
    --pool_enc_hidden_states_for_dec \
    --latent_space_type=wae \
    --beta_start_step=4000 \
    --latent_size=768 \
    --wae_z_enc_type=deterministic \
    --no_separate_latent_enc \
    --no_separate_latent_dec \
    --lambda_contrastive_cyc=1.0 \
    --contrastive_cyc_start_step 4000 
    
