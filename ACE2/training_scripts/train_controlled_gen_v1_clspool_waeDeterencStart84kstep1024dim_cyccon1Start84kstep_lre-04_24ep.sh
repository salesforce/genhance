#!bin/sh

python train_controlled_generator.py \
    --output_dir=./congen/v1/clspool_waeDeterencStart84kstep1024dim_cyccon1Start84kstep_lre-04_24ep  \
    --lr=1e-04 \
    --num_train_epochs=24 \
    --train_ratio=0.9 \
    --lambda_contrastive=1.0 \
    --latent_pooler=cls \
    --pool_enc_hidden_states_for_dec \
    --latent_space_type=wae \
    --beta_start_step=84000 \
    --latent_size=1024 \
    --wae_z_enc_type=deterministic \
    --no_separate_latent_enc \
    --no_separate_latent_dec \
    --lambda_contrastive_cyc=1.0 \
    --contrastive_cyc_start_step 84000
