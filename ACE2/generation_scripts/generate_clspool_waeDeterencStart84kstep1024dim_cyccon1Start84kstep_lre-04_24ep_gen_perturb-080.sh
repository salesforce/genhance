#!bin/sh

python generate_sequences_controlledGen_batchopt_targetnum.py \
    --num_generations=260000 \
    --unique_gen \
    --gen_pretrained_dir=/export/share/alvinchan/models/ACE/congen/clspool_waeDeterencStart84kstep1024dim_cyccon1Start84kstep_lre-04_24ep \
    --generation_output_dir=generated_seqs/congen/clspool_waeDeterencStart84kstep1024dim_cyccon1Start84kstep_lre-04_24ep_gen_perturb-080 \
    --prepend_output_name=unique250K_clspool_waeDeterencStart84kstep1024dim_cyccon1Start84kstep_lre-04_24ep_gen_perturb-080- \
    --latent_pooler=cls \
    --pool_enc_hidden_states_for_dec \
    --latent_space_type=wae \
    --latent_size=1024 \
    --wae_z_enc_type=deterministic \
    --no_separate_latent_enc \
    --no_separate_latent_dec \
    --z_tar_edit_before_dec=-0.80

    # based on std of value_pred: 3.182558, from analysis notebook Analyze Controlled Generators-clspool_waeDeterencStart84kstep1024dim_cyccon1Start84kstep_lre-04_24ep_19Apr