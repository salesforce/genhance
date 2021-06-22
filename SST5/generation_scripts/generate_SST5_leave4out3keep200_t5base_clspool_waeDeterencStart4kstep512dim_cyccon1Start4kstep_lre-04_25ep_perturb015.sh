#!bin/sh

python generate_SST5_sequences_controlledGen_batchopt_targetnum.py \
    --num_generations=25000 \
    --num_gen_samples_per_input=16 \
    --unique_gen \
    --tokenizer_pretrained_dir=t5-base \
    --gen_pretrained_dir=/export/share/alvinchan/models/SST5/congen/SST5_leave4out3keep200_t5base_clspool_waeDeterencStart4kstep512dim_cyccon1Start4kstep_lre-04_25ep \
    --generation_output_dir=generated_seqs/congen_SST5/SST5_leave4out3keep200_t5base_clspool_waeDeterencStart4kstep512dim_cyccon1Start4kstep_lre-04_25ep_perturb015_trainlabel2geninput \
    --prepend_output_name=16xgen25kunique_SST5_leave4out3keep200_t5base_clspool_waeDeterencStart4kstep512dim_cyccon1Start4kstep_lre-04_25ep_perturb015- \
    --latent_pooler=cls \
    --pool_enc_hidden_states_for_dec \
    --latent_space_type=wae \
    --latent_size=768 \
    --wae_z_enc_type=deterministic \
    --no_separate_latent_enc \
    --no_separate_latent_dec \
    --gen_input_labels 2 \
    --z_tar_edit_before_dec=0.15

    # based on std of value_pred: 2.561073, from analysis notebook SST5 Analyze Controlled Generators-SST5_leave4out3keep200_t5base_clspool_waeDeterencStart4kstep512dim_cyccon1Start4kstep_lre-04_25ep_29Apr
    # eval mean_diffs:  [1.054855644840253, 2.026913994960652, 1.7773462760244072]
    # train std of value_pred::  2.9451044
    # train mean_diffs:  [3.3895915586696397, 3.875208520363145, 0.15090647591899975]