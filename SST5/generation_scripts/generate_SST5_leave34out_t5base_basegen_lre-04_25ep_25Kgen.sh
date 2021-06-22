#!bin/bash

python generate_SST5_sequences_baselines_gen_batchopt.py \
    --tokenizer_pretrained_dir=t5-base \
    --gen_pretrained_dir=/export/share/alvinchan/models/SST5/basegen/SST5_leave34out_t5base_basegen_lre-04_25ep \
    --num_generations=25000 \
    --generation_output_dir=generated_seqs/basegen_SST5/SST5_leave34out_t5base_basegen_lre-04_25ep \
    --prepend_output_name=25Kgen_SST5_leave34out_t5base_basegen_lre-04_25ep- \
    --disc_pretrained_dir=disc/SST5/SST5_discT5base_leave34out_lre-04_25ep \
    --disc_latent_pooler=cls \
    --gen_batch_size=1600 
    