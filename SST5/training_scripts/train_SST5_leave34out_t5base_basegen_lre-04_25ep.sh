#!bin/bash

python train_baseline_generator_sst5.py \
    --pretrained_dir=t5-base \
    --output_dir=/export/share/alvinchan/models/SST5/basegen/SST5_leave34out_t5base_basegen_lre-04_25ep \
    --lr=1e-04 \
    --num_train_epochs=25 \
    --train_omitted_labels 3 4 



