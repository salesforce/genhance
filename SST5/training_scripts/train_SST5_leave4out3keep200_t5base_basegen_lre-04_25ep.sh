#!bin/bash

python train_baseline_generator_sst5.py \
    --pretrained_dir=t5-base \
    --output_dir=/export/share/alvinchan/models/SST5/basegen/SST5_leave4out3keep200_t5base_basegen_lre-04_25ep \
    --lr=1e-04 \
    --num_train_epochs=25 \
    --train_omitted_labels 4 \
    --train_reduced_labels 3 \
    --reduced_labels_keep_num 200



