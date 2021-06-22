#!bin/bash

python train_baseline_generator.py \
    --train_ratio=0.9 \
    --data_dir=/export/share/alvinchan/data/ACE/data/gen_train_data/top_half_ddG \
    --output_dir=./gen/tophalf_12ep/results \
    --logging_dir=./gen/tophalf_12ep/logs \
    --num_train_epochs=12 