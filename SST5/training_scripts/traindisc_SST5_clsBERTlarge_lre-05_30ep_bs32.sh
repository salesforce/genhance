#!bin/sh

python train_classifier_sst5.py \
    --pretrained_dir=bert-large-uncased \
    --output_dir=/export/share/alvinchan/models/SST5/disc/SST5_clsBERTlarge_lre-05_30ep_bs32_rerunwneutralacc \
    --lr=1e-05 \
    --per_device_train_batch_size=32 \
    --num_train_epochs=30
