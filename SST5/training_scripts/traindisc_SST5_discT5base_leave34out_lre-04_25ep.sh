#!bin/sh

python train_discriminator_sst5.py \
    --pretrained_dir=t5-base \
    --output_dir=/export/share/alvinchan/models/SST5/disc/SST5_discT5base_leave34out_lre-04_25ep \
    --lr=1e-04 \
    --latent_pooler=cls \
    --num_train_epochs=25 \
    --train_omitted_labels 3 4 
