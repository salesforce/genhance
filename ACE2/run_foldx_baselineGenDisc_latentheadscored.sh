#!bin/bash

# cp generated_seqs/congen/meanpool_gen_perturb-5/meanpool_gen_perturb-5-congen_seqs250000_filtered.tsv ../../prot5_alvin/utils/congen_data/

python foldx_stability_eval_new.py \
    --repair_pdb_dir ./repaired_lib2 \
    --foldx_batch_size 250 \
    --workers 95 \
    -i /export/home/experiments/progen_alvin/pytorch_progeny/generated_seqs/baseline_gen/tophalf-basegen_top10Klatentheadfiltered.tsv \
    -o foldx_sim_results/tophalf-basegen_top10Klatentheadfiltered
