#!bin/bash

python foldx_stability_eval_new.py --repair_pdb_dir ./repaired_lib2 --foldx_batch_size 1000 --workers 95 -i basegen_data/tophalf-basegen_top10K-Dscore_250Kgen_dropped.tsv -o foldx_sim_results/tophalf-basegen_top10K-Dscore_250Kgen