#!bin/bash

# cp generated_seqs/congen/meanpool_gen_perturb-5/meanpool_gen_perturb-5-congen_seqs250000_filtered.tsv ../../prot5_alvin/utils/congen_data/

python foldx_stability_eval_new.py \
    --repair_pdb_dir ./repaired_lib2 \
    --foldx_batch_size 500 \
    --workers 95 \
    -i /export/home/experiments/progen_alvin/pytorch_progeny/generated_seqs/mcmc_ACE/top12500input1Kiter_temp01_trustr18/top12500input1Kiter_temp01_trustr18-mcmc_seqs_discfiltered_seqsforEmin_foldx.tsv \
    -o foldx_sim_results/top12500input1Kiter_temp01_trustr18-mcmc_seqs_discfiltered_seqsforEmin

python foldx_stability_eval_new.py \
    --repair_pdb_dir ./repaired_lib2 \
    --foldx_batch_size 250 \
    --workers 95 \
    --start_batch_ind 12 \
    -i /export/home/experiments/progen_alvin/pytorch_progeny/generated_seqs/mcmc_ACE/top12500input1Kiter_temp01_trustr18/top12500input1Kiter_temp01_trustr18-mcmc_seqs_top10Kdiscfiltered.tsv \
    -o foldx_sim_results/top12500input1Kiter_temp01_trustr18-mcmc_seqs_top10Kdiscfiltered
