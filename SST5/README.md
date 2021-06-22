##  Script names
Scripts with `leave34out` stand for scenario where all positive samples are left out in the training data while scripts with `leave4out3keep200` stand for scenario where all Strong-Positive samples are left out and 200 randomly sampled Weak-Positive samples are kept in the training data.  

###  Training scripts
- `train_controlled_generator_sst5.py`: trains GENhance models  
- `train_classifier_sst5.py`: trains ground-truth sentiment oracle  
- `train_discriminator_sst5.py`: trains baseline discrminator models  
- `train_baseline_generator_sst5.py`: trains baseline generator models  
- `training_scripts/train_SST5_leave4out3keep200_t5base_basegen_lre-04_25ep.sh`: script to train baseline generator reported in paper  
- `training_scripts/traindisc_SST5_discT5base_leave4out3keep200_lre-04_25ep.sh`: script to train baseline discrminator reported in paper  
- `training_scripts/train_congen_SST5_leave4out3keep200_t5base_clspool_waeDeterencStart4kstep512dim_cyccon1Start4kstep_lre-04_25ep.sh`: script to train GENhance, for 200-Pos setup  


###  Generations scripts
- `generate_SST5_sequences_baselines_gen.py`: generates from baseline gen-disc models  
- `generate_SST5_sequences_controlledGen.py`: generates from GENhance models  
- `generate_SST5_mh_mcmc.py`: generates from MCMC models  
- `generation_scripts/generate_SST5_leave4out3keep200_t5base_basegen_lre-04_25ep_25Kgen.sh`: script to generate and rank generations from the baseline gen-disc method  
- `generation_scripts/generate_SST5_mcmc_leave4out3keep200_trainlabel2initseqs_100iter_temp01_t5mut_maxmask2.sh`: script to generate and rank generations from the MCMC-T5 method  
- `generation_scripts/generate_SST5_mcmc_leave4out3keep200_trainlabel2initseqs_100iter_temp01.sh`: script to generate and rank generations from the MCMC-Random method  
- `generation_scripts/generate_SST5_leave4out3keep200_t5base_clspool_waeDeterencStart4kstep512dim_cyccon1Start4kstep_lre-04_25ep_perturb015.sh`: script to generate and rank generations from the GENhance, for 200-Pos setup  


##  Notebooks
- `SST5 Analyze Controlled Generators*`: Compute standard deviation of Z_parallel to determine how much to perturb during GENhance generation.  
- `SST5 Analyze perturbed generations*`: Compute the metrics reported in the paper for GENhance generations.  
- `SST5 Analyze MCMC*`: Compute the metrics reported in the paper for MCMC generations.  
- `SST5/SST5 Analyze baseline generations.ipynb`: Compute the metrics reported in the paper for baseline gen-disc generations.  
