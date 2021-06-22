###  Training scripts
- `train_controlled_generator.py`: trains GENhance models  
- `train_baseline_generator.py`: trains baseline generator models  
- `python drive_train_ddG_discriminator.py`: trains baseline discrminator models  
- `training_scripts/train_basegen_tophalf.sh`: script to train baseline generator reported in paper
- `training_scripts/train_controlled_gen_v1_clspool_waeDeterencStart84kstep1024dim_cyccon1Start84kstep_lre-04_24ep.sh`: script to train GENhance  


###  Generations scripts
- `generate_SST5_sequences_baselines_gen.py`: generates from baseline gen-disc models  
- `generate_SST5_sequences_controlledGen.py`: generates from GENhance models  
- `generate_SST5_mh_mcmc.py`: generates from MCMC models  
- `generation_scripts/generate_250Kseqs_tophalf_basegen.sh`: script to generate from the baseline gen-disc method  
- `generation_scripts/generate_mcmc_baseline_1Kiter_temp01_trustr18.sh`: script to generate from the MCMC method  
- `generation_scripts/generate_clspool_waeDeterencStart84kstep1024dim_cyccon1Start84kstep_lre-04_24ep_gen_perturb-080.sh`: script to generate from the GENhance  
- `Analyze Controlled Generators*`: Compute standard deviation of Z_parallel to determine how much to perturb during GENhance generation.  
