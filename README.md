# GENhance
Deep Extrapolation for Attribute-Enhanced Generation

Objective: Generate sequences, in natural language and proteins, that go beyond the label training distribution


## Data
ACE2 250K sequences with FoldX ddG values: gs://sfr-amadani-conference-data/genhance/ACE2_subdomain_ddG_data.tar.gz
SST-5 data splits: gs://sfr-amadani-conference-data/genhance/SST5_data.tar.gz

## Models
GENhance SST-5 (leave all positives out) - gs://sfr-amadani-conference-data/genhance/GENhance_model_SST5_34.tar.gz
GENhance SST-5 (keep 200 positives) - gs://sfr-amadani-conference-data/genhance/GENhance_model_SST5_4.tar.gz
GENhance ACE2 subdomain - gs://sfr-amadani-conference-data/genhance/GENhance_model_ACE2.tar.gz

## Code overview
- `ACE/`: code and data for ACE2 experiments
- `SST5/`: code and data for SST5 experiments

  
**Requirements**
- If running on A100s, needs PyTorch 1.6 or 1.7 (with CUDA 11), requires >= 2x A100 GPUs to run
- tape proteins (tries to downgrade to PyTorch 1.4, so with `pip install --no-dependencies tape_proteins`)
- huggingface transformers
