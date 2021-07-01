'''
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import tape
from tape_train_new import run_train

def main():

    run_train(data_dir='data/250K_full', test_name=None, model_type="transformer", task="stability", learning_rate=5e-6, sgd=False, from_pretrained="bert-base", num_train_epochs=10,
            batch_size=32, warmup_steps=200, dropout=0.0, model_parallel=True, prior_decay=0.0, weight_decay=0.0, max_grad_norm=10,
            seed=42, init_model = "/export/share/bkrause/progen/progeny/t5_base_uniref_bfd50/")

main()
