# Experiment 3: UKB domain adaptation fine-tuning
# Goal: Reduce external domain gap on UKB

model_type = 'composite'
init_from = 'resume'
out_dir = 'out'

# Low-LR adaptation
learning_rate = 1e-4
warmup_iters = 300
max_iters = 8000
lr_decay_iters = 8000
min_lr = 1e-5

# Domain adaptation (UKB)
domain_adaptation_enabled = True
domain_adapt_mode = 'mix'                  # 'mix' or 'finetune'
domain_adapt_data_path = '../data/UKB_extval.bin'
domain_adapt_patient_fraction = 0.1
domain_adapt_mix_ratio = 0.5
domain_adapt_start_iter = 0
domain_adapt_lr_scale = 0.2
domain_adapt_sampling_strategy = 'uniform'
domain_adapt_seed = 42

