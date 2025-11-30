# Large Composite Delphi Training Config
# 6-column data format: (ID, AGE, DATA, DOSE, TOTAL, UNIT)
# For larger-scale training

model_type = 'composite'
out_dir = 'out-composite-large'

# data
dataset = 'ukb_data'
batch_size = 32
block_size = 48
gradient_accumulation_steps = 4

# model
n_layer = 12
n_head = 12
n_kv_head = 4
n_embd = 384
dropout = 0.1

# vocabulary sizes for composite model
data_vocab_size = 1500
dose_vocab_size = 16
total_vocab_size = 128
unit_vocab_size = 8

# loss weights
loss_weight_data = 1.0
loss_weight_dose = 0.5
loss_weight_total = 0.5
loss_weight_unit = 0.5
loss_weight_time = 1.0

# modern features
use_moe = True
num_experts = 8
experts_per_token = 2
sliding_window = 256

# training
learning_rate = 3e-4
max_iters = 50000
warmup_iters = 5000
lr_decay_iters = 50000
min_lr = 3e-5

# delphi specific
token_dropout = 0.0
t_min = 0.1
mask_ties = True
ignore_tokens = [0]
no_event_token_rate = 5

# system
device = 'cuda'
dtype = 'bfloat16'
compile = True

# logging
wandb_log = True
wandb_project = 'composite-delphi'

