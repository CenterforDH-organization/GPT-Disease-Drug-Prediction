# Modern Delphi Training Config
# 3-column data format: (ID, AGE, TOKEN)

model_type = 'modern'
out_dir = 'out-modern'

# data
dataset = 'ukb_data'
batch_size = 96
block_size = 24

# model
n_layer = 6
n_head = 6
n_kv_head = 2
n_embd = 96
dropout = 0.2
vocab_size = 1290  # Includes Death token (raw 1288 → shifted 1289)

# modern features
use_moe = False
sliding_window = 128

# training
learning_rate = 6e-4
max_iters = 10000
warmup_iters = 2000

# delphi specific
token_dropout = 0.0
t_min = 0.0
mask_ties = True
ignore_tokens = [0]
no_event_token_rate = 5

# system
device = 'cuda'
dtype = 'bfloat16'
compile = False

# logging
wandb_log = False

