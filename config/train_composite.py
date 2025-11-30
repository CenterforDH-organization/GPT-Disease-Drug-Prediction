# Composite Delphi Training Config  
# 6-column data format: (ID, AGE, DATA, DOSE, TOTAL, UNIT)

model_type = 'composite'
out_dir = 'out-composite'

# data
dataset = 'ukb_data'
batch_size = 64
block_size = 24

# model
n_layer = 6
n_head = 6
n_kv_head = 2
n_embd = 128
dropout = 0.2

# vocabulary sizes for composite model
data_vocab_size = 1500  # 약품/질병 코드 수
dose_vocab_size = 16    # 이산화된 dose 버킷 수
total_vocab_size = 128  # 기간 vocab
unit_vocab_size = 8     # 단위 vocab

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
sliding_window = 128

# training
learning_rate = 6e-4
max_iters = 20000
warmup_iters = 2000

# delphi specific
token_dropout = 0.0
t_min = 0.1
mask_ties = True
ignore_tokens = [0]
no_event_token_rate = 5

# system
device = 'cuda'
dtype = 'bfloat16'
compile = False

# logging
wandb_log = False

