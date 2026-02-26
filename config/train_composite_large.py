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
# Drug tokens: Metformin(1278)~Death(1288) → after +1 shift: 1279~1289 → vocab >= 1290
data_vocab_size = 1290   # Includes Death token (raw 1288 → shifted 1289)
dose_vocab_size = 1001   # DOSE range 0-1000 → indices 0-1000
total_vocab_size = 552   # TOTAL range 0-550 → +1 shift → indices 0-551
unit_vocab_size = 5      # UNIT range 0-3 → +1 shift → indices 0-4

# UNIT-conditioned DOSE semantics
# user-provided mapping:
# - UNIT=1 -> tab
# - UNIT=2 -> mg
# - UNIT=3 -> U di.
unit_id_tab = 1
unit_id_mg = 2
# Scaling/clamping for regression stability (match train_1206.bin observed maxima)
dose_scale_tab = 32.0
dose_scale_mg = 1000.0
dose_scale_other = 225.0

# UNIT imbalance handling (no oversampling)
# - UNIT=0 is padding/unknown
# - Recommend focal loss + class weights (approx sqrt inverse frequency)
unit_loss_type = 'focal'           # 'ce' or 'focal'
unit_ignore_index = 0
unit_focal_gamma = 2.0
unit_class_weights = [0.0, 1.0, 60.0, 6.0]  # [pad, tab, mg, U di.]

# loss weights
loss_weight_data = 1.0
loss_weight_dose = 0.5
loss_weight_total = 0.5
loss_weight_unit = 0.5
loss_weight_time = 1.0

# architecture features
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

