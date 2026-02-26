"""
Composite Delphi v2 Training Script
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch

from model_v2 import (
    CompositeDelphiV2,
    CompositeDelphiV2Config,
)
from utils import get_p2i_composite, get_batch_composite

# =============================================================================
# Default Configuration
# =============================================================================

out_dir = 'out'
eval_interval = 2000
log_interval = 100
eval_iters = 200
eval_only = False
always_save_checkpoint = False
init_from = 'scratch'  # 'scratch' or 'resume'
seed = 42

# wandb logging
wandb_log = False
wandb_project = 'composite-delphi-v2'
wandb_run_name = 'run' + str(time.time())

# data
gradient_accumulation_steps = 1
batch_size = 96
block_size = 512

# model selection (composite only)
model_type = 'composite'

# Model config
n_layer = 8          # Scaled up from 6
n_head = 6
n_kv_head = 2  # GQA (must divide n_head evenly: 6/2=3 heads per group)
n_embd = 192         # Scaled up from 96 for better capacity
dropout = 0.3
bias = False
# Composite Delphi model config (5-column data)
data_vocab_size = 1290   # DATA: 약품/질병 코드 수 (Classification)
shift_vocab_size = 5     # SHIFT: Classification (values 0-4)
total_vocab_size = 552   # TOTAL: Embedding vocab

# SHIFT imbalance handling
shift_loss_type = 'focal'           # 'ce' or 'focal'
shift_ignore_index = 0
shift_focal_gamma = 2.0  # Reduced from 5.0 to standard value to prevent hallucinations
shift_class_weights = []  # Empty list = unweighted

# Loss weights for composite model
loss_weight_data = 1.0
loss_weight_shift = 15.0  # Increased from 5.0 to heavily emphasize SHIFT learning
loss_weight_total = 100.0
loss_weight_time = 0.1

# architecture features
use_moe = True
num_experts = 8
experts_per_token = 2
sliding_window = 128

# Drug-conditioning
use_drug_conditioning = True
rope_theta = 10000.0

# FPG conditioning (token ids assume padding=0; adjust if apply_token_shift=True)
use_fpg_conditioning = True
fpg_token_ids = [19, 20, 21]
fpg_condition_scale = 1.0

# adamw optimizer
learning_rate = 6e-4
max_iters = 20000        # Increased from 10000
# max_iters = 2000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay settings
decay_lr = True
warmup_iters = 1000
lr_decay_iters = 19000   # Adjusted for 20000 max_iters
min_lr = 3e-5

# system
gpu_id = 0  # GPU device ID (e.g., 0, 1, 2, ...)
device = 'cpu'  # Will be set after config parsing
# dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
dtype = 'float32'  # Force float32 for MPS stability on Mac
compile = False  # torch.compile (requires PyTorch 2.0+)

# delphi training
token_dropout = 0.0
t_min = 0.1  # Prevent log(0) numerical instability
mask_ties = True
ignore_tokens = [0]
data_fraction = 1.0
no_event_token_rate = 5
apply_token_shift = False

# Time-to-Event distribution: 'exponential' or 'weibull'
time_distribution = 'exponential'

TRAIN_DATA_PATH = '../data/kr_train.bin'
VAL_DATA_PATH = '../data/kr_val.bin'
# JMDC path for domain generalization (mixing)
JMDC_DATA_PATH = '../data/JMDC_extval.bin'

# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, list))]
exec(open('configurator.py').read())
if apply_token_shift:
    fpg_token_ids = [token_id + 1 for token_id in fpg_token_ids]
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

if model_type != 'composite':
    raise ValueError("Only composite model_type is supported.")

# Set device after config parsing (gpu_id can be overridden via command line)
if torch.cuda.is_available():
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(gpu_id)
    print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
    print("Using MPS (Apple Silicon)")
else:
    device = 'cpu'
    print("Using CPU")

tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
print(f"Tokens per iteration: {tokens_per_iter:,}")

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else ('mps' if 'mps' in device else 'cpu')
ptdtype = {'float32': torch.float32, 'float64': torch.float64,
           'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

torch.set_default_dtype(ptdtype)

# =============================================================================
# Data Loading
# =============================================================================

# data_dir = '../data'

def _compute_shift_class_weights(shift_values, shift_vocab_size, shift_ignore_index):
    counts = np.bincount(shift_values, minlength=shift_vocab_size).astype(np.float64)
    if shift_ignore_index is not None and 0 <= shift_ignore_index < shift_vocab_size:
        counts[shift_ignore_index] = 0.0
    nonzero = counts > 0
    weights = np.zeros(shift_vocab_size, dtype=np.float32)
    if nonzero.any():
        weights[nonzero] = counts[nonzero].sum() / (counts[nonzero] * nonzero.sum())
    return weights.tolist()

# 6-column structured data: (ID, AGE, DATA, DOSE, TOTAL, UNIT)
composite_dtype = np.dtype([
    ('ID', np.uint32),
    ('AGE', np.uint32),
    ('DATA', np.uint32),
    ('SHIFT', np.uint32),
    ('TOTAL', np.uint32)
])

train_data = np.fromfile(TRAIN_DATA_PATH, dtype=composite_dtype)
val_data = np.fromfile(VAL_DATA_PATH, dtype=composite_dtype)

train_p2i = get_p2i_composite(train_data)
val_p2i = get_p2i_composite(val_data)

print(f"Loaded composite data: train={len(train_data)}, val={len(val_data)}")
print(f"Unique patients: train={len(train_p2i)}, val={len(val_p2i)}")

if not shift_class_weights:
    drug_token_min = 1279 if apply_token_shift else 1278
    drug_token_max = 1289 if apply_token_shift else 1288
    drug_mask = (train_data['DATA'] >= drug_token_min) & (train_data['DATA'] <= drug_token_max)
    shift_values = train_data['SHIFT'][drug_mask].astype(np.int64)
    if apply_token_shift:
        shift_values = shift_values + 1
    shift_class_weights = _compute_shift_class_weights(
        shift_values,
        shift_vocab_size,
        shift_ignore_index,
    )
    print(f"Computed shift class weights (drug-token subset): {shift_class_weights}")

print("Computing patient-level sampling weights for SHIFT balancing...")
minority_classes = [1, 3] if not apply_token_shift else [2, 4]
patient_weights = np.zeros(len(train_p2i), dtype=np.float32)
for pid, (start_idx, length) in enumerate(train_p2i):
    patient_data = train_data[start_idx:start_idx + length]
    drug_mask = (patient_data['DATA'] >= drug_token_min) & (patient_data['DATA'] <= drug_token_max)
    patient_shifts = patient_data['SHIFT'][drug_mask]
    minority_count = sum((patient_shifts == c).sum() for c in minority_classes)
    patient_weights[pid] = 1.0 + minority_count * 1.0

patient_weights = patient_weights / patient_weights.sum()
patient_weights_tensor = torch.from_numpy(patient_weights)
minority_patient_count = (patient_weights > 1.0 / len(train_p2i)).sum()
print(f"  Patients with minority SHIFT events: {minority_patient_count:,} / {len(train_p2i):,}")
print(f"  Max sampling weight: {patient_weights.max():.4f}, Min: {patient_weights.min():.6f}")

# Downsample to requested fraction
if data_fraction < 1.0:
    train_p2i = train_p2i[:int(data_fraction * len(train_p2i))]
    print(f"Using {data_fraction*100:.1f}% of training data: {len(train_p2i)} patients")

iter_num = 0
best_val_loss = 1e9

# =============================================================================
# Model Initialization
# =============================================================================

model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_kv_head=n_kv_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    dropout=dropout,
    token_dropout=token_dropout,
    t_min=t_min,
    mask_ties=mask_ties,
    ignore_tokens=ignore_tokens,
    use_moe=use_moe,
    num_experts=num_experts,
    experts_per_token=experts_per_token,
    sliding_window=sliding_window,
    rope_theta=rope_theta,
    use_drug_conditioning=use_drug_conditioning,
    use_fpg_conditioning=use_fpg_conditioning,
    fpg_token_ids=fpg_token_ids,
    fpg_condition_scale=fpg_condition_scale,
    data_vocab_size=data_vocab_size,
    shift_vocab_size=shift_vocab_size,
    total_vocab_size=total_vocab_size,
    shift_loss_type=shift_loss_type,
    shift_ignore_index=shift_ignore_index,
    shift_focal_gamma=shift_focal_gamma,
    shift_class_weights=shift_class_weights,
    loss_weight_data=loss_weight_data,
    loss_weight_shift=loss_weight_shift,
    loss_weight_total=loss_weight_total,
    loss_weight_time=loss_weight_time,
    time_distribution=time_distribution,
)

if init_from == 'scratch':
    print("Initializing a new Composite Delphi v2 model from scratch")
    gptconf = CompositeDelphiV2Config(**model_args)
    model = CompositeDelphiV2(gptconf)
elif init_from == 'resume':
    print(f"Resuming Composite Delphi v2 training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt_composite_v2.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_kv_head', 'n_embd', 'block_size', 'bias',
              'data_vocab_size', 'shift_vocab_size', 'total_vocab_size']:
        if k in checkpoint_model_args:
            model_args[k] = checkpoint_model_args[k]
    gptconf = CompositeDelphiV2Config(**model_args)
    model = CompositeDelphiV2(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

model.to(device)

print(f"Model type: {model_type}")
print(f"Model parameters: {model.get_num_params()/1e6:.2f}M")

# =============================================================================
# Optimizer & Scaler
# =============================================================================

scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16' and device_type == 'cuda'))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

# Compile
if compile:
    print("Compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

# =============================================================================
# Loss Estimation Functions
# =============================================================================

@torch.no_grad()
def estimate_loss():
    """Estimate loss for Composite Delphi (5-column data)"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, 5)  # loss, data, shift, total, time
        data = train_data if split == 'train' else val_data
        p2i = train_p2i if split == 'train' else val_p2i
        for k in range(eval_iters):
            ix = torch.randint(len(p2i), (batch_size,))
            batch = get_batch_composite(ix, data, p2i, block_size=block_size,
                                        device=device, select='left',
                                        no_event_token_rate=no_event_token_rate,
                                        cut_batch=True,
                                        apply_token_shift=apply_token_shift)
            x_data, x_shift, x_total, x_ages, y_data, y_shift, y_total, y_ages = batch
            
            with ctx:
                logits, loss, _ = model(
                    x_data, x_shift, x_total, x_ages,
                    y_data, y_shift, y_total, y_ages,
                    validation_loss_mode=True
                )
            losses[k] = torch.stack([
                loss['loss'],
                loss['loss_data'],
                loss['loss_shift'],
                loss['loss_total'],
                loss['loss_time']
            ])
        out[split] = losses.mean(0)
    model.train()
    return out

# =============================================================================
# Learning Rate Scheduler
# =============================================================================

def get_lr(it):
    # Linear warmup
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # After decay, return min_lr
    if it > lr_decay_iters:
        return min_lr
    # Cosine decay
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# =============================================================================
# Logging Setup
# =============================================================================

if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# =============================================================================
# Training Loop
# =============================================================================

print(f"\n{'='*60}")
print("Starting training: COMPOSITE Delphi")
print(f"{'='*60}")
print(f"  Device: {device}")
print(f"  Batch size: {batch_size}")
print(f"  Block size: {block_size}")
print(f"  Max iterations: {max_iters}")
print(f"  Learning rate: {learning_rate}")
print(f"{'='*60}\n")

# Initial batch (weighted sampling for SHIFT class balance)
ix = torch.multinomial(patient_weights_tensor, batch_size, replacement=True)
batch = get_batch_composite(ix, train_data, train_p2i, block_size=block_size, device=device,
                            padding='random', lifestyle_augmentations=True, select='left',
                            no_event_token_rate=no_event_token_rate,
                            apply_token_shift=apply_token_shift)
x_data, x_shift, x_total, x_ages, y_data, y_shift, y_total, y_ages = batch

t0 = time.time()
local_iter_num = 0
val_loss = None

while True:
    # Set learning rate
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluate and checkpoint
    # Note: All processes must call estimate_loss() to avoid DDP deadlock
    if iter_num % eval_interval == 0 and iter_num > 0:
        losses = estimate_loss()

        if val_loss is None:
            val_loss_unpooled = losses['val']
        val_loss_unpooled = 0.1 * losses['val'] + 0.9 * val_loss_unpooled
        val_loss = val_loss_unpooled[0].item()

        train_breakdown = losses['train']
        val_breakdown = losses['val']
        print(f"step {iter_num}: train loss {train_breakdown[0].item():.4f}, val loss {val_breakdown[0].item():.4f} (ema {val_loss:.4f})")
        print(
            "  breakdown (train/val) - "
            f"data: {train_breakdown[1].item():.4f}/{val_breakdown[1].item():.4f}, "
            f"shift: {train_breakdown[2].item():.4f}/{val_breakdown[2].item():.4f}, "
            f"total: {train_breakdown[3].item():.4f}/{val_breakdown[3].item():.4f}, "
            f"time: {train_breakdown[4].item():.4f}/{val_breakdown[4].item():.4f}"
        )

        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'][0].item(),
                "val/loss": val_loss,
                "val/loss_data": val_loss_unpooled[1].item(),
                "val/loss_shift": val_loss_unpooled[2].item(),
                "val/loss_total": val_loss_unpooled[3].item(),
                "val/loss_time": val_loss_unpooled[4].item(),
            })

        # Save best checkpoint
        if always_save_checkpoint or val_loss < best_val_loss:
            best_val_loss = val_loss
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': val_loss,
                    'config': config,
                    'model_type': model_type,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt_composite_v2.pt'))

        # Save periodic checkpoint
        if iter_num % 10_000 == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
                'model_type': model_type,
            }
            print(f"saving periodic checkpoint to {out_dir}")
            ckpt_name = f'ckpt_composite_v2_{iter_num}.pt'
            torch.save(checkpoint, os.path.join(out_dir, ckpt_name))

    if iter_num == 0 and eval_only:
        break

    # Training step
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss, att = model(
                x_data, x_shift, x_total, x_ages,
                y_data, y_shift, y_total, y_ages
            )

        ix = torch.multinomial(patient_weights_tensor, batch_size, replacement=True)
        batch = get_batch_composite(ix, train_data, train_p2i, block_size=block_size, device=device,
                                    padding='random', lifestyle_augmentations=True, select='left',
                                    no_event_token_rate=no_event_token_rate, cut_batch=True,
                                    apply_token_shift=apply_token_shift)
        x_data, x_shift, x_total, x_ages, y_data, y_shift, y_total, y_ages = batch
        total_loss = loss['loss']

        scaler.scale(total_loss).backward()

    # Gradient clipping
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Gradient monitoring for debugging (every 1000 iters)
    if iter_num % 1000 == 0:
        grad_info = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                if 'total_head' in name or 'time_head' in name or 'time_shape_head' in name:
                    grad_norm = param.grad.norm().item()
                    grad_info.append(f"{name.split('.')[-2]}={grad_norm:.6f}")
        if grad_info:
            print(f"  [GRAD DEBUG] {', '.join(grad_info)}")

    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    
    if iter_num % log_interval == 0:
        lossf = total_loss.item()
        # Show lr trend
        if iter_num > 0 and iter_num % (log_interval * 10) == 0:
            prev_lr = get_lr(iter_num - log_interval) if decay_lr else learning_rate
            lr_change = "↑" if lr > prev_lr else "↓" if lr < prev_lr else "="
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.2e} {lr_change} (warmup: {iter_num < warmup_iters}, decay: {iter_num > warmup_iters})")
        else:
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.2e}")

        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": total_loss.item(),
                "train/loss_data": loss['loss_data'].item(),
                "train/loss_shift": loss['loss_shift'].item(),
                "train/loss_total": loss['loss_total'].item(),
                "train/loss_time": loss['loss_time'].item(),
                "lr": lr,
            })

    iter_num += 1
    local_iter_num += 1

    # Termination
    if iter_num > max_iters:
        break

print(f"\n{'='*60}")
print(f"Training completed!")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Total iterations: {iter_num}")
print(f"{'='*60}")
