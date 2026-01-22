"""
Modern/Composite Delphi Training Script
Supports both ModernDelphi (3-column) and CompositeDelphi (6-column) models
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch

from model import ModernDelphi, ModernDelphiConfig, CompositeDelphi, CompositeDelphiConfig
from utils import get_p2i, get_batch, get_p2i_composite, get_batch_composite

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
wandb_project = 'modern-delphi'
wandb_run_name = 'run' + str(time.time())

# data
gradient_accumulation_steps = 1
batch_size = 96
block_size = 512

# model selection: 'modern' or 'composite'
model_type = 'composite'

# Modern Delphi model config (3-column data)
n_layer = 8          # Scaled up from 6
n_head = 6
n_kv_head = 2  # GQA (must divide n_head evenly: 6/2=3 heads per group)
n_embd = 192         # Scaled up from 96 for better capacity
dropout = 0.3
bias = False
vocab_size = 1290  # Must include Death token (raw 1288 → shifted 1289)

# Composite Delphi model config (5-column data)
data_vocab_size = 1290   # DATA: 약품/질병 코드 수 (Classification)
shift_vocab_size = 5     # SHIFT: Classification (values 0-4)
total_vocab_size = 552   # TOTAL: Embedding vocab

# SHIFT imbalance handling
shift_loss_type = 'focal'           # 'ce' or 'focal'
shift_ignore_index = 0
shift_focal_gamma = 3.0
shift_class_weights = []  # Empty list = unweighted

# Loss weights for composite model
loss_weight_data = 1.0
loss_weight_shift = 1.0
loss_weight_total = 100.0
loss_weight_time = 0.1

# modern features
use_moe = True
num_experts = 8
experts_per_token = 2
sliding_window = 128

# Drug-conditioning
use_drug_conditioning = True
rope_theta = 10000.0

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

# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, list))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

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

if model_type == 'composite':
    # 6-column structured data: (ID, AGE, DATA, DOSE, TOTAL, UNIT)
    # composite_dtype = np.dtype([
    #     ('ID', '<u4'),
    #     ('AGE', '<u4'),
    #     ('DATA', '<u4'),
    #     ('DOSE', '<f4'),
    #     ('TOTAL', '<u4'),
    #     ('UNIT', '<u4')
    # ])

    composite_dtype = np.dtype([
        ('ID', np.uint32),
        ('AGE', np.uint32),
        ('DATA', np.uint32),
        ('SHIFT', np.uint32),
        ('TOTAL', np.uint32)
    ])

    # train_data = np.memmap(TRAIN_DATA_PATH, dtype=composite_dtype, mode='r')
    # val_data = np.memmap(VAL_DATA_PATH, dtype=composite_dtype, mode='r')

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
        shift_class_weights[4] *= 2.0
        print(f"Computed shift class weights (drug-token subset): {shift_class_weights}")
else:
    # 3-column data: (ID, AGE, TOKEN)
    train_data = np.memmap(os.path.join(data_dir, TRAIN_DATA_PATH), dtype=np.uint32, mode='r').reshape(-1, 3)
    val_data = np.memmap(os.path.join(data_dir, VAL_DATA_PATH), dtype=np.uint32, mode='r').reshape(-1, 3)
    
    train_p2i = get_p2i(train_data)
    val_p2i = get_p2i(val_data)
    
    print(f"Loaded 3-column data: train={len(train_data)}, val={len(val_data)}")
    print(f"Unique patients: train={len(train_p2i)}, val={len(val_p2i)}")

# Downsample to requested fraction
if data_fraction < 1.0:
    train_p2i = train_p2i[:int(data_fraction * len(train_p2i))]
    print(f"Using {data_fraction*100:.1f}% of training data: {len(train_p2i)} patients")

iter_num = 0
best_val_loss = 1e9

# =============================================================================
# Model Initialization
# =============================================================================

if model_type == 'composite':
    # Composite Delphi with multi-head output
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
        # Composite-specific
        data_vocab_size=data_vocab_size,
        shift_vocab_size=shift_vocab_size,
        total_vocab_size=total_vocab_size,
        # SHIFT loss options
        shift_loss_type=shift_loss_type,
        shift_ignore_index=shift_ignore_index,
        shift_focal_gamma=shift_focal_gamma,
        shift_class_weights=shift_class_weights,
        loss_weight_data=loss_weight_data,
        loss_weight_shift=loss_weight_shift,
        loss_weight_total=loss_weight_total,
        loss_weight_time=loss_weight_time,
        # Time-to-Event distribution
        time_distribution=time_distribution,
    )
    
    if init_from == 'scratch':
        print("Initializing a new Composite Delphi model from scratch")
        gptconf = CompositeDelphiConfig(**model_args)
        model = CompositeDelphi(gptconf)
    elif init_from == 'resume':
        print(f"Resuming Composite Delphi training from {out_dir}")
        ckpt_path = os.path.join(out_dir, 'ckpt_composite.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        for k in ['n_layer', 'n_head', 'n_kv_head', 'n_embd', 'block_size', 'bias',
                  'data_vocab_size', 'shift_vocab_size', 'total_vocab_size']:
            if k in checkpoint_model_args:
                model_args[k] = checkpoint_model_args[k]
        gptconf = CompositeDelphiConfig(**model_args)
        model = CompositeDelphi(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
else:
    # Modern Delphi (original)
    model_args = dict(
        n_layer=n_layer,
        n_head=n_head,
        n_kv_head=n_kv_head,
        n_embd=n_embd,
        block_size=block_size,
        bias=bias,
        vocab_size=vocab_size,
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
        # Time-to-Event distribution
        time_distribution=time_distribution,
    )
    
    if init_from == 'scratch':
        print("Initializing a new Modern Delphi model from scratch")
        gptconf = ModernDelphiConfig(**model_args)
        model = ModernDelphi(gptconf)
    elif init_from == 'resume':
        print(f"Resuming Modern Delphi training from {out_dir}")
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        for k in ['n_layer', 'n_head', 'n_kv_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            if k in checkpoint_model_args:
                model_args[k] = checkpoint_model_args[k]
        gptconf = ModernDelphiConfig(**model_args)
        model = ModernDelphi(gptconf)
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
def estimate_loss_modern():
    """Estimate loss for Modern Delphi (3-column data)"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, 2)
        data = train_data if split == 'train' else val_data
        p2i = train_p2i if split == 'train' else val_p2i
        for k in range(eval_iters):
            ix = torch.randint(len(p2i), (batch_size,))
            X, A, Y, B = get_batch(ix, data, p2i, block_size=block_size,
                                   device=device, select='left',
                                   no_event_token_rate=no_event_token_rate,
                                   cut_batch=True)
            with ctx:
                logits, loss, _ = model(X, A, Y, B, validation_loss_mode=True)
            losses[k] = torch.stack([loss['loss_ce'], loss['loss_dt']])
        out[split] = losses.mean(0)
    model.train()
    return out

@torch.no_grad()
def estimate_loss_composite():
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


def estimate_loss():
    if model_type == 'composite':
        return estimate_loss_composite()
    else:
        return estimate_loss_modern()

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
print(f"Starting training: {model_type.upper()} Delphi")
print(f"{'='*60}")
print(f"  Device: {device}")
print(f"  Batch size: {batch_size}")
print(f"  Block size: {block_size}")
print(f"  Max iterations: {max_iters}")
print(f"  Learning rate: {learning_rate}")
print(f"{'='*60}\n")

# Initial batch
if model_type == 'composite':
    ix = torch.randint(len(train_p2i), (batch_size,))
    batch = get_batch_composite(ix, train_data, train_p2i, block_size=block_size, device=device,
                                padding='random', lifestyle_augmentations=True, select='left',
                                no_event_token_rate=no_event_token_rate,
                                apply_token_shift=apply_token_shift)
    x_data, x_shift, x_total, x_ages, y_data, y_shift, y_total, y_ages = batch
else:
    ix = torch.randint(len(train_p2i), (batch_size,))
    X, A, Y, B = get_batch(ix, train_data, train_p2i, block_size=block_size, device=device,
                           padding='random', lifestyle_augmentations=True, select='left',
                           no_event_token_rate=no_event_token_rate)

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
        
        if model_type == 'composite':
            # Composite model has 5 loss components
            if val_loss is None:
                val_loss_unpooled = losses['val']
            val_loss_unpooled = 0.1 * losses['val'] + 0.9 * val_loss_unpooled
            val_loss = val_loss_unpooled[0].item()  # Total loss
            
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
        else:
            # Modern model has 2 loss components (ce, dt)
            if val_loss is None:
                val_loss_unpooled = losses['val']
            val_loss_unpooled = 0.1 * losses['val'] + 0.9 * val_loss_unpooled
            val_loss = val_loss_unpooled.sum().item()
            
            print(f"step {iter_num}: train loss {losses['train'].sum().item():.4f}, val loss {losses['val'].sum().item():.4f} ({val_loss:.4f})")
            
            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/agg_loss": losses['train'].sum().item(),
                    "val/loss": val_loss,
                    "val/loss_ce": val_loss_unpooled[0].item(),
                    "val/loss_dt": val_loss_unpooled[1].item(),
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
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

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
            torch.save(checkpoint, os.path.join(out_dir, f'ckpt_{iter_num}.pt'))

    if iter_num == 0 and eval_only:
        break

    # Training step
    for micro_step in range(gradient_accumulation_steps):
        if model_type == 'composite':
            with ctx:
                logits, loss, att = model(
                    x_data, x_shift, x_total, x_ages,
                    y_data, y_shift, y_total, y_ages
                )
            
            # Prefetch next batch
            ix = torch.randint(len(train_p2i), (batch_size,))
            batch = get_batch_composite(ix, train_data, train_p2i, block_size=block_size, device=device,
                                        padding='random', lifestyle_augmentations=True, select='left',
                                        no_event_token_rate=no_event_token_rate, cut_batch=True,
                                        apply_token_shift=apply_token_shift)
            x_data, x_shift, x_total, x_ages, y_data, y_shift, y_total, y_ages = batch
            
            # Total loss
            total_loss = loss['loss']
        else:
            with ctx:
                logits, loss, att = model(X, A, Y, B)
            
            # Prefetch next batch
            ix = torch.randint(len(train_p2i), (batch_size,))
            X, A, Y, B = get_batch(ix, train_data, train_p2i, block_size=block_size, device=device,
                                   padding='random', lifestyle_augmentations=True, select='left',
                                   no_event_token_rate=no_event_token_rate, cut_batch=True)
            
            # Combined loss
            total_loss = loss['loss_ce'] + loss['loss_dt']

        scaler.scale(total_loss).backward()

    # Gradient clipping
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Gradient monitoring for debugging (every 1000 iters)
    if iter_num % 1000 == 0 and model_type == 'composite':
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
            log_dict = {
                "iter": iter_num,
                "train/loss": total_loss.item(),
                "lr": lr,
            }
            if model_type == 'composite':
                log_dict.update({
                    "train/loss_data": loss['loss_data'].item(),
                    "train/loss_shift": loss['loss_shift'].item(),
                    "train/loss_total": loss['loss_total'].item(),
                    "train/loss_time": loss['loss_time'].item(),
                })
            wandb.log(log_dict)

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
