# Modern/Composite Delphi Training

## Overview

This repository contains training code for two variants of the Delphi model:

1. **Modern Delphi**: Original 3-column data format (ID, AGE, TOKEN) with modern architecture improvements (RoPE, GQA, RMSNorm, SwiGLU)
2. **Composite Delphi**: New 6-column data format (ID, AGE, DATA, DOSE, TOTAL, UNIT) with multi-head output for predicting multiple fields

## Quick Start

### Modern Delphi (3-column data)

```bash
# Default training
python train_model.py

# With config file
python train_model.py config/train_modern.py

# Override specific parameters
python train_model.py --device=cuda --batch_size=64
```

### Composite Delphi (6-column data)

```bash
# Basic training
python train_model.py config/train_composite.py

# Large scale training
python train_model.py config/train_composite_large.py

# Override parameters
python train_model.py config/train_composite.py --max_iters=50000 --use_moe=True
```

## Data Format

### Modern Delphi (3-column)

Binary file with shape `(N, 3)` and dtype `np.uint32`:
- Column 0: Patient ID
- Column 1: Age (in days)
- Column 2: Token ID

### Composite Delphi (6-column)

Structured numpy array with dtype:
```python
dtype = np.dtype([
    ('ID', '<u4'),      # Patient ID
    ('AGE', '<u4'),     # Age in days
    ('DATA', '<u4'),    # Data token (drug/disease code)
    ('DOSE', '<f4'),    # Dose value
    ('TOTAL', '<u4'),   # Duration
    ('UNIT', '<u4')     # Unit code
])
```

## Model Architecture

### Modern Features

- **RoPE**: Rotary Position Embedding for better position encoding
- **GQA**: Grouped Query Attention for efficiency
- **RMSNorm**: Root Mean Square normalization
- **SwiGLU**: Gated linear unit activation
- **Sliding Window**: Local attention for long sequences
- **MoE**: Optional Mixture of Experts

### Composite Model Multi-Head Output

The Composite Delphi model predicts:
1. **DATA head**: Next data token (cross-entropy loss)
2. **DOSE head**: Next dose value (cross-entropy with discretization)
3. **TOTAL head**: Next duration (cross-entropy loss)
4. **UNIT head**: Next unit (cross-entropy loss)
5. **TIME head**: Time-to-event (exponential sampling loss)

## Configuration

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_type` | `'modern'` | `'modern'` or `'composite'` |
| `n_layer` | 6 | Number of transformer layers |
| `n_head` | 6 | Number of attention heads |
| `n_kv_head` | 2 | Number of KV heads (GQA) |
| `n_embd` | 96 | Embedding dimension |
| `use_moe` | False | Enable Mixture of Experts |
| `sliding_window` | 128 | Sliding window size (0=disabled) |

### Composite-specific Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_vocab_size` | 1500 | Size of data vocabulary |
| `dose_vocab_size` | 16 | Number of dose buckets |
| `total_vocab_size` | 128 | Size of duration vocabulary |
| `unit_vocab_size` | 8 | Size of unit vocabulary |
| `loss_weight_*` | varies | Weights for each loss component |

## Training Output

Checkpoints are saved to `out_dir`:
- `ckpt.pt`: Best model checkpoint
- `ckpt_{iter}.pt`: Periodic checkpoints every 10k iterations

## Evaluation

See `evaluate_auc.py` and `evaluate_delphi.ipynb` in the `delphi/` directory for evaluation scripts.

## WandB Logging

Enable with `--wandb_log=True`:

```bash
python train_model.py config/train_composite.py --wandb_log=True --wandb_project=my-project
```

