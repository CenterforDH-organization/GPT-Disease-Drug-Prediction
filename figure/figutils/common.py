"""
common.py - Shared utilities for model & data loading
======================================================
Centralizes model loading, data loading, and dataset/dataloader creation
so that each notebook doesn't reinvent the wheel.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup: ensure project root is importable
# ---------------------------------------------------------------------------
FIGURE_DIR = Path(__file__).resolve().parent.parent  # /gpt/figure
PROJECT_ROOT = FIGURE_DIR.parent                      # /gpt

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Composite binary dtype (shared across all data loading)
# ---------------------------------------------------------------------------
COMPOSITE_DTYPE = np.dtype([
    ('ID',    np.uint32),
    ('AGE',   np.uint32),
    ('DATA',  np.uint32),
    ('SHIFT', np.uint32),
    ('TOTAL', np.uint32),
])


def load_model(ckpt_path, device='cpu', strip_prefix=True):
    """
    Load a CompositeDelphi model from a checkpoint.

    Parameters
    ----------
    ckpt_path : str or Path
        Path to the .pt checkpoint file.
    device : str
        Target device ('cpu', 'cuda', 'cuda:0', …).
    strip_prefix : bool
        If True, remove '_orig_mod.' prefix from state-dict keys
        (common when checkpoints are saved from torch.compile).

    Returns
    -------
    model : CompositeDelphi
        Model in eval mode on the requested device.
    checkpoint : dict
        Raw checkpoint dictionary (contains 'model_args', 'iter_num', etc.).
    """
    from model import CompositeDelphi, CompositeDelphiConfig

    print(f"[INFO] Loading model from {ckpt_path} → {device}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    config = CompositeDelphiConfig(**checkpoint['model_args'])
    model = CompositeDelphi(config)

    state_dict = checkpoint['model']
    if strip_prefix:
        prefix = '_orig_mod.'
        state_dict = {
            (k[len(prefix):] if k.startswith(prefix) else k): v
            for k, v in state_dict.items()
        }

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[OK]  Model loaded ({n_params:.2f}M params)")
    return model, checkpoint


def load_composite_data(data_path):
    """
    Load a composite binary dataset.

    Parameters
    ----------
    data_path : str or Path
        Path to the .bin file.

    Returns
    -------
    data_raw : np.ndarray
        Structured array with fields (ID, AGE, DATA, SHIFT, TOTAL).
    data_2d : np.ndarray
        Same data reshaped to (N, 5) uint32 — needed by get_batch variants.
    p2i : np.ndarray
        Patient-to-index pointer array.
    """
    # Import from project-root utils.py (not our figutils package)
    from utils import get_p2i_composite

    print(f"[INFO] Loading data from {data_path}")
    data_raw = np.fromfile(str(data_path), dtype=COMPOSITE_DTYPE)
    data_2d = data_raw.view(np.uint32).reshape(-1, 5)

    p2i = get_p2i_composite(data_raw)
    print(f"[OK]  {len(data_raw):,} events, {len(p2i):,} patients")
    return data_raw, data_2d, p2i


def make_dataloader(data_raw, p2i, block_size=512, batch_size=32,
                    apply_token_shift=True, max_patients=-1):
    """
    Create a PyTorch DataLoader for CompositeDelphi evaluation.

    Parameters
    ----------
    data_raw : np.ndarray
        Structured array from load_composite_data.
    p2i : np.ndarray
        Patient-to-index pointer array.
    block_size : int
        Maximum sequence length.
    batch_size : int
        Batch size.
    apply_token_shift : bool
        Whether to apply token shift (must match training config).
    max_patients : int
        Limit number of patients (-1 = all).

    Returns
    -------
    dataloader : DataLoader
    """
    from torch.utils.data import DataLoader, Dataset
    # Import from project-root utils.py
    from utils import get_batch_composite

    if 0 < max_patients < len(p2i):
        p2i = p2i[:max_patients]
        print(f"[INFO] Limited to {max_patients} patients")

    class _Dataset(Dataset):
        def __init__(self):
            self.data = data_raw
            self.p2i = p2i
            self.block_size = block_size
            self.shift = apply_token_shift

        def __len__(self):
            return len(self.p2i)

        def __getitem__(self, idx):
            ix = torch.tensor([idx])
            return get_batch_composite(
                ix, self.data, self.p2i,
                block_size=self.block_size,
                device='cpu',
                select='left',
                padding='none',
                no_event_token_rate=0,
                cut_batch=True,
                apply_token_shift=self.shift,
            )

    def _collate(batch):
        max_len = max(item[0].shape[1] for item in batch)

        def _pad(tensor, pad_val=0):
            if tensor.shape[1] < max_len:
                p = torch.full(
                    (tensor.shape[0], max_len - tensor.shape[1]),
                    pad_val, dtype=tensor.dtype,
                )
                return torch.cat([tensor, p], dim=1)
            return tensor

        # batch items: (x_data, x_shift, x_total, x_ages, y_data, y_shift, y_total, y_ages)
        pad_vals = [0, 0, 0, -10000, 0, 0, 0, -10000]
        return tuple(
            torch.cat([_pad(item[i], pad_vals[i]) for item in batch], dim=0)
            for i in range(8)
        )

    ds = _Dataset()
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=0, collate_fn=_collate)
    print(f"[OK]  DataLoader: {len(dl)} batches (bs={batch_size})")
    return dl
