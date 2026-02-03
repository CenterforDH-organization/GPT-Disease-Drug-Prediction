# utils_umap.py
import os
import sys
import torch
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from collections import Counter

# =========================================================
# Add parent directory to Python path
# (Allows importing model definitions from the project root)
# =========================================================
current_dir = os.path.dirname(os.path.abspath(__file__)) # e.g., /gpt/figure
parent_dir = os.path.dirname(current_dir)                # e.g., /gpt
sys.path.append(parent_dir)

from model import CompositeDelphi, CompositeDelphiConfig

# =============================================================================
# Configuration Defaults
# =============================================================================
DEFAULT_SIZE_LEVELS = {
    'Low': {'thresh': 1000, 'size': 30, 'label': '< 1k'},
    'Mid': {'thresh': 50000, 'size': 100, 'label': '1k ~ 50k'},
    'High': {'thresh': float('inf'), 'size': 350, 'label': '> 50k'}
}

# =============================================================================
# 1. Data Loading Functions
# =============================================================================
def load_token_frequencies(data_bin_path):
    """
    Compute token frequencies from a binary training dataset.

    Parameters
    ----------
    data_bin_path : str
        Path to the binary dataset file.

    Returns
    -------
    dict
        Dictionary mapping token_id -> frequency.
    """
    print(f"[INFO] Computing token frequencies from: {data_bin_path}")
    
    if not os.path.exists(data_bin_path):
        print("[WARN] Dataset file not found. Returning empty counts.")
        return {}

    try:
        # Define composite binary structure
        composite_dtype = np.dtype([
            ('ID', np.uint32),
            ('AGE', np.uint32),
            ('DATA', np.uint32),
            ('SHIFT', np.uint32),
            ('TOTAL', np.uint32)
        ])
        
        data_raw = np.fromfile(data_bin_path, dtype=composite_dtype)
        
        # Raw token IDs + 1 correspond to model token IDs
        shifted_tokens = data_raw['DATA'] + 1
        
        unique, counts = np.unique(shifted_tokens, return_counts=True)
        token_counts = dict(zip(unique, counts))
        
        print(f"[OK] Frequency computation completed. Unique tokens: {len(token_counts)}")
        return token_counts

    except Exception as e:
        print(f"[WARN] Failed to load dataset ({e}).")
        return {}

def load_chapter_metadata(csv_path, start_token_id=22):
    """
    Load and preprocess token-to-chapter metadata.

    Parameters
    ----------
    csv_path : str
        Path to the chapter metadata CSV file.
    start_token_id : int, optional
        Minimum token ID to include (default: 22).

    Returns
    -------
    token_meta : dict
        Dictionary mapping token_id -> metadata fields.
    legend_info : pd.DataFrame
        DataFrame containing unique (chapter_short, color) pairs for legends.
    """
    print("[INFO] Loading chapter metadata...")
    try:
        df = pd.read_csv(csv_path, header=None)
        
        # Automatically infer column format
        if df.shape[1] == 6:
            df.columns = ['raw_idx', 'name', 'token_id', 'chapter_full', 'chapter_short', 'color']
        elif df.shape[1] == 5:
            df.columns = ['name', 'token_id', 'chapter_full', 'chapter_short', 'color']
        else:
            raise ValueError(f"Unexpected column count: {df.shape[1]}")

        df['token_id'] = pd.to_numeric(df['token_id'], errors='coerce')
        df = df.dropna(subset=['token_id'])
        df['token_id'] = df['token_id'].astype(int)
        
        # Filter by minimum token ID
        filtered_df = df[df['token_id'] >= start_token_id].copy()
        token_meta = filtered_df.set_index('token_id').to_dict('index')
        
        legend_info = filtered_df[['chapter_short', 'color']].drop_duplicates()
        
        return token_meta, legend_info

    except Exception as e:
        print(f"[ERROR] Failed to load chapter metadata: {e}")
        raise

# =============================================================================
# 2. Model & Embedding Functions
# =============================================================================
def get_embeddings(ckpt_path, token_meta):
    """
    Extract token embeddings from a trained model checkpoint.

    Parameters
    ----------
    ckpt_path : str
        Path to the model checkpoint.
    token_meta : dict
        Token metadata dictionary (used to select valid token IDs).

    Returns
    -------
    filtered_embeddings : np.ndarray
        Embedding matrix for valid tokens only.
    valid_token_ids : list[int]
        Sorted list of token IDs corresponding to the embeddings.
    """
    print(f"[INFO] Loading model from {ckpt_path}...")
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        config = CompositeDelphiConfig(**checkpoint['model_args'])
        model = CompositeDelphi(config)
        
        state_dict = checkpoint['model']
        # Remove '_orig_mod.' prefix if present
        new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        
        full_embeddings = model.composite_emb.data_emb.weight.detach().numpy()
        
        # Keep only tokens present in metadata
        valid_token_ids = sorted(token_meta.keys())
        filtered_embeddings = full_embeddings[valid_token_ids]
        
        return filtered_embeddings, valid_token_ids

    except Exception as e:
        print(f"[ERROR] Embedding extraction failed: {e}")
        raise

# =============================================================================
# 3. Processing & Plotting
# =============================================================================
def run_umap(embeddings, n_neighbors=15, min_dist=0.1, metric='cosine'):
    """
    Perform UMAP dimensionality reduction.

    Parameters
    ----------
    embeddings : np.ndarray
        High-dimensional embedding matrix.
    n_neighbors : int
        Number of neighbors for UMAP.
    min_dist : float
        Minimum distance parameter for UMAP.
    metric : str
        Distance metric.

    Returns
    -------
    np.ndarray
        2D UMAP embedding.
    """
    print("[INFO] Running UMAP...")
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=42)
    return reducer.fit_transform(embeddings)

def draw_umap_plot(embedding_2d, valid_token_ids, token_meta, token_counts, 
                   legend_info, target_label_ids=None, size_levels=DEFAULT_SIZE_LEVELS):
    """
    Visualize UMAP embeddings as a scatter plot.

    Parameters
    ----------
    embedding_2d : np.ndarray
        2D UMAP embedding.
    valid_token_ids : list[int]
        Token IDs corresponding to the embeddings.
    token_meta : dict
        Token metadata (names, colors, chapters).
    token_counts : dict
        Token frequency dictionary.
    legend_info : pd.DataFrame
        Chapter legend information.
    target_label_ids : list[int], optional
        Token IDs to annotate on the plot.
    size_levels : dict
        Size thresholds and marker sizes.

    Returns
    -------
    matplotlib.figure.Figure
        Rendered UMAP figure.
    """
    print("[INFO] Rendering scatter plot...")
    
    # Build plotting DataFrame
    df_plot = pd.DataFrame(embedding_2d, columns=['x', 'y'])
    df_plot['token_id'] = valid_token_ids
    df_plot['color'] = [token_meta[tid]['color'] for tid in valid_token_ids]
    df_plot['name'] = [token_meta[tid]['name'] for tid in valid_token_ids]

    # Compute marker sizes based on token frequency
    sizes = []
    for tid in valid_token_ids:
        count = token_counts.get(tid, 0)
        if count < size_levels['Low']['thresh']:
            sizes.append(size_levels['Low']['size'])
        elif count < size_levels['Mid']['thresh']:
            sizes.append(size_levels['Mid']['size'])
        else:
            sizes.append(size_levels['High']['size'])
    df_plot['size'] = sizes

    # Create figure
    fig, ax = plt.subplots(figsize=(18, 14))

    # Scatter plot
    ax.scatter(
        df_plot['x'], 
        df_plot['y'], 
        c=df_plot['color'], 
        s=df_plot['size'], 
        alpha=0.7, 
        edgecolors='white', 
        linewidth=0.3
    )

    # Annotate selected tokens
    if target_label_ids:
        texts = []
        for _, row in df_plot.iterrows():
            tid = int(row['token_id'])
            if tid in target_label_ids:
                label_text = f"{row['name']} ({tid})"
                t = ax.text(row['x'], row['y'], label_text, 
                            fontsize=10, fontweight='bold', color='black')
                texts.append(t)
        
        # Adjust text to reduce overlap (if available)
        try:
            from adjustText import adjust_text
            adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5), ax=ax)
        except ImportError:
            print("[WARN] 'adjustText' library not found. Text overlap might occur.")

    # Legend 1: Chapter colors
    color_patches = [mpatches.Patch(color=row['color'], label=row['chapter_short']) 
                     for _, row in legend_info.iterrows()]
    legend1 = ax.legend(
        handles=color_patches, 
        bbox_to_anchor=(1.02, 1), 
        loc='upper left', 
        title="Chapters",
        fontsize=9
    )
    ax.add_artist(legend1)

    # Legend 2: Token frequency (marker size)
    size_handles = []
    for level in ['Low', 'Mid', 'High']:
        info = size_levels[level]
        # Line2D markersize는 지름, scatter s는 면적 -> sqrt 변환 필요
        handle = mlines.Line2D([], [], color='white', marker='o', markerfacecolor='gray',
                               markersize=np.sqrt(info['size']), 
                               label=info['label'])
        size_handles.append(handle)

    ax.legend(
        handles=size_handles,
        bbox_to_anchor=(1.02, 0.4),
        loc='upper left',
        title="Frequency",
        fontsize=10,
        labelspacing=1.5
    )

    ax.set_title("Tokens UMAP Embedding", fontsize=20)
    plt.tight_layout()
    
    return fig