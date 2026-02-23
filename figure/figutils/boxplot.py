"""
boxplot.py - AUC boxplot by ICD-10 chapter
============================================
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =============================================================================
# Constants
# =============================================================================
EXCLUDE_CHAPTERS = ["Death", "XVI"]

ROMAN_ORDER = [
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
    "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX",
    "XXI", "XXII",
]

DATASET_MAP = {
    "val":  ("Internal Validation",   "val_df_both.csv"),
    "test": ("Internal Test",         "test_df_both.csv"),
    "ext1": ("External Validation 1", "extval_jmdc_df_both.csv"),
    "ext2": ("External Validation 2", "extval_ukb_df_both.csv"),
}

# =============================================================================
# Loading helpers
# =============================================================================

def load_chapter_metadata(labels_path):
    """
    Load ICD-10 chapter metadata.

    Returns
    -------
    chapter_df : pd.DataFrame  (token_id, chapter_roman)
    color_palette : dict        chapter_roman → color hex
    """
    print(f"[INFO] Loading metadata from: {labels_path}")
    df = pd.read_csv(labels_path, header=None, sep=None, engine='python')

    if df.shape[1] == 6:
        df.columns = ['raw_idx', 'name', 'token_id', 'chapter_full', 'chapter_short', 'color']
    elif df.shape[1] == 5:
        df.columns = ['name', 'token_id', 'chapter_full', 'chapter_short', 'color']
    else:
        raise ValueError("Unexpected column count in chapter metadata")

    df['token_id'] = pd.to_numeric(df['token_id'], errors='coerce')
    df = df.dropna(subset=['token_id'])
    df['token_id'] = df['token_id'].astype(int)
    df['chapter_roman'] = df['chapter_short'].str.split('.').str[0].str.strip()
    df = df[~df['chapter_roman'].isin(EXCLUDE_CHAPTERS)]

    palette = (
        df[['chapter_roman', 'color']]
        .drop_duplicates()
        .set_index('chapter_roman')['color']
        .to_dict()
    )
    return df[['token_id', 'chapter_roman']], palette


def load_dataset(result_dir, filename, chapter_df):
    """
    Load a result CSV and merge chapter metadata.

    Returns
    -------
    pd.DataFrame or None
    """
    path = os.path.join(result_dir, filename)
    if not os.path.exists(path):
        print(f"[ERROR] Not found: {path}")
        return None

    df = pd.read_csv(path)
    if 'status' in df.columns:
        df = df[df['status'] == 'ok'].copy()
    df['auc'] = pd.to_numeric(df['auc'], errors='coerce')
    df = df.dropna(subset=['auc'])
    return df.merge(chapter_df, left_on='token', right_on='token_id', how='inner')


# =============================================================================
# Plotting
# =============================================================================

def draw_boxplot(data, color_palette, title="AUC Distribution",
                 figsize=(14, 6), save_path=None):
    """
    Draw AUC boxplot grouped by ICD-10 chapter.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    sns.set_style('whitegrid', {'grid.linestyle': ':'})
    fig = plt.figure(figsize=figsize)

    present = data['chapter_roman'].unique()
    order = [c for c in ROMAN_ORDER if c in present]

    sns.boxplot(
        data=data, x='chapter_roman', y='auc',
        hue='chapter_roman', order=order,
        palette=color_palette, width=0.5,
        linewidth=0.8, fliersize=3, legend=False,
        flierprops=dict(marker='o', markerfacecolor='none',
                        markeredgecolor='gray', markeredgewidth=0.8, alpha=0.7),
    )

    plt.axhline(0.5, ls='--', color='gray', lw=1, alpha=0.5)
    plt.ylim(0.0, 1.05)
    plt.xlabel('ICD-10 Chapter', fontsize=12)
    plt.ylabel('AUC', fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK]  Saved → {save_path}")

    return fig
