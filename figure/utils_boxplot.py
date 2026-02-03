import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =============================================================================
# Configuration Constants
# =============================================================================
# Chapters to exclude and canonical Roman numeral order
EXCLUDE_CHAPTERS = ["Death", "XVI"]
ROMAN_ORDER_TEMPLATE = [
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
    "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX",
    "XXI", "XXII"
]

# Dataset mapping (used as keys in notebooks)
DATASET_MAP = {
    "val":  ("Internal Validation", "val_df_both.csv"),
    "test": ("Internal Test", "test_df_both.csv"),
    "ext1": ("External Validation 1", "extval_1_df_both.csv"),
    "ext2":  ("External Validation 2", "extval_2_df_both.csv"),
}

# =============================================================================
# Helper Functions
# =============================================================================
def load_chapter_metadata(labels_path):
    """
    Load and preprocess ICD-10 chapter metadata.

    Parameters
    ----------
    labels_path : str
        Path to the chapter metadata CSV file.

    Returns
    -------
    chapter_df : pd.DataFrame
        DataFrame containing token_id and chapter_roman columns.
    color_palette : dict
        Mapping from chapter_roman -> color (for plotting).
    """
    print(f"[INFO] Loading metadata from: {labels_path}")
    try:
        df = pd.read_csv(labels_path, header=None, sep=None, engine="python")

        # Automatically infer column format        
        if df.shape[1] == 6:
            df.columns = ["raw_idx", "name", "token_id", "chapter_full", "chapter_short", "color"]
        elif df.shape[1] == 5:
            df.columns = ["name", "token_id", "chapter_full", "chapter_short", "color"]
        else:
            raise ValueError("Unexpected number of columns in chapter metadata")

        df["token_id"] = pd.to_numeric(df["token_id"], errors="coerce")
        df = df.dropna(subset=["token_id"])
        df["token_id"] = df["token_id"].astype(int)

        # Extract Roman numeral chapter (e.g., 'IX' from 'IX.SomeName')
        df["chapter_roman"] = df["chapter_short"].apply(lambda x: str(x).split(".")[0])

        # Exclude specified chapters
        df = df[~df["chapter_roman"].isin(EXCLUDE_CHAPTERS)]

        # Build color palette per chapter
        color_palette = (
            df[["chapter_roman", "color"]]
            .drop_duplicates()
            .set_index("chapter_roman")["color"]
            .to_dict()
        )

        return df[["token_id", "chapter_roman"]], color_palette

    except Exception as e:
        print(f"[ERROR] Failed to load chapter metadata: {e}")
        raise

def load_dataset(result_dir, filename, chapter_df):
    """
    Load a result CSV file and merge chapter metadata.

    Parameters
    ----------
    result_dir : str
        Directory containing result CSV files.
    filename : str
        CSV filename to load.
    chapter_df : pd.DataFrame
        DataFrame with token_id and chapter_roman columns.

    Returns
    -------
    pd.DataFrame or None
        Merged dataset with chapter information, or None if file not found.
    """
    path = os.path.join(result_dir, filename)

    if not os.path.exists(path):
        print(f"[ERROR] Dataset file not found: {path}")
        return None

    df = pd.read_csv(path)
    
    # Filter only successful runs if status column exists
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
        
    df["auc"] = pd.to_numeric(df["auc"], errors="coerce")
    df = df.dropna(subset=["auc"])

    # Merge chapter metadata
    df = df.merge(
        chapter_df,
        left_on="token",
        right_on="token_id",
        how="inner"
    )

    return df

def draw_boxplot(data, color_palette, title="AUC Distribution"):
    """
    Draw a boxplot of AUC values grouped by ICD-10 chapter.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing 'chapter_roman' and 'auc' columns.
    color_palette : dict
        Mapping from chapter_roman -> color.
    title : str, optional
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
        Generated figure object.
    """
    sns.set_style("whitegrid", {"grid.linestyle": ":"})
    fig = plt.figure(figsize=(14, 6))

    present_chapters = data["chapter_roman"].unique()
    plot_order = [c for c in ROMAN_ORDER_TEMPLATE if c in present_chapters]

    sns.boxplot(
        data=data,
        x="chapter_roman",
        y="auc",
        hue="chapter_roman",
        order=plot_order,
        palette=color_palette,
        width=0.5,
        linewidth=0.8,
        fliersize=3,
        legend=False,  # avoids seaborn FutureWarning
        flierprops=dict(
            marker="o",
            markerfacecolor="none",
            markeredgecolor="gray",
            markeredgewidth=0.8,
            alpha=0.7
        )
    )

    plt.axhline(0.5, linestyle="--", color="gray", linewidth=1, alpha=0.5)
    plt.ylim(0.0, 1.05)

    plt.xlabel("ICD-10 Chapter", fontsize=12)
    plt.ylabel("AUC", fontsize=12)
    plt.title(title, fontsize=14)

    plt.tight_layout()
    
    return fig