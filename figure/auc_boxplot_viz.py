import os
import argparse
import logging

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# =============================================================================
# Configuration
# =============================================================================
RESULT_DIR = "./results"
LABELS_CHAPTER_PATH = "labels_chapter.csv"

DATASET_MAP = {
    "val":  ("Internal Validation", "val_df_both.csv"),
    "test": ("Internal Test", "test_df_both.csv"),
    "jmdc": ("External Validation (JMDC)", "extval_jmdc_df_both.csv"),
    "ukb":  ("External Validation (UKB)", "extval_ukb_df_both.csv"),
}

EXCLUDE_CHAPTERS = ["Death", "XVI"]

ROMAN_ORDER_TEMPLATE = [
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
    "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX",
    "XXI", "XXII"
]


# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)


# =============================================================================
# Argument Parsing
# =============================================================================
parser = argparse.ArgumentParser(
    description="Generate AUC boxplots by ICD-10 chapter"
)
parser.add_argument(
    "--target",
    type=str,
    default="val",
    choices=DATASET_MAP.keys(),
    help="Target dataset to visualize"
)
args = parser.parse_args()

TARGET_TITLE, TARGET_FILE = DATASET_MAP[args.target]
OUTPUT_FILE = f"auc_boxplot_{args.target}.png"


# =============================================================================
# Chapter Metadata Loading
# =============================================================================
def load_chapter_metadata():
    logging.info("Loading ICD-10 chapter metadata")

    try:
        df = pd.read_csv(
            LABELS_CHAPTER_PATH,
            header=None,
            sep=None,
            engine="python"
        )

        if df.shape[1] == 6:
            df.columns = [
                "raw_idx", "name", "token_id",
                "chapter_full", "chapter_short", "color"
            ]
        elif df.shape[1] == 5:
            df.columns = [
                "name", "token_id",
                "chapter_full", "chapter_short", "color"
            ]
        else:
            raise ValueError("Unexpected number of columns in chapter metadata")

        df["token_id"] = pd.to_numeric(df["token_id"], errors="coerce")
        df = df.dropna(subset=["token_id"])
        df["token_id"] = df["token_id"].astype(int)

        df["chapter_roman"] = df["chapter_short"].apply(
            lambda x: str(x).split(".")[0]
        )

        df = df[~df["chapter_roman"].isin(EXCLUDE_CHAPTERS)]

        color_palette = (
            df[["chapter_roman", "color"]]
            .drop_duplicates()
            .set_index("chapter_roman")["color"]
            .to_dict()
        )

        return df[["token_id", "chapter_roman"]], color_palette

    except Exception as e:
        logging.error(f"Failed to load chapter metadata: {e}")
        raise


# =============================================================================
# Dataset Loading
# =============================================================================
def load_dataset(filename, chapter_df):
    path = os.path.join(RESULT_DIR, filename)

    if not os.path.exists(path):
        logging.error(f"Dataset file not found: {path}")
        return None

    df = pd.read_csv(path)
    df = df[df["status"] == "ok"].copy()
    df["auc"] = pd.to_numeric(df["auc"], errors="coerce")
    df = df.dropna(subset=["auc"])

    df = df.merge(
        chapter_df,
        left_on="token",
        right_on="token_id",
        how="inner"
    )

    return df


# =============================================================================
# Plotting
# =============================================================================
def plot_auc_boxplot(data, color_palette):
    sns.set_style("whitegrid", {"grid.linestyle": ":"})
    plt.figure(figsize=(14, 6))

    present_chapters = data["chapter_roman"].unique()
    plot_order = [
        c for c in ROMAN_ORDER_TEMPLATE if c in present_chapters
    ]

    sns.boxplot(
        data=data,
        x="chapter_roman",
        y="auc",
        order=plot_order,
        palette=color_palette,
        width=0.5,
        linewidth=0.8,
        fliersize=3,
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

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    plt.close()

    logging.info(f"Figure saved to {OUTPUT_FILE}")


# =============================================================================
# Main
# =============================================================================
def main():
    chapter_df, color_palette = load_chapter_metadata()
    data = load_dataset(TARGET_FILE, chapter_df)

    if data is None or data.empty:
        logging.warning("No valid data available. Exiting.")
        return

    logging.info(f"Generating AUC boxplot: {TARGET_TITLE}")
    plot_auc_boxplot(data, color_palette)


if __name__ == "__main__":
    main()
