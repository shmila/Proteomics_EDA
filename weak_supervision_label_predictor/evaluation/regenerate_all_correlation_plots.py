"""
Batch script that regenerates correlation plots for every trained
(protein, normalization) pair.

Loops through the hardcoded PROTEIN_NORM_PAIRS list and calls
generate_correlation_plots.main() for each dataset directory,
saving results into a single output directory with per-pair filenames.

Configure the paths and pair list in `main()` and run:
    python -m weak_supervision_label_predictor.evaluation.regenerate_all_correlation_plots
"""

from pathlib import Path
import pandas as pd

from weak_supervision_label_predictor.evaluation.multi_run_aggregator import (
    create_correlation_plots,
)


PROTEIN_NORM_PAIRS = [
    ("14-3-3 protein beta_alpha", "LFQ"),
    ("Actin, cytoplasmic 2", "Intensity"),
    ("Annexin A7", "Intensity"),
    ("Cornulin", "Intensity"),
    ("Cornulin", "LFQ"),
    ("Elongation factor 1-alpha 1", "LFQ"),
    ("Gelsolin", "Intensity"),
    ("Proliferation marker protein Ki-67", "Intensity"),
    ("Proliferation marker protein Ki-67", "LFQ"),
    ("Thioredoxin", "Intensity"),
    ("Ubiquitin-conjugating enzyme E2 L3", "LFQ"),
]


def _safe_name(protein, norm):
    return protein.replace(" ", "_").replace(",", "") + "_" + norm


def regenerate_all(datasets_root, output_dir, pairs=PROTEIN_NORM_PAIRS):
    """
    Args:
        datasets_root: Directory containing the `tiles_dataset_100x100_<Protein>_<Norm>_cv`
                       subdirectories.
        output_dir: Single directory where all PNGs will be saved.
                    Files are named `<Protein>_<Norm>__by_fold.png` /
                    `<Protein>_<Norm>__by_slide.png`.
        pairs: List of (protein, norm) tuples to regenerate. Defaults to
               PROTEIN_NORM_PAIRS above.
    """
    import tempfile
    import shutil

    datasets_root = Path(datasets_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for protein, norm in pairs:
        dataset_dir = datasets_root / f"tiles_dataset_100x100_ {protein}_{norm}_cv"
        if not dataset_dir.exists():
            print(f"SKIP (missing): {dataset_dir}")
            continue

        candidates = sorted(dataset_dir.glob("consolidated_evaluation_*"),
                             key=lambda x: x.stat().st_mtime)
        if not candidates:
            print(f"SKIP (no evaluation): {protein} ({norm})")
            continue

        csv_path = candidates[-1] / "all_slide_metrics.csv"
        if not csv_path.exists():
            print(f"SKIP (no CSV): {protein} ({norm})")
            continue

        slides_df = pd.read_csv(csv_path)

        # Write to a temp dir first, then rename with the per-pair prefix
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            create_correlation_plots(slides_df, tmp)
            prefix = _safe_name(protein, norm)
            for src in tmp.glob("correlation_plots__*.png"):
                dst_name = src.name.replace("correlation_plots__", f"{prefix}__")
                shutil.copy2(src, output_dir / dst_name)

        print(f"OK: {protein} ({norm})")

    total = len(list(output_dir.glob("*.png")))
    print(f"Done. Total PNGs in {output_dir}: {total}")


def main(datasets_root, output_dir):
    """Entry point — edit the arguments below to run."""
    regenerate_all(datasets_root=datasets_root, output_dir=output_dir)


if __name__ == "__main__":
    # ---- Configure paths here ----
    DATASETS_ROOT = r"C:\path\to\dataset"
    OUTPUT_DIR = r"C:\path\to\output"

    main(
        datasets_root=DATASETS_ROOT,
        output_dir=OUTPUT_DIR,
    )
