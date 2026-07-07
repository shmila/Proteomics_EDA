"""
Standalone script that regenerates correlation plots for a given
(protein, normalization) dataset directory.

Reads the latest consolidated_evaluation_*/all_slide_metrics.csv under
`dataset_dir` and writes correlation_plots__by_fold.png and
correlation_plots__by_slide.png to `output_dir`.

Configure the paths in `main()` and run:
    python -m weak_supervision_label_predictor.evaluation.generate_correlation_plots
"""

from pathlib import Path
import pandas as pd

from weak_supervision_label_predictor.evaluation.multi_run_aggregator import (
    create_correlation_plots,
)


def generate_plots_for_dataset(dataset_dir, output_dir,
                                consolidated_dir=None,
                                slide_metrics_csv=None):
    """
    Args:
        dataset_dir: Path to a `tiles_dataset_100x100_<Protein>_<Norm>_cv` directory.
        output_dir: Where the two PNGs will be saved.
        consolidated_dir: Optional. Explicit path to a consolidated_evaluation_*
            subdirectory. If None, the most recent one under `dataset_dir` is used.
        slide_metrics_csv: Optional. Explicit path to an `all_slide_metrics.csv` file.
            Overrides `dataset_dir` / `consolidated_dir` when provided.
    """
    dataset_dir = Path(dataset_dir) if dataset_dir else None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if slide_metrics_csv:
        csv_path = Path(slide_metrics_csv)
    else:
        if consolidated_dir:
            consol = Path(consolidated_dir)
        else:
            candidates = sorted(dataset_dir.glob("consolidated_evaluation_*"),
                                 key=lambda x: x.stat().st_mtime)
            if not candidates:
                raise FileNotFoundError(
                    f"No consolidated_evaluation_* subdirectories found in {dataset_dir}")
            consol = candidates[-1]
        csv_path = consol / "all_slide_metrics.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Slide metrics CSV not found: {csv_path}")

    slides_df = pd.read_csv(csv_path)
    print(f"Loaded {len(slides_df)} rows from {csv_path}")

    create_correlation_plots(slides_df, output_dir)
    print(f"Wrote correlation plots to {output_dir}")


def main(dataset_dir,
         output_dir,
         consolidated_dir=None,
         slide_metrics_csv=None):
    """Entry point — edit the arguments below to run."""
    generate_plots_for_dataset(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        consolidated_dir=consolidated_dir,
        slide_metrics_csv=slide_metrics_csv,
    )


if __name__ == "__main__":
    # ---- Configure paths here ----
    DATASET_DIR = r"C:\path\to\tiles_dataset_100x100_ Thioredoxin_Intensity_cv"
    OUTPUT_DIR = r"C:\path\to\output"

    main(
        dataset_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR,
    )
