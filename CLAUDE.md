# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Research Context

M.Sc. thesis project by Eliran Shmila (Reichman University, advised by Prof. Zohar Yakhini & Prof. Arik Shamir): **"Inferring Protein Expression from H&E Whole-Slide Images via Weak Supervision"**.

The core hypothesis is that morphological patterns visible in standard H&E-stained histopathology slides encode information about underlying protein expression, and a CNN can learn these associations even without spatially-resolved protein labels (weak supervision).

### BCC Dataset (Current)

- **Source**: Basal-cell carcinoma (BCC) skin cancer biopsies, 35 patients
- **Filtered cohort**: 16 patients with ≥3 valid Lysis measurements across biopsy positions
- **Proteomics**: Bulk LC-MS/MS measurements per biopsy position (not spatially registered to WSI regions)
- **Key challenge**: No direct WSI-to-proteomics-position spatial mapping — WSIs belong to patients but aren't registered to the exact biopsy positions where proteomics was measured. All tiles from a patient inherit the same label.

### Planned CyCIF Extension

A breast cancer dataset with IHC/CyCIF spatial protein labels is planned. This will require image registration between H&E and CyCIF serial slices, enabling spatially-resolved (tile-level) ground truth.

## Pipeline Overview

```
WSI TIFFs → Tissue Detection & Tiling → Stable Protein Selection (CV analysis)
  → Binary Labeling (per patient) → 5-Fold Stratified CV → ResNet18 Training
  → Multi-Run Test-Time Tile Sampling (20 runs × 100 tiles) → Correlation Evaluation
```

### Stable Protein Selection

Intra-patient coefficient of variation (CV = std/mean) is computed across biopsy positions for each protein. The top 20 lowest-CV proteins are selected as stable targets — these are proteins whose expression is consistent within a patient, making patient-level labeling meaningful.

### Binary Labeling Scheme

For each selected protein: per-patient mean expression is compared to the global median across all patients. If mean ≥ median → "high" (1), else "low" (0). All tiles from that patient inherit this binary label.

### Tile Filtering

Two-stage tissue detection: (1) global signal fraction — pixel grayscale < 240 must exceed threshold, (2) quadrant concentration check — ensures tissue is distributed across the tile, not just a sliver at the border.

### Three Normalization Types

Proteomics data is analyzed under three independent normalization pipelines: **Intensity**, **iBAQ**, and **LFQ**. Each produces its own protein tables, stable protein rankings, models, and evaluation results.

### Training & Evaluation

- **Model**: ResNet18 (ImageNet pretrained) → Dropout(0.5) → Linear(512→1) → Sigmoid; BCELoss; Adam optimizer; mixed precision
- **Cross-validation**: 5-fold StratifiedGroupKFold at patient level (prevents data leakage)
- **Tile sampling**: 100 tiles/slide for train/val; all tiles used at test time
- **Multi-run robustness**: 20 runs × 100 random tiles per test slide → distribution of metrics
- **Slide-level scores**: (1) fraction of tiles predicted "high", (2) mean prediction score
- **Evaluation metrics**: Pearson r and Spearman ρ (one-tailed p-values) between slide-level predictions and actual proteomics expression

### Key Findings

- **EF1-α 1 (LFQ)**: Strongest signal, Spearman ρ ≈ 0.45 (p ≈ 1.3e-31)
- **Thioredoxin (Intensity)**: Weak but significant, ρ ≈ 0.09 (p ≈ 0.01)
- Many proteins show no reliable morphology-to-expression inference — the approach works selectively

## Code Architecture

### WSI Processing
- `wsi_slice_detection_and_alignment.py` → `wsi_slices_processing_pipeline.py` → `process_contoured_wsis.py` — detect tissue regions, align slices, generate averaged outputs (OpenSlide + OpenCV)
- `tiler.py` — splits WSIs into 100×100 tile grid, filters for tissue content, uses ThreadPoolExecutor

### ML Pipeline (`weak_supervision_label_predictor/`)
- `create_protein_specific_dataset.py` / `balanced_dataset_creator.py` — generate 5-fold CV splits (StratifiedGroupKFold) from tiles + proteomics labels
- `protein_expression_model.py` — ResNet18 model, TileDataset, training loop
- `evaluation_report_generator.py` — tile/slide/patient-level metrics (largest file, ~40K chars)
- `multi_run_eval_flow.py` — evaluate multiple runs with different tile samples
- `multi_run_aggregation_flow.py` — aggregate results across runs and folds
- `generate_wsi_heatmap.py` — spatial prediction heatmaps overlaid on WSIs

### Data Exploration & Visualization
- `protein_expression_distribution_analysis.py`, `relevant_dataframes_generator_per_norm_type.py`, `generate_top_20_proteins_dataframes.py` — proteomics EDA and stable protein selection
- `dash_app.py` — interactive Dash dashboard (Plotly)
- `standalone_html_dashboard_generator.py` — static HTML reports

## Running Scripts

No build system or package manager config. Scripts run independently via `python <script>.py`. Each uses `if __name__ == "__main__":` blocks with hardcoded paths pointing to `C:\Users\elira\ShmilaJustSolveIt Dropbox\...\Thesis\`.

**Dash app**: `python dash_app.py` starts a local web server.

## Key Dependencies

PyTorch, torchvision, OpenSlide, OpenCV (cv2), pandas, numpy, scipy, scikit-learn, matplotlib, seaborn, plotly, dash, dash-bootstrap-components, tqdm, PIL

## Important Patterns

- **Patient-level cross-validation**: Splits stratified at patient level to prevent data leakage
- **ImageNet normalization**: All tile transforms use `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`
- **Parallel processing**: WSI processing uses ProcessPoolExecutor; tiling uses ThreadPoolExecutor
- **All paths are hardcoded absolute Windows paths** — update when running on a different machine
