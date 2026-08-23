# SECOM ML Prediction

This project explores the [SECOM semiconductor manufacturing dataset](seacom_dataset/secom.names) and builds a machine learning workflow to detect defective batches.

The dataset contains 1,567 samples, 591 sensor/process features, and a strongly imbalanced binary label (~14:1). The notebooks in this repository walk through exploratory data analysis, leakage-safe preprocessing, dimensionality reduction, and classification models tuned specifically for defect recall.

## Project Goal

The main objective is to identify failing batches reliably, with particular attention on recall for the defect class. In a manufacturing setting, missing a defective sample is usually more costly than flagging an extra one — so the modeling choices here (class weighting, threshold tuning) are all built around that priority rather than optimizing for accuracy.

## What's Included

- `notebooks/01_exploration.ipynb` - initial data exploration and visual analysis
- `notebooks/02_preprocessing.ipynb` - train/test split, missing-value handling, feature filtering, imputation, scaling, and PCA preparation
- `notebooks/03_anomaly_detection.ipynb` - Logistic Regression models evaluated with class weighting, SMOTE comparison, and manual decision-threshold tuning
- `reports/figures/` - saved plots from the exploration and preprocessing steps
- `notebooks/prepared_pca_datasets/` - saved PCA-transformed train/test datasets and variance artifacts
- `seacom_dataset/` - raw SECOM data and label files

## Workflow

1. Explore the raw SECOM data
   - inspect missing values
   - study class imbalance
   - review feature correlations

2. Preprocess the data carefully
   - split into train and test before any transformation
   - drop high-missing features only when they also have low label correlation
   - impute remaining missing values with KNN imputation
   - standardize features
   - apply PCA to reduce dimensionality (compared 80%, 90%, and 95% explained-variance thresholds)

3. Train and evaluate models
   - use class-weighted baselines (`class_weight='balanced'`)
   - manually lower the decision threshold on predicted probabilities to target better recall, rather than relying on the default 0.5 cutoff
   - compare performance across different PCA thresholds
   - focus on recall, precision-recall trade-off, and confusion matrices — not accuracy alone, since a trivial "always pass" model already scores ~93% accuracy on this data

## Key Ideas

- All preprocessing decisions are made on the training data only to avoid leakage.
- Correlation-based filtering uses point-biserial correlation against the binary label, computed on the training split only.
- PCA is used because many SECOM features are correlated and noisy; fewer, more concentrated components (80% variance threshold) outperformed a higher threshold (95%) on defect recall, suggesting the additional components added mostly noise for this task.
- Class imbalance is primarily handled with `class_weight='balanced'` combined with manual decision-threshold tuning (lowering the cutoff below the default 0.5), since the default cutoff under-detects the minority (defect) class regardless of how the model is trained.
- **SMOTE was tested and did not improve results** — it performed the same or slightly worse than class-weighting alone. With only ~80 real defect samples, several likely distinct failure modes, and residual high dimensionality even after PCA, SMOTE's interpolated synthetic samples added noise rather than useful signal. This is a known limitation of SMOTE on small, high-dimensional, real-world minority classes, not an implementation issue — see `reports/why_smote_underperformed.md` for the full breakdown.

## Key Results

- Best-performing setup: Logistic Regression, `class_weight='balanced'`, 80% PCA variance threshold, with the decision threshold manually lowered from the default 0.5 to catch more real defects.
- This setup caught roughly half of the real defect cases in the test set, at the cost of a substantially elevated false-positive rate — an explicit, deliberate trade-off given that missing a real defect is treated as more costly than an extra inspection.

## Requirements

Create and activate the Conda environment, then install the project dependencies:

```bash
conda create -n seacom python=3.11
conda activate seacom
pip install -r requirements.txt
```

The main packages used are:

- `pandas`
- `numpy`
- `scikit-learn`
- `imbalanced-learn`
- `matplotlib`
- `seaborn`
- `jupyter`

## How to Run

1. Create the Conda environment and activate it:
   ```bash
   conda create -n seacom python=3.11
   conda activate seacom
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebooks in order:
   - `01_exploration.ipynb`
   - `02_preprocessing.ipynb`
   - `03_anomaly_detection.ipynb`
4. Run the cells top to bottom

If you are working in Jupyter, start it with:

```bash
jupyter notebook
```

If the `seacom` environment already exists, activate it with:

```bash
conda activate seacom
```

## Dataset Notes

The SECOM dataset comes from a semiconductor manufacturing process and is commonly used for classification and fault-detection experiments. It contains many missing values, so preprocessing is a major part of the project.

The label file uses:

- `+1` for a bad/defective batch
- `-1` for a good/passing batch

Defects make up roughly 7% of the data (~1:14 imbalance), which drives most of the modeling decisions in this project.

## Outputs

The repository already includes generated figures such as:

- missing-value plots
- class imbalance plots
- correlation heatmaps

It also includes saved PCA datasets for different variance thresholds:

- 80%
- 90%
- 95%

## Repository Layout

```text
.
├── notebooks/
├── reports/
├── seacom_dataset/
├── requirements.txt
└── README.md
```

## Further Improvements Needed

- Hyperparameter tuning for Logistic Regression (`C`, L1/Lasso vs L2 penalty) via cross-validated grid search, rather than using default regularization.
- Systematic, data-driven threshold selection (e.g., via precision-recall curve analysis) instead of a manually chosen cutoff.
- Evaluate additional model families (Random Forest, XGBoost) with imbalance-aware configurations for comparison against Logistic Regression.
- Cross-validated recall/F2 estimates instead of a single train/test split, since the test set contains only 21 defect samples and single-split results can shift noticeably from sample to sample.

## Notes

- Some notebook cells use absolute local file paths. If you move the project, update those paths to use relative paths.
- The repository does not currently include a packaged training script or app entrypoint; the notebooks are the main source of truth.
- Folder name `seacom_dataset/` is used as-is throughout this repo for consistency with existing file paths; double-check if you intended `secom_dataset/` when setting this up.