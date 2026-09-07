# Titanic Survival Classification

An end-to-end Titanic survival classification project, covering data acquisition from Kaggle, exploratory analysis, preprocessing, PyTorch model training, and a Streamlit app for validation analysis and inference on new data.

Only `train.csv` from the Kaggle competition is used. The held-out validation
set is carved out of it by a seeded stratified split; `test.csv` and
`gender_submission.csv` are never read.

---

## Results

Measured on the 179-row held-out validation split (seed 42, 80/20 stratified):

| Metric | Value |
|---|---|
| Accuracy | **0.827** |
| Precision | 0.828 |
| Recall | 0.696 |
| F1 | 0.756 |
| ROC-AUC | 0.864 |

For reference, always predicting the majority class ("did not survive") gives 0.615 accuracy on the same split.

At the default 0.5 threshold, the model favors precision over recall: when it predicts survival, it is usually correct, but it still misses some survivors. The threshold slider in the Streamlit app makes this trade-off easy to explore interactively.



### How much confidence do these numbers carry?

A bootstrap estimate over the validation predictions gives a 95% confidence interval of **[0.771, 0.883]** for accuracy (2,000 resamples).

With only 179 validation rows, small differences should be interpreted cautiously: one passenger changes accuracy by about 0.56 percentage points. For that reason, I avoided broad hyperparameter tuning against the validation split and treated it primarily as a held-out evaluation set.

At the default 0.5 threshold, the model favors precision over recall: when it predicts survival, it is usually correct, but it still misses some survivors. The threshold slider in the Streamlit app makes this trade-off easy to explore interactively.

---

## Setup

### Requirements

- Python 3.11 (developed on 3.11.4)
- A Kaggle account for downloading the dataset

### Installation

```bash
git clone https://github.com/mir1998/titanic-survival-classification.git
cd titanic-survival-classification
pip install -r requirements.txt
```

### Kaggle credentials

The dataset is downloaded in code via `kagglehub`, which requires Kaggle API credentials.

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings) → **API Tokens**
   → **Generate New Token**.
2. Save the token to `~/.kaggle/access_token` (or on Windows,
   `C:\Users\<you>\.kaggle\access_token`).
3. **Accept the competition rules** at
   [kaggle.com/competitions/titanic](https://www.kaggle.com/competitions/titanic)
   by clicking **Join Competition**. The API returns `403 Forbidden` until this
   is done, regardless of whether the token is valid.

The legacy `kaggle.json` credential format is also supported.

`data/sample_titanic.csv` is committed to the repo, so the inference screen can
be exercised without Kaggle access.

---

## Running

### Train the model

```bash
python train.py
```

This downloads the data if it is not already available locally, fits the preprocessing pipeline on the training split, trains the PyTorch model with early stopping, and writes the following artifacts to `artifacts/`:

| File | Contents |
|---|---|
| `model.pt` | Trained model weights (`state_dict`) |
| `preprocessor.joblib` | Fitted `ColumnTransformer` |
| `metadata.json` | Model architecture, hyperparameters, feature names, and validation metrics |
| `val_predictions.csv` | Per-row `y_true` and `y_prob` for the held-out validation split |

Training parameters can be overridden from the command line, for example:
```bash
python train.py --epochs 300 --lr 5e-4 --seed 7
```

### Run the Streamlit app

```bash
streamlit run ds_app.py
```

The app reads the artifacts produced by `train.py`, so training must be run at least once before launching it.

### Exploratory analysis

```bash
jupyter lab notebooks/eda.ipynb
```

---

## Example usage

### Validation Results screen

![Validation results](docs/app_validation.png)

The validation screen summarizes performance on the held-out split and lets you explore the precision-recall trade-off by changing the decision threshold interactively.

| | |
|---|---|
| ![Confusion matrix](docs/validation_confusion_matrix.png) | ![ROC curve](docs/validation_roc_curve.png) |
| ![Precision-recall curve](docs/validation_pr_curve.png) | ![Probability distribution](docs/validation_prob_distribution.png) |

### Inference screen

![Inference](docs/app_inference.png)

Enter a path to a CSV and run the saved preprocessing pipeline and model on new data. A sample file is included for a quick test:

```
data/sample_titanic.csv
```

If the CSV contains a `Survived` column, the app also shows evaluation metrics and plots. 
Without labels, it simply returns predictions, which is the normal inference workflow.

---

## Architecture

```
src/data.py          Kaggle download, loading, stratified train/val split
src/preprocess.py    Feature engineering + the fitted ColumnTransformer
src/model.py         TitanicNet (PyTorch) + weight persistence
src/evaluate.py      Metrics and plot helpers
train.py             Standalone training script -> artifacts/
ds_app.py            Streamlit app (validation + inference screens)
notebooks/eda.ipynb  Exploratory analysis
```

The `src/` modules contain the reusable logic shared by the notebook, `train.py`, and the Streamlit app.
`model.py` is kept separate from `train.py` so the app can rebuild the network and load saved weights without depending on the training loop.

### Model

A small MLP: `17 → 64 → 32 → 1`, ReLU activations, dropout 0.3, 3,265
parameters. It outputs a raw logit; the sigmoid is applied by
`BCEWithLogitsLoss` during training and by `predict_proba` at inference, which
is numerically more stable than a `Sigmoid` + `BCELoss` pair.

Training uses Adam with a learning rate of `1e-3`, batch size 32, and up to 200 epochs, with early stopping on validation loss (patience 20). In the final run, training stopped at epoch 30 and restored the best weights from epoch 10.

Given the small training set (712 rows), the model capacity is intentionally modest, and dropout is used as a simple form of regularization.

### Preprocessing

Preprocessing is split into two separate stages:

**1. Row-local feature engineering** (`add_engineered_features`) - each engineered feature is derived only from that passenger's row, so the same transformation can be applied consistently during training and inference:

| Feature | Derivation |
|---|---|
| `AgeMissing` | `Age.isna()` — computed *before* imputation, or the signal is lost |
| `FamilySize` | `SibSp + Parch + 1` |
| `IsAlone` | `FamilySize == 1` |
| `CabinRecorded` | `Cabin.notna()` |
| `TitleGrouped` | Title from `Name`; rare titles collapsed to `Rare` |

**2. A learned `ColumnTransformer`**, fitted on the training split only and then applied unchanged to the validation and inference data:

| Group | Features | Steps |
|---|---|---|
| Continuous | `Age`, `FamilySize` | median impute → scale |
| Skewed | `Fare` | median impute → `log1p` → scale |
| Ordinal | `Pclass` | mode impute → scale |
| Binary | `AgeMissing`, `IsAlone`, `CabinRecorded` | passthrough |
| Categorical | `Sex`, `Embarked`, `TitleGrouped` | mode impute → one-hot |

Output: 17 features.

---

## Exploratory analysis

Full analysis in [`notebooks/eda.ipynb`](notebooks/eda.ipynb). The split is
performed before any exploration, and every cell operates on the training
split alone. Three findings were especially useful for the preprocessing decisions that followed.

### The target is imbalanced

![Survival distribution](docs/eda_target_distribution.png)

439 of 712 training passengers did not survive (61.7%). A model that predicts
"did not survive" for everyone would therefore score 0.615 accuracy without
learning anything — which is why accuracy is reported alongside precision,
recall, F1 and ROC-AUC rather than on its own.

### Survival falls monotonically with passenger class

![Survival by passenger class](docs/eda_pclass_survival.png)

64.9%, 44.7%, 24.3% across first, second and third class. The decline is monotonic, which supports treating `Pclass` as an ordered numerical feature.
I therefore kept it as a single scaled ordinal feature rather than expanding it into three one-hot columns.

### `Fare` is heavily right-skewed

![Fare distribution](docs/eda_fare_distribution.png)

Most fares sit below 100 while the tail reaches 512.3, and the mean (31.8) is
more than double the median (14.5). This motivated the `log1p` transform to reduce skew, together with median imputation for robustness to extreme values.

---

## Design choices

### Validation data is kept separate from preprocessing and EDA

The split happens before any exploratory analysis, and every cell in the EDA notebook operates on the training split alone. Imputation values, scaling parameters, and encoder categories are all learned inside a `ColumnTransformer` fitted only on the training data.
The validation split is used only for model evaluation and early stopping, not for EDA or fitting preprocessing statistics.

### A feature was dropped because it is not reproducible at inference time

`TicketGroupSize` (how many passengers share a ticket) showed a clear
association with survival during EDA. It is excluded anyway: unlike every other
engineered feature, it depends on *which rows are present in the file* rather
than only on the passenger's own attributes.

Recomputing it on the training split alone changes the value for 15.6% of rows,
and a small inference CSV would give almost every passenger a group size of 1.
Keeping it would therefore introduce train/serve skew: the feature would mean
something different during training than it does at inference time.

### `Pclass` treated as ordinal-numeric rather than one-hot

Survival falls monotonically across passenger classes (64.9%, 44.7%, 24.3%),
so the ordering carries useful information. I therefore keep `Pclass` as a
single ordinal-numeric feature rather than expanding it into three one-hot
columns.

It is standardized before entering the network so its scale is aligned with
the other numerical inputs..

### `Fare` is log-transformed

The EDA showed heavy right skew: mean 31.8 against a median of 14.5, with a
maximum of 512.3. `log1p` compresses the long right tail before scaling.

For robustness, negative values are clipped to zero before the transform. They
are not expected in the Titanic data, but this prevents malformed inference
inputs from producing invalid values.

### The validation screen uses saved predictions

`train.py` writes `val_predictions.csv`, and the validation screen renders
metrics and plots directly from that file. This means the validation page does
not need Kaggle access or the original dataset, and the threshold slider updates
instantly because the model does not need to run again.

### Metrics beyond accuracy

Because only 38.3% of passengers survived, accuracy alone does not tell the full
story: a trivial majority-class classifier already reaches 0.615 accuracy.

I therefore report precision, recall, F1, and ROC-AUC alongside accuracy. The
Streamlit threshold slider also makes the precision/recall trade-off visible
instead of fixing the model to a single operating point.

---

## Reproducibility

`set_seed()` seeds Python's `random`, NumPy, and PyTorch, while both the
train/validation split and `DataLoader` shuffling are seeded explicitly.
`requirements.txt` pins the package versions used for the project.

The pipeline was also checked end to end by deleting `artifacts/`, rerunning
`python train.py`, and reloading the saved model and preprocessor. The regenerated
validation probabilities matched the previous run to within `3e-08`.
