"""Feature engineering and the scikit-learn preprocessing pipeline.

The design here follows directly from ``notebooks/eda.ipynb`` (Section 6).

Two stages, deliberately kept separate:

1. :func:`add_engineered_features` - row-local feature construction. Every
   feature it creates depends only on that passenger's own values, so it
   produces identical output whether it runs on the full training set or on a
   single-row CSV uploaded to the Streamlit app. (This is exactly why
   ``TicketGroupSize`` was excluded during EDA: it depends on which other
   passengers happen to be in the file.)

2. :func:`build_preprocessor` - a ``ColumnTransformer`` whose statistics
   (imputation values, scaling parameters, encoder categories) are *learned*.
   It must be fitted on the training split only and merely applied to the
   validation split and to any inference input.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# --- Feature groups (from notebooks/eda.ipynb, Section 6) --------------------

# Continuous numerical features -> median imputation + scaling
CONTINUOUS_FEATURES = ["Age","FamilySize",]

# Fare gets its own branch because EDA showed strong right skew
SKEWED_NUMERICAL_FEATURES = ["Fare",]

# Ordinal feature -> keep its natural ordering
ORDINAL_FEATURES = ["Pclass",]

# Already binary 0/1 -> pass through unchanged
BINARY_FEATURES = ["AgeMissing","IsAlone","CabinRecorded",]

# Low-cardinality categorical features -> impute + one-hot encode
CATEGORICAL_FEATURES = ["Sex","Embarked","TitleGrouped",]

#: Titles kept as their own category; everything else collapses to "Rare".
COMMON_TITLES = ["Mr", "Miss", "Mrs", "Master"]

#: Regex that pulls the title out of "Surname, Title. Given Names".
TITLE_PATTERN = r",\s*([^.]*)\."


# --- Transform helpers -------------------------------------------------------

def log1p_clipped(x: np.ndarray) -> np.ndarray:
    """Apply ``log1p`` after clipping negatives to zero.

    Defined at module level rather than as a lambda **on purpose**: joblib
    pickles by qualified name, and an anonymous function has none. A lambda
    here makes the whole fitted preprocessor unsaveable, which surfaces only
    when ``train.py`` tries to write its artifacts.

    The clip guards the Streamlit inference path: ``log1p`` is undefined
    below -1, and an arbitrary input CSV could contain a negative fare.
    """
    return np.log1p(np.clip(x, 0, None))


# --- Stage 1: row-local feature engineering ---------------------------------

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the engineered features identified during EDA.
    Creates "AgeMissing", "FamilySize", "IsAlone", "CabinRecorded"
    and "TitleGrouped". Every one of them is computed from the passenger's
    own row, so the result does not depend on which other rows are present.
    Parameters
    ----------
    df:
        Raw dataframe with the original Kaggle columns.

    Returns
    -------
    pandas.DataFrame
        A copy with the engineered columns added. The input is not modified.
    """
    # Work on a copy so the caller's dataframe is never modified in place
    df = df.copy()
    # Preserve whether Age was originally missing before imputation happens
    df["AgeMissing"] = df["Age"].isna().astype(int)
    # Combine family-related information into a single feature
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    # Indicate whether the passenger traveled without family members
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    # Preserve whether Cabin information was recorded
    df["CabinRecorded"] = df["Cabin"].notna().astype(int)
    # Extract the passenger title from Name
    titles = (
        df["Name"]
        .str.extract(TITLE_PATTERN, expand=False)
        .str.strip()
    )

    # Group infrequent titles into a single category
    df["TitleGrouped"] = titles.where(
        titles.isin(COMMON_TITLES),
        "Rare"
    )

    return df


# --- Stage 2: the learned pipeline ------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """Build the (unfitted) preprocessing pipeline.

    Five parallel branches, one per feature group:

    ===================  =====================================================
    Group                Steps
    ===================  =====================================================
    continuous           median impute -> standard scale
    skewed numerical     median impute -> log1p -> standard scale
    ordinal              most-frequent impute -> standard scale
    binary               passthrough
    categorical          most-frequent impute -> one-hot
    ===================  =====================================================

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Unfitted. Call ``.fit()`` on the training split only.
    """

    # Continuous numerical features: median imputation + scaling
    continuous_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Fare: median imputation + log transform + scaling
    skewed_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "log_transform",
            FunctionTransformer(
                log1p_clipped,
                feature_names_out="one-to-one",
            ),
        ),
        ("scaler", StandardScaler()),
    ])
    # Ordinal features: impute missing class values, then scale
    ordinal_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("scaler", StandardScaler()),
    ])
    # Categorical features: mode imputation + one-hot encoding
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ])

    # Apply the appropriate transformation to each feature group
    preprocessor = ColumnTransformer([
    ("continuous", continuous_pipeline, CONTINUOUS_FEATURES),
    ("skewed", skewed_pipeline, SKEWED_NUMERICAL_FEATURES),
    ("ordinal", ordinal_pipeline, ORDINAL_FEATURES),
    ("binary", "passthrough", BINARY_FEATURES),
    ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
])

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return the output feature names of a fitted preprocessor."""

    feature_names = preprocessor.get_feature_names_out()

    return [str(name) for name in feature_names]
  


# --- Persistence -------------------------------------------------------------

def save_preprocessor(
    preprocessor: ColumnTransformer,
    path: Path | str
) -> Path:
    """Persist a fitted preprocessor to disk with joblib."""

    path = Path(path)

    # Ensure the destination directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor, path)

    return path


def load_preprocessor(path: Path | str) -> ColumnTransformer:
    """Load a fitted preprocessor from disk."""

    path = Path(path)

    # Fail with a clear message if training artifacts do not exist yet
    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessor not found at '{path}'. "
            "Run `python train.py` first to generate the training artifacts."
        )

    return joblib.load(path)


# --- Smoke test --------------------------------------------------------------

if __name__ == "__main__":
    # Block 3 verification: `python -m src.preprocess` from the project root.
    from src.data import download_titanic, load_train_csv, split_train_val

    train_df, val_df = split_train_val(load_train_csv(download_titanic()))

    train_feat = add_engineered_features(train_df)
    val_feat = add_engineered_features(val_df)
    print(f"engineered columns added: "
          f"{sorted(set(train_feat.columns) - set(train_df.columns))}")

    pre = build_preprocessor()

    # THE RULE: fit on train only, transform validation.
    X_train = pre.fit_transform(train_feat)
    X_val = pre.transform(val_feat)

    print(f"X_train shape : {X_train.shape}")
    print(f"X_val shape   : {X_val.shape}")
    print(f"columns match : {X_train.shape[1] == X_val.shape[1]}  <- must be True")
    print(f"n features    : {len(get_feature_names(pre))}")
    print(f"any NaN left  : {np.isnan(np.asarray(X_train)).any()}  <- must be False")

    # Round-trip through disk. train.py saves the fitted preprocessor and the
    # Streamlit app loads it, so a preprocessor that fits but cannot be
    # pickled is still broken - this is the check that catches that.
    import tempfile

    tmp_path = Path(tempfile.gettempdir()) / "preprocessor_smoketest.joblib"
    save_preprocessor(pre, tmp_path)
    reloaded = load_preprocessor(tmp_path)
    X_val_reloaded = reloaded.transform(val_feat)
    identical = np.allclose(np.asarray(X_val), np.asarray(X_val_reloaded))
    print(f"save/load ok  : {identical}  <- must be True")
    tmp_path.unlink()
