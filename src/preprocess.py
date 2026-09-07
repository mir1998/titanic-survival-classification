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

#: Continuous / ordinal features -> impute + scale.
NUMERICAL_FEATURES = ["Pclass", "Age", "FamilySize"]

#: Fare gets its own branch because it needs a log1p transform before scaling
#: (EDA: strongly right-skewed, max 512.3 vs median 14.5).
SKEWED_NUMERICAL_FEATURES = ["Fare"]

#: Already 0/1 -> no imputation or scaling needed, passed through as-is.
BINARY_FEATURES = ["AgeMissing", "IsAlone", "CabinRecorded"]

#: Low-cardinality categoricals -> impute + one-hot.
CATEGORICAL_FEATURES = ["Sex", "Embarked", "TitleGrouped"]

#: Titles kept as their own category; everything else collapses to "Rare".
COMMON_TITLES = ["Mr", "Miss", "Mrs", "Master"]

#: Regex that pulls the title out of "Surname, Title. Given Names".
TITLE_PATTERN = r",\s*([^.]*)\."


# --- Stage 1: row-local feature engineering ---------------------------------

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the engineered features identified during EDA.

    Creates ``AgeMissing``, ``FamilySize``, ``IsAlone``, ``CabinRecorded``
    and ``TitleGrouped``. Every one of them is computed from the passenger's
    own row, so the result does not depend on which other rows are present.

    Parameters
    ----------
    df:
        Raw dataframe with the original Kaggle columns.

    Returns
    -------
    pandas.DataFrame
        A **copy** with the engineered columns added. The input is not
        modified.
    """
    # TODO (Miriam):
    #  1. Work on a copy - never mutate the caller's dataframe.
    #  2. AgeMissing  = Age.isna() as int.
    #     ORDER MATTERS: this must be computed here, BEFORE the pipeline
    #     imputes Age. Once SimpleImputer fills the median, the information
    #     is gone. This is a silent bug if the two stages ever get swapped.
    #  3. FamilySize  = SibSp + Parch + 1
    #  4. IsAlone     = FamilySize == 1, as int
    #  5. CabinRecorded = Cabin.notna() as int
    #  6. TitleGrouped:
    #       - extract with TITLE_PATTERN (str.extract, expand=False), strip it
    #       - keep the titles in COMMON_TITLES, map everything else to "Rare"
    #       - optional refinement you flagged in EDA: Mlle/Ms -> Miss,
    #         Mme -> Mrs before the Rare bucket. Only ~3 passengers, your call.
    #  7. Return the copy.
    raise NotImplementedError


# --- Stage 2: the learned pipeline ------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """Build the (unfitted) preprocessing pipeline.

    Four parallel branches, one per feature group:

    ===================  =====================================================
    Group                Steps
    ===================  =====================================================
    numerical            median impute -> standard scale
    skewed numerical     median impute -> log1p -> standard scale
    binary               passthrough (already 0/1, derived so never missing)
    categorical          most-frequent impute -> one-hot
    ===================  =====================================================

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Unfitted. Call ``.fit()`` on the **training split only**.
    """
    # TODO (Miriam):
    #  1. numeric_pipeline = Pipeline([SimpleImputer(strategy="median"),
    #                                  StandardScaler()])
    #  2. skewed_pipeline  = same, but with a
    #     FunctionTransformer(np.log1p) between the imputer and the scaler.
    #     Note: log1p is undefined for values < -1. Titanic fares are all >= 0,
    #     but an arbitrary inference CSV could contain a negative. Decide
    #     whether to clip at 0 first - cheap insurance for the Streamlit path.
    #  3. categorical_pipeline = Pipeline([SimpleImputer(strategy="most_frequent"),
    #                                      OneHotEncoder(handle_unknown="ignore")])
    #     handle_unknown="ignore" is REQUIRED, not optional: an inference CSV
    #     may contain a Title or Embarked value never seen during training,
    #     and the default setting raises instead of encoding it as all-zeros.
    #  4. Assemble with ColumnTransformer([...]) mapping each pipeline to its
    #     feature list. Use "passthrough" for BINARY_FEATURES.
    #  5. Anything not listed is dropped (that is the default) - which is how
    #     PassengerId / Name / Ticket / Cabin / SibSp / Parch disappear.
    raise NotImplementedError


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return the output column names of a fitted preprocessor.

    Needed for ``metadata.json`` so the Streamlit app can report which
    features the model actually consumes, and to sanity-check that the
    network's input dimension matches.

    Parameters
    ----------
    preprocessor:
        A **fitted** ColumnTransformer.

    Returns
    -------
    list of str
    """
    # TODO (Miriam): sklearn gives you this almost for free -
    # look at ColumnTransformer.get_feature_names_out(). Convert to a plain
    # list of str so it is JSON-serializable.
    raise NotImplementedError


# --- Persistence -------------------------------------------------------------

def save_preprocessor(preprocessor: ColumnTransformer, path: Path | str) -> Path:
    """Persist a fitted preprocessor to disk with joblib."""
    # TODO (Miriam): mkdir the parent if needed, joblib.dump, return the Path.
    raise NotImplementedError


def load_preprocessor(path: Path | str) -> ColumnTransformer:
    """Load a fitted preprocessor from disk.

    Raises
    ------
    FileNotFoundError
        With a message telling the user to run ``python train.py`` first -
        this is the error the Streamlit app will surface if artifacts are
        missing.
    """
    # TODO (Miriam): check exists() -> clear FileNotFoundError, else joblib.load.
    raise NotImplementedError


# --- Smoke test --------------------------------------------------------------

if __name__ == "__main__":
    # Block 3 verification: `python -m src.preprocess` from the project root.
    from src.data import TARGET, download_titanic, load_train_csv, split_train_val

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
