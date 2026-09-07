"""Dataset acquisition and splitting for the Titanic classification pipeline.

This module is the single entry point for getting data into the project. It
downloads the Kaggle Titanic competition data, loads **only** ``train.csv``,
and produces a reproducible stratified train/validation split.

Per the assignment constraints, ``test.csv`` and ``gender_submission.csv`` are
never read. The held-out validation set is carved out of ``train.csv`` by
:func:`split_train_val`.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# --- Project-level constants -------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

COMPETITION = "titanic"
TRAIN_FILENAME = "train.csv"
SAMPLE_FILENAME = "sample_titanic.csv"

TARGET = "Survived"

#: Columns we expect in the raw Kaggle ``train.csv``. Used to fail loudly and
#: early if the file we were handed is not the dataset we think it is.
EXPECTED_COLUMNS = [
    "PassengerId", "Survived", "Pclass", "Name", "Sex", "Age",
    "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked",
]

DEFAULT_SEED = 42
DEFAULT_VAL_SIZE = 0.2


# --- Acquisition -------------------------------------------------------------

def download_titanic(dest_dir: Path = DATA_DIR, force: bool = False) -> Path:
    """Download the Kaggle Titanic dataset and return the path to ``train.csv``.

    Parameters
    ----------
    dest_dir:
        Directory the CSV should end up in. Created if it does not exist.
    force:
        If ``True``, re-download even when a local copy already exists.

    Returns
    -------
    Path
        Path to the local ``train.csv``.

    Raises
    ------
    RuntimeError
        If the download fails, with a message explaining how to fix the most
        likely causes (missing API token / competition rules not accepted).
    """
    dest_dir = Path(dest_dir)
    dest_path = dest_dir / TRAIN_FILENAME

    # Reuse a local copy when we have one -> reproducible and works offline.
    if dest_path.exists() and not force:
        return dest_path

    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Imported lazily so that anyone who already has train.csv locally
        # can still `import src.data` without kagglehub installed.
        import kagglehub

        # competition_download() returns a directory, not a file path.
        downloaded_dir = Path(kagglehub.competition_download(COMPETITION))
        matches = list(downloaded_dir.rglob(TRAIN_FILENAME))
        if not matches:
            raise FileNotFoundError(
                f"'{TRAIN_FILENAME}' was not found under {downloaded_dir}"
            )
        shutil.copy(matches[0], dest_path)
    except Exception as exc:  # noqa: BLE001 - surfaced as a clear RuntimeError
        raise RuntimeError(
            "Failed to download the Titanic dataset from Kaggle.\n"
            "Checklist:\n"
            "  1. Create a Kaggle API token: kaggle.com/settings -> API -> "
            "'Create New Token'. Place the downloaded kaggle.json at "
            f"{Path.home() / '.kaggle' / 'kaggle.json'}\n"
            "  2. Accept the competition rules at "
            "kaggle.com/competitions/titanic by clicking 'Join Competition' "
            "-> the API returns 403 Forbidden until this is done.\n"
            f"Original error: {exc}"
        ) from exc

    return dest_path


# --- Loading -----------------------------------------------------------------

def load_train_csv(csv_path: Path | str) -> pd.DataFrame:
    """Load the raw ``train.csv`` and validate its shape.

    Parameters
    ----------
    csv_path:
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        The raw dataframe, unmodified.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If expected columns are missing.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"'{csv_path}' does not exist. Call download_titanic() first, "
            "or pass the correct path to the Kaggle train.csv."
        )

    df = pd.read_csv(csv_path)

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"'{csv_path}' is missing expected column(s): {missing_cols}. "
            "Make sure this is the Kaggle Titanic train.csv, not test.csv "
            "or gender_submission.csv."
        )

    return df


# --- Splitting ---------------------------------------------------------------

def split_train_val(
    df: pd.DataFrame,
    val_size: float = DEFAULT_VAL_SIZE,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataframe into train and validation sets.

    The split is stratified on the target so both sides keep the same
    survival rate, and it is seeded so the same split is reproduced by
    ``train.py`` and by the Streamlit app.

    Parameters
    ----------
    df:
        The full dataframe loaded from ``train.csv``.
    val_size:
        Fraction held out for validation.
    seed:
        Random seed controlling the split.

    Returns
    -------
    tuple of (pandas.DataFrame, pandas.DataFrame)
        ``(train_df, val_df)``, each with a fresh 0..n-1 index.
    """
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=seed,
        shuffle=True,
        stratify=df[TARGET],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# --- Sample for the repo -----------------------------------------------------

def make_sample_csv(
    df: pd.DataFrame,
    n: int = 30,
    out_path: Path | None = None,
    seed: int = DEFAULT_SEED,
) -> Path:
    """Write a small sample CSV that is safe to commit to the repository.

    The assignment asks for a ``data/`` folder containing a small sample
    dataset. This sample lets a reviewer exercise the inference screen
    without Kaggle credentials. The sample keeps both classes represented
    (stratified), so the demo isn't accidentally all-survivors or
    all-non-survivors.

    Parameters
    ----------
    df:
        Source dataframe to sample from.
    n:
        Number of rows in the sample.
    out_path:
        Destination path. Defaults to ``data/sample_titanic.csv``.
    seed:
        Random seed for the sample.

    Returns
    -------
    Path
        The path written to.
    """
    if out_path is None:
        out_path = DATA_DIR / SAMPLE_FILENAME
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = min(n, len(df))

    def _take(group: pd.DataFrame) -> pd.DataFrame:
        share = max(1, round(n * len(group) / len(df)))
        return group.sample(n=min(share, len(group)), random_state=seed)

    parts = [_take(group) for _, group in df.groupby(TARGET)]
    sample = (
        pd.concat(parts)
        .sample(frac=1, random_state=seed)  # shuffle rows
        .reset_index(drop=True)
    )

    sample.to_csv(out_path, index=False)
    return out_path


# --- Smoke test --------------------------------------------------------------

if __name__ == "__main__":
    # Block 1 verification: `python -m src.data` from the project root.
    csv = download_titanic()
    print(f"train.csv -> {csv}")

    frame = load_train_csv(csv)
    print(f"loaded shape      : {frame.shape}")
    print(f"survival rate     : {frame[TARGET].mean():.3f}")

    train_df, val_df = split_train_val(frame)
    print(f"train shape       : {train_df.shape}")
    print(f"val shape         : {val_df.shape}")
    print(f"rows accounted for: {len(train_df) + len(val_df)} (expected {len(frame)})")
    print(f"train survival    : {train_df[TARGET].mean():.3f}")
    print(f"val survival      : {val_df[TARGET].mean():.3f}  <-- should be close")

    sample = make_sample_csv(frame)
    print(f"sample written    : {sample}")
