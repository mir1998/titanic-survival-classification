"""Dataset acquisition and splitting for the Titanic classification pipeline.

This module is the single entry point for getting data into the project. It
downloads the Kaggle Titanic competition data, loads **only** ``train.csv``,
and produces a reproducible stratified train/validation split.

Per the assignment constraints, ``test.csv`` and ``gender_submission.csv`` are
never read. The held-out validation set is carved out of ``train.csv`` by
:func:`split_train_val`.
"""

from __future__ import annotations

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
    # TODO (מרים):
    #  1. אם dest_dir/train.csv כבר קיים ו-force הוא False -> להחזיר אותו מיד.
    #     (למה: לא רוצים להוריד מחדש בכל הרצה. זה גם מה שמאפשר לעבוד אופליין.)
    #  2. ליצור את dest_dir אם הוא לא קיים -> mkdir(parents=True, exist_ok=True)
    #  3. import kagglehub  (ייבוא *בתוך* הפונקציה בכוונה — ככה מי שכבר יש לו
    #     את הקובץ מקומית יכול להשתמש במודול בלי שה-import ייכשל.)
    #  4. kagglehub.competition_download(COMPETITION) מחזיר נתיב לתיקייה,
    #     לא לקובץ. צריך לחפש בתוכה את train.csv.
    #  5. להעתיק את train.csv אל dest_dir (shutil.copy) ולהחזיר את הנתיב החדש.
    #  6. לעטוף 3-5 ב-try/except ולזרוק RuntimeError עם הודעה ברורה:
    #     - איפה לשים את kaggle.json
    #     - שצריך ללחוץ "Join Competition" בעמוד התחרות
    #     זה בדיוק ה-"error handling" שנבדק במטלה.
    raise NotImplementedError


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
    # TODO (מרים):
    #  1. להמיר ל-Path ולבדוק exists() -> אחרת FileNotFoundError עם הודעה
    #     שמזכירה להריץ את download_titanic().
    #  2. pd.read_csv
    #  3. לבדוק שכל EXPECTED_COLUMNS נמצאות. אם חסרות -> ValueError שמפרט
    #     *אילו* עמודות חסרות (הודעה שימושית, לא סתם "invalid file").
    #  4. להחזיר את ה-DataFrame.
    raise NotImplementedError


# --- Splitting ---------------------------------------------------------------

def split_train_val(
    df: pd.DataFrame,
    val_size: float = DEFAULT_VAL_SIZE,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataframe into train and validation sets.

    The split is **stratified** on the target so both sides keep the same
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
        ``(train_df, val_df)``.
    """
    # TODO (מרים):
    #  1. train_test_split על ה-DataFrame כולו (לא לפצל X ו-y בנפרד — נוח יותר
    #     להחזיר DataFrames שלמים, וה-preprocessing יקרה אחר כך).
    #  2. stratify=df[TARGET]   <-- אל תשכחי. בלי זה יחס השורדים בוולידציה
    #     יכול לצאת שונה מה-train והמדידה תהיה מוטה.
    #  3. random_state=seed, shuffle=True
    #  4. להחזיר (train_df, val_df).
    raise NotImplementedError


# --- Sample for the repo -----------------------------------------------------

def make_sample_csv(
    df: pd.DataFrame,
    n: int = 30,
    out_path: Path | None = None,
    seed: int = DEFAULT_SEED,
) -> Path:
    """Write a small sample CSV that is safe to commit to the repository.

    The assignment asks for a ``data/`` folder containing a small sample
    dataset. This sample lets a reviewer exercise the inference screen without
    Kaggle credentials.

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
    # TODO (מרים):
    #  1. אם out_path הוא None -> DATA_DIR / SAMPLE_FILENAME
    #  2. df.sample(n=..., random_state=seed)
    #     שקלי להשתמש ב-groupby(TARGET) כדי שהדגימה תכיל גם שורדים וגם לא —
    #     דגימה של 30 שורות אקראיות עלולה לצאת חד-צדדית ואז הגרפים במסך
    #     ה-inference ייראו מוזר.
    #  3. to_csv(out_path, index=False)  <-- index=False, אחרת נוצרת עמודה מיותרת
    #  4. להחזיר את out_path.
    raise NotImplementedError


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
