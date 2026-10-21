"""
cleaning.py
-----------
Runs automatically on every dataset upload (see data_service.save_dataset).
Implements the manuscript's Data Preparation / Cleaning steps and the record
Exclusion Criteria (3.2.3.2) - every change is counted and returned in a
report dict that the Upload page displays.

PANEL REVISION (ODRF): the app now asks ONLY for the columns that actually
feed the prediction. The final model (XGBoost, AURA-Final-Model-Training
notebook) uses exactly three inputs:

    program         BSCS / BSIT / BMMA
    year_level      1, 2, 3 ...
    first_sem_gwa   1st-semester GWA (AUF percentage scale, 75 = passing)

plus two non-predictive columns:

    student_id      required - identifies the student (never used by the model)
    target_class    OPTIONAL - the recorded class (At-Risk / Stable /
                    High-Performing). Never used to predict; only shown next
                    to the AI's prediction so the two can be compared.

Anything else in the file (annual / 2nd-sem GWA, attendance, LMS, total
classes, demographics ...) is dropped and listed in the report as
"not used by the model".

What it does, in order:
  1. Normalizes headers (trim, lower-case, spaces/dashes -> underscores) and
     maps accepted aliases ("first_sem_gwa(previous_gwa)" -> "first_sem_gwa").
  2. Keeps only the columns above; reports demographic columns and any other
     columns that were ignored.
  3. Trims text cells; treats empty strings / "n/a" / "-" as missing.
  4. Drops fully-empty rows, exact duplicates, rows without a student_id and
     duplicate student_id rows (keeps the first) - one row per student.
  5. Coerces numbers (bad entries -> missing): "88.5 " -> 88.5, "2nd" /
     "Year 2" -> 2 for year level.
  6. Clips out-of-range values (GWA above 100 -> 100, year level 0 -> 1).
  7. Normalizes program codes ("b.s.i.t." -> "BSIT") and target labels
     ("at risk" -> "At-Risk"); flags programs the model was not trained on.
  8. EXCLUSION CRITERIA (manuscript 3.2.3.2): every model input is a
     CRITICAL variable, so a record missing program, year level or
     1st-semester GWA is excluded - the app never invents a grade.

target_class is never imputed - inventing a label would fabricate an outcome.
"""

import re

import numpy as np
import pandas as pd

ID_COLUMN = "student_id"
TARGET_COLUMN = "target_class"

# The three inputs of the final model (order = how they are shown in the UI)
MODEL_INPUT_COLUMNS = ["program", "year_level", "first_sem_gwa"]
CATEGORICAL_COLUMNS = ["program"]

# (column, min, max, integer?)
NUMERIC_BOUNDS = [
    ("year_level", 1, 6, True),
    ("first_sem_gwa", 0, 100, False),
]
NUMERIC_COLUMNS = [c for c, *_ in NUMERIC_BOUNDS]
INTEGER_COLUMNS = [c for c, lo, hi, is_int in NUMERIC_BOUNDS if is_int]
DATE_COLUMNS = []          # no date columns are stored anymore

# Every column the app stores, in display order.
SCHEMA_COLUMNS = [ID_COLUMN] + MODEL_INPUT_COLUMNS + [TARGET_COLUMN]
REQUIRED_COLUMNS = [ID_COLUMN] + MODEL_INPUT_COLUMNS
OPTIONAL_COLUMNS = [TARGET_COLUMN]

# A record missing any model input is excluded (manuscript: "records with
# missing values in critical variables ... shall be excluded regardless of
# the overall percentage").
CRITICAL_COLUMNS = list(MODEL_INPUT_COLUMNS)

# Programs present in the training data (CCS, AY 2025-2026)
KNOWN_PROGRAMS = ["BSCS", "BSIT", "BMMA"]

# accepted spellings -> canonical names
ALIASES = {
    "first_sem_gwa(previous_gwa)": "first_sem_gwa",
    "previous_gwa": "first_sem_gwa",
    "1st_sem_gwa": "first_sem_gwa",
    "1st_semester_gwa": "first_sem_gwa",
    "first_semester_gwa": "first_sem_gwa",
    "gwa_1st_sem": "first_sem_gwa",
    "gwa_first_sem": "first_sem_gwa",
    "studentid": "student_id",
    "student_no": "student_id",
    "student_number": "student_id",
    "id_number": "student_id",
    "year": "year_level",
    "yr_level": "year_level",
    "yearlevel": "year_level",
    "course": "program",
    "programme": "program",
    "target": "target_class",
    "class": "target_class",
    "performance": "target_class",
    "recorded_class": "target_class",
}

# demographic-type columns are reported separately (study scope: none)
DEMOGRAPHIC_NAMES = {
    "age", "sex", "gender", "family_income", "income", "household_income",
    "socioeconomic_status", "ses", "residence", "address", "location", "province",
    "city", "barangay", "region", "ethnicity", "religion", "civil_status", "nationality",
}

LABEL_MAP = {
    "at-risk": "At-Risk", "at risk": "At-Risk", "atrisk": "At-Risk", "at_risk": "At-Risk",
    "stable": "Stable",
    "high-performing": "High-Performing", "high performing": "High-Performing",
    "highperforming": "High-Performing", "high_performing": "High-Performing",
}
CLASS_ORDER = ["At-Risk", "Stable", "High-Performing"]

_MISSING_TEXT = {"", "nan", "none", "null", "n/a", "na", "-", "--", "?", "tbd", "not available"}


def normalize_header(name) -> str:
    key = str(name).strip().lower()
    key = re.sub(r"[\s\-]+", "_", key)
    return ALIASES.get(key, key)


def missing_required_columns(columns):
    """Columns a file must have for the model to predict meaningfully."""
    cols = {normalize_header(c) for c in columns}
    return [c for c in REQUIRED_COLUMNS if c not in cols]


def normalize_program(value):
    """'b.s.i.t.' / ' bsit ' / 'BS IT' -> 'BSIT'. Unknown codes are kept upper-cased."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    code = re.sub(r"[^A-Za-z]", "", str(value)).upper()
    return code or None


def _to_number(value):
    """88.5 / '88.5 ' / '1,234' / '88.50%' -> float; anything else -> NaN."""
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip().replace(",", "").rstrip("%").strip()
    try:
        return float(text)
    except ValueError:
        return np.nan


def _to_year(value):
    """2 / '2' / '2nd' / 'Year 2' / '2nd yr' -> 2.0; 'first' .. 'fourth' too."""
    n = _to_number(value)
    if not np.isnan(n):
        return n
    text = str(value).strip().lower()
    words = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
    for w, k in words.items():
        if w in text:
            return float(k)
    m = re.search(r"\d+", text)
    return float(m.group()) if m else np.nan


def clean_dataset(df: pd.DataFrame):
    """Returns (cleaned_df, report). Never raises on messy input."""
    report = {
        "original_rows": int(len(df)),
        "columns_renamed": {},
        "demographic_columns_removed": [],
        "unused_columns_ignored": [],
        "empty_rows_removed": 0,
        "duplicate_rows_removed": 0,
        "duplicate_ids_removed": 0,
        "columns_trimmed": [],
        "numeric_values_coerced": {},
        "values_clipped": {},
        "categorical_values_normalized": [],
        "unknown_programs": {},
        "labels_unrecognized": 0,
        "excluded_missing_critical": 0,
        "excluded_by_column": {},
        "has_labels": False,
        "final_rows": 0,
    }

    df = df.copy()

    # 1. Headers -> schema names
    new_cols = []
    for c in df.columns:
        norm = normalize_header(c)
        if norm != str(c):
            report["columns_renamed"][str(c)] = norm
        new_cols.append(norm)
    df.columns = new_cols
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    # don't list renames of columns that are thrown away anyway
    report["columns_renamed"] = {k: v for k, v in report["columns_renamed"].items() if v in SCHEMA_COLUMNS}

    # 2. Keep only what the model (and the student list) needs
    demo = [c for c in df.columns if c in DEMOGRAPHIC_NAMES]
    report["demographic_columns_removed"] = demo
    unused = [c for c in df.columns if c not in SCHEMA_COLUMNS and c not in DEMOGRAPHIC_NAMES
              and not c.startswith("unnamed")]
    report["unused_columns_ignored"] = unused[:40]
    df = df[[c for c in SCHEMA_COLUMNS if c in df.columns]]

    # 3. Trim text cells, "" / "n/a" / "-" -> missing (only touches text values)
    for col in df.columns:
        is_text = df[col].map(lambda v: isinstance(v, str))
        if not is_text.any():
            continue
        texts = df.loc[is_text, col]
        cleaned = texts.str.strip()
        blank = cleaned.str.lower().isin(_MISSING_TEXT)
        if (cleaned != texts).any() or blank.any():
            report["columns_trimmed"].append(col)
        df[col] = df[col].astype(object)
        df.loc[is_text, col] = cleaned.where(~blank, None)

    # 4. Empty rows / duplicates / one row per student
    empty_mask = df.isna().all(axis=1)
    report["empty_rows_removed"] = int(empty_mask.sum())
    df = df[~empty_mask]

    dup_mask = df.duplicated()
    report["duplicate_rows_removed"] = int(dup_mask.sum())
    df = df[~dup_mask]

    if ID_COLUMN in df.columns:
        no_id = df[ID_COLUMN].isna()
        df = df[~no_id]
        report["empty_rows_removed"] += int(no_id.sum())
        df[ID_COLUMN] = df[ID_COLUMN].map(
            lambda v: str(int(v)) if isinstance(v, float) and float(v).is_integer() else str(v).strip())
        id_dup_mask = df.duplicated(subset=[ID_COLUMN])
        report["duplicate_ids_removed"] = int(id_dup_mask.sum())
        df = df[~id_dup_mask]

    # 5. Coerce numbers ("2nd" / "Year 3" / "3rd yr" are read as year levels)
    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue
        original_notna = df[col].notna()
        coerced = df[col].map(_to_year if col == "year_level" else _to_number).astype(float)
        newly_invalid = int((original_notna & coerced.isna()).sum())
        if newly_invalid:
            report["numeric_values_coerced"][col] = newly_invalid
        df[col] = coerced

    # 6. Clip out-of-range values
    for col, lo, hi, _ in NUMERIC_BOUNDS:
        if col not in df.columns:
            continue
        mask = pd.Series(False, index=df.index)
        if lo is not None:
            mask |= df[col] < lo
        if hi is not None:
            mask |= df[col] > hi
        clipped = int(mask.sum())
        if clipped:
            report["values_clipped"][col] = clipped
            df[col] = df[col].clip(lower=lo, upper=hi)

    # 7. Program codes and target labels
    if "program" in df.columns:
        before = df["program"].copy()
        df["program"] = df["program"].map(normalize_program)
        if not before.astype(str).equals(df["program"].astype(str)):
            report["categorical_values_normalized"].append("program")
        unknown = df["program"].dropna()
        unknown = unknown[~unknown.isin(KNOWN_PROGRAMS)]
        report["unknown_programs"] = {str(k): int(v) for k, v in unknown.value_counts().items()}
    if TARGET_COLUMN in df.columns:
        before = df[TARGET_COLUMN].copy()

        def _label(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return None
            return LABEL_MAP.get(str(v).strip().lower().replace("–", "-"), None)

        mapped = df[TARGET_COLUMN].map(_label)
        report["labels_unrecognized"] = int((before.notna() & mapped.isna()).sum())
        df[TARGET_COLUMN] = mapped
        if not before.astype(str).equals(df[TARGET_COLUMN].astype(str)):
            report["categorical_values_normalized"].append(TARGET_COLUMN)
        report["has_labels"] = bool(df[TARGET_COLUMN].notna().any())

    # 8. Exclusion criteria (manuscript 3.2.3.2) - all model inputs are critical
    critical = [c for c in CRITICAL_COLUMNS if c in df.columns]
    if critical:
        crit_mask = df[critical].isna().any(axis=1)
        report["excluded_missing_critical"] = int(crit_mask.sum())
        report["excluded_by_column"] = {c: int(df[c].isna().sum()) for c in critical if df[c].isna().any()}
        df = df[~crit_mask]

    for col in INTEGER_COLUMNS:
        if col in df.columns:
            df[col] = df[col].round().astype("Int64")
    if "first_sem_gwa" in df.columns:
        df["first_sem_gwa"] = df["first_sem_gwa"].round(2)

    df = df.reset_index(drop=True)
    report["final_rows"] = int(len(df))
    return df, report


def total_changes(report) -> int:
    """Single number for the upload flash message."""
    return (
        report.get("empty_rows_removed", 0)
        + report.get("duplicate_rows_removed", 0)
        + report.get("duplicate_ids_removed", 0)
        + report.get("excluded_missing_critical", 0)
        + sum(report.get("numeric_values_coerced", {}).values())
        + sum(report.get("values_clipped", {}).values())
        + report.get("labels_unrecognized", 0)
    )
