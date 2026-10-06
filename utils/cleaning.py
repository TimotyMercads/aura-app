"""
cleaning.py
-----------
Runs automatically on every dataset upload (see data_service.save_dataset).
Implements the manuscript's Data Preparation / Cleaning steps (3.2.6) and
the record Exclusion Criteria (3.2.3.2) for the AURA AY 2025-2026 schema,
without silently destroying data - every change is counted and returned in
a report dict that the Upload page displays.

What it does, in order:
  1. Normalizes column headers to the dataset schema (trim, lower-case,
     spaces -> underscores) and maps accepted aliases, e.g.
     "first_sem_gwa(previous_gwa)" -> "first_sem_gwa" (same aliases as the
     training notebook).
  2. Removes demographic-type columns if any slipped in (the study's scope
     excludes demographics) and ignores columns that aren't in the schema.
  3. Trims text cells; treats empty strings as missing.
  4. Drops fully-empty rows, exact duplicates, and duplicate student_id
     rows (keeps the first) - one row per student.
  5. Coerces numeric/date columns (bad entries -> missing) and converts
     attendance/absenteeism rates given in percent (0-100) to proportions
     (0-1), the unit the dataset uses.
  6. Clips out-of-range values (e.g. a GWA of 104 -> 100).
  7. Normalizes program codes ("bsit" -> "BSIT") and target labels
     ("at risk" -> "At-Risk").
  8. EXCLUSION CRITERIA (manuscript 3.2.3.2):
       - records missing a critical model input (program, year level,
         1st-semester GWA, total classes) are excluded;
       - records with more than 20% of the required attributes missing are
         excluded.
  9. Imputes the remaining missing values in non-critical columns
     (median for numeric, most frequent for text). target_class is NEVER
     imputed - inventing a label would be fabricating an outcome - and
     dates are left empty.

Nothing here invents data that wasn't inferable from the column itself.
"""

import re

import numpy as np
import pandas as pd

ID_COLUMN = "student_id"
TARGET_COLUMN = "target_class"

CONTEXT_COLUMNS = ["academic_year", "semester"]
CATEGORICAL_COLUMNS = ["program"]
DATE_COLUMNS = ["first_activity_date", "last_activity_date"]

# (column, min, max, integer?) - None means "no bound on that side"
NUMERIC_BOUNDS = [
    ("year_level", 1, 6, True),
    ("gwa", 0, 100, False),
    ("first_sem_gwa", 0, 100, False),
    ("assignment_average", 0, 100, False),
    ("second_sem_gwa", 0, 100, False),
    ("total_classes", 0, None, True),
    ("classes_attended", 0, None, True),
    ("absences", 0, None, True),
    ("attendance_rate", 0, 1, False),
    ("absenteeism_rate", 0, 1, False),
    ("lms_login_count", 0, None, True),
    ("lms_resource_views", 0, None, True),
    ("lms_assignment_submissions", 0, None, True),
    ("on_time_submissions", 0, None, True),
    ("late_submissions", 0, None, True),
    ("lms_activity_count", 0, None, True),
    ("avg_weekly_lms_logins", 0, None, False),
    ("avg_daily_activity", 0, None, False),
]
NUMERIC_COLUMNS = [c for c, *_ in NUMERIC_BOUNDS]
INTEGER_COLUMNS = [c for c, lo, hi, is_int in NUMERIC_BOUNDS if is_int]
RATE_COLUMNS = ["attendance_rate", "absenteeism_rate"]

# Every column of the AURA dataset sheet, in the dataset's order.
SCHEMA_COLUMNS = (
    [ID_COLUMN] + CONTEXT_COLUMNS + ["program", "year_level",
     "gwa", "first_sem_gwa", "assignment_average", "second_sem_gwa",
     "total_classes", "classes_attended", "absences", "attendance_rate", "absenteeism_rate",
     "lms_login_count", "lms_resource_views", "lms_assignment_submissions",
     "on_time_submissions", "late_submissions", "lms_activity_count",
     "avg_weekly_lms_logins", "avg_daily_activity"] + DATE_COLUMNS + [TARGET_COLUMN]
)

# Inputs the selected model actually uses. A record missing one of these is
# excluded (manuscript: "records with missing values in critical variables
# ... shall be excluded regardless of the overall percentage").
CRITICAL_COLUMNS = ["program", "year_level", "first_sem_gwa", "total_classes"]
MAX_MISSING_SHARE = 0.20

# accepted spellings -> canonical names (same as the training notebook)
ALIASES = {
    "first_sem_gwa(previous_gwa)": "first_sem_gwa",
    "previous_gwa": "first_sem_gwa",
    "semester_gwa": "second_sem_gwa",
    "second_sem_gwa(semester_gwa)": "second_sem_gwa",
    "studentid": "student_id",
    "student_no": "student_id",
    "student_number": "student_id",
    "target": "target_class",
    "class": "target_class",
    "performance": "target_class",
    "annual_gwa": "gwa",
}

# demographic-type columns are removed automatically (study scope: none)
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


def normalize_header(name) -> str:
    key = str(name).strip().lower()
    key = re.sub(r"\s+", "_", key)
    return ALIASES.get(key, key)


def missing_required_columns(columns):
    """Columns a file must have for the model to predict meaningfully."""
    cols = {normalize_header(c) for c in columns}
    return [c for c in [ID_COLUMN] + CRITICAL_COLUMNS if c not in cols]


def clean_dataset(df: pd.DataFrame):
    """Returns (cleaned_df, report). Never raises on messy input."""
    report = {
        "original_rows": int(len(df)),
        "columns_renamed": {},
        "demographic_columns_removed": [],
        "unknown_columns_ignored": [],
        "empty_rows_removed": 0,
        "duplicate_rows_removed": 0,
        "duplicate_ids_removed": 0,
        "columns_trimmed": [],
        "numeric_values_coerced": {},
        "rates_converted_from_percent": [],
        "values_clipped": {},
        "categorical_values_normalized": [],
        "labels_unrecognized": 0,
        "excluded_missing_critical": 0,
        "excluded_too_many_missing": 0,
        "missing_before_impute": {},
        "values_imputed": {},
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

    # 2. Scope: drop demographics and anything not in the schema
    demo = [c for c in df.columns if c in DEMOGRAPHIC_NAMES]
    if demo:
        report["demographic_columns_removed"] = demo
    unknown = [c for c in df.columns if c not in SCHEMA_COLUMNS and c not in DEMOGRAPHIC_NAMES]
    if unknown:
        report["unknown_columns_ignored"] = unknown[:30]
    df = df[[c for c in SCHEMA_COLUMNS if c in df.columns]]

    # 3. Trim text cells, "" -> missing (only touches actual text values)
    for col in df.columns:
        is_text = df[col].map(lambda v: isinstance(v, str))
        if not is_text.any():
            continue
        texts = df.loc[is_text, col]
        cleaned = texts.str.strip()
        blank = cleaned.isin(["", "nan", "None", "NaN"])
        if (cleaned != texts).any() or blank.any():
            report["columns_trimmed"].append(col)
        df[col] = df[col].astype(object)
        df.loc[is_text, col] = cleaned.where(~blank, None)

    # 4. Empty rows / duplicates
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
        df[ID_COLUMN] = df[ID_COLUMN].astype(str).str.strip()
        id_dup_mask = df.duplicated(subset=[ID_COLUMN])
        report["duplicate_ids_removed"] = int(id_dup_mask.sum())
        df = df[~id_dup_mask]

    # 5. Coerce numbers / dates
    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue
        original_notna = df[col].notna()
        coerced = pd.to_numeric(df[col], errors="coerce")
        newly_invalid = int((original_notna & coerced.isna()).sum())
        if newly_invalid:
            report["numeric_values_coerced"][col] = newly_invalid
        df[col] = coerced
    for col in RATE_COLUMNS:
        if col in df.columns:
            pct = df[col] > 1.5                # value written as 0-100 -> 0-1 (per value,
            if pct.any():                      # so one percent entry can't rescale the rest)
                df.loc[pct, col] = df.loc[pct, col] / 100.0
                report["rates_converted_from_percent"].append(f"{col} ({int(pct.sum())})")
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

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
        df["program"] = df["program"].map(lambda v: str(v).strip().upper() if pd.notna(v) else v)
        if not before.astype(str).equals(df["program"].astype(str)):
            report["categorical_values_normalized"].append("program")
    if TARGET_COLUMN in df.columns:
        before = df[TARGET_COLUMN].copy()

        def _label(v):
            if pd.isna(v):
                return None
            return LABEL_MAP.get(str(v).strip().lower().replace("–", "-"), None)

        mapped = df[TARGET_COLUMN].map(_label)
        report["labels_unrecognized"] = int((before.notna() & mapped.isna()).sum())
        df[TARGET_COLUMN] = mapped
        if not before.astype(str).equals(df[TARGET_COLUMN].astype(str)):
            report["categorical_values_normalized"].append(TARGET_COLUMN)
        report["has_labels"] = bool(df[TARGET_COLUMN].notna().any())

    # 8. Exclusion criteria (manuscript 3.2.3.2)
    critical = [c for c in CRITICAL_COLUMNS if c in df.columns]
    if critical:
        crit_mask = df[critical].isna().any(axis=1)
        report["excluded_missing_critical"] = int(crit_mask.sum())
        df = df[~crit_mask]
    attr_cols = [c for c in df.columns if c not in (ID_COLUMN, TARGET_COLUMN)]
    if attr_cols:
        share = df[attr_cols].isna().mean(axis=1)
        too_many = share > MAX_MISSING_SHARE
        report["excluded_too_many_missing"] = int(too_many.sum())
        df = df[~too_many]

    # 9. Report + impute what is left (never the label, never dates)
    for col in df.columns:
        missing = int(df[col].isna().sum())
        if missing:
            report["missing_before_impute"][col] = missing
    for col in df.columns:
        if col in (ID_COLUMN, TARGET_COLUMN) or col in DATE_COLUMNS:
            continue
        missing = int(df[col].isna().sum())
        if not missing:
            continue
        if col in NUMERIC_COLUMNS:
            fill_value = df[col].median()
            if pd.isna(fill_value):
                continue
            if col in INTEGER_COLUMNS:
                fill_value = float(np.round(fill_value))
            df[col] = df[col].fillna(fill_value)
            report["values_imputed"][col] = {"count": missing, "method": "median", "value": round(float(fill_value), 2)}
        else:
            mode = df[col].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            df[col] = df[col].fillna(fill_value)
            report["values_imputed"][col] = {"count": missing, "method": "most frequent", "value": str(fill_value)}

    for col in INTEGER_COLUMNS:
        if col in df.columns:
            df[col] = df[col].round().astype("Int64")

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
        + report.get("excluded_too_many_missing", 0)
        + sum(report.get("numeric_values_coerced", {}).values())
        + sum(report.get("values_clipped", {}).values())
        + sum(v["count"] for v in report.get("values_imputed", {}).values())
    )
