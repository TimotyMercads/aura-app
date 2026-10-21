"""
data_service.py
----------------
Everything related to loading, storing, and reading the student dataset.

PANEL REVISION (ODRF): an upload only needs the columns that actually feed
the prediction. The final model (XGBoost, AURA-Final-Model-Training
notebook) uses three inputs, so the expected file is:

    student_id      required - identifies the student (not a model input)
    program         required - BSCS / BSIT / BMMA
    year_level      required - 1, 2, 3 ...
    first_sem_gwa   required - 1st-semester GWA (AUF percentage scale)
    target_class    OPTIONAL - recorded class (At-Risk / Stable /
                    High-Performing), only shown next to the prediction

Any other column in the file (annual / 2nd-sem GWA, attendance, LMS, total
classes, demographics ...) is ignored and listed in the cleaning report.
Every upload is run through the selected model and its prediction
(predicted_class) is the status shown across the app.

If no dataset has been uploaded yet, every function degrades gracefully
and returns None / demo-friendly defaults so the UI never crashes.
"""

import os
import io
import json
import datetime as dt

import numpy as np
import pandas as pd
from sqlalchemy import text, inspect

from utils.cleaning import (
    clean_dataset, missing_required_columns, normalize_header,
    ID_COLUMN, TARGET_COLUMN, SCHEMA_COLUMNS, NUMERIC_COLUMNS, INTEGER_COLUMNS,
    DATE_COLUMNS, CLASS_ORDER,
)
from utils.db import get_engine, is_postgres, dialect_name

PREDICTION_COLUMNS = ["predicted_class", "prediction_confidence", "prob_at_risk"]
STATUS_COLUMN = "status"            # computed in memory: prediction first, recorded label second
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
PREFERRED_SHEET = "dataset"         # the AURA workbook's data sheet

# Columns of the `students` table (besides id / upload_id)
KNOWN_STUDENT_COLUMNS = SCHEMA_COLUMNS + PREDICTION_COLUMNS

AT_RISK, STABLE, HIGH = CLASS_ORDER
LABEL_COLORS = {AT_RISK: "#9C3A56", STABLE: "#3E6FA8", HIGH: "#57B27F"}


class UnsupportedFileType(Exception):
    """Raised when an uploaded file isn't a .csv/.xlsx/.xls."""


class MissingRequiredColumn(Exception):
    """Raised when an uploaded file lacks student_id or a model input."""


# --------------------------------------------------------------------------
# Students table DDL + one-time automatic migration from the old schema
# --------------------------------------------------------------------------

_SQL_TYPES = {
    "student_id": "VARCHAR(50) NOT NULL",
    "program": "VARCHAR(20) NULL",
    "year_level": "INT NULL",
    "first_sem_gwa": "FLOAT NULL",
    "target_class": "VARCHAR(20) NULL",
    "predicted_class": "VARCHAR(20) NULL",
    "prediction_confidence": "FLOAT NULL",
    "prob_at_risk": "FLOAT NULL",
}

# Columns of the previous (full-schema) students table that the panel-revised
# model no longer uses. If any of these exist, the table is migrated.
LEGACY_STUDENT_COLUMNS = [
    "academic_year", "semester", "gwa", "assignment_average", "second_sem_gwa",
    "total_classes", "classes_attended", "absences", "attendance_rate", "absenteeism_rate",
    "lms_login_count", "lms_resource_views", "lms_assignment_submissions",
    "on_time_submissions", "late_submissions", "lms_activity_count",
    "avg_weekly_lms_logins", "avg_daily_activity", "first_activity_date", "last_activity_date",
]


def _column_type(col):
    if col in _SQL_TYPES:
        return _SQL_TYPES[col]
    return "INT NULL" if col in INTEGER_COLUMNS else "FLOAT NULL"


def students_table_ddl(dialect):
    """CREATE TABLE statement for the current schema (mysql / postgresql / sqlite)."""
    if dialect == "mysql":
        pk = "id INT AUTO_INCREMENT PRIMARY KEY"
        tail = ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"
    elif dialect.startswith("postgres"):
        pk, tail = "id SERIAL PRIMARY KEY", ")"
    else:
        pk, tail = "id INTEGER PRIMARY KEY AUTOINCREMENT", ")"
    lines = [pk, "upload_id INT NULL REFERENCES uploads(id) ON DELETE SET NULL"]
    lines += [f"{c} {_column_type(c)}" for c in KNOWN_STUDENT_COLUMNS]
    return "CREATE TABLE students (\n    " + ",\n    ".join(lines) + "\n" + tail


_schema_checked = False


def _create_students_table(conn, dialect):
    conn.execute(text(students_table_ddl(dialect)))
    conn.execute(text("CREATE INDEX idx_student_id ON students (student_id)"))
    conn.execute(text("CREATE INDEX idx_students_upload ON students (upload_id)"))


def ensure_students_schema():
    """
    Makes sure `students` has the current (panel-revised) columns -
    automatically, once per process. Only the students table is touched;
    users, upload history, intervention checklists and action plans are kept.

      * no table                      -> create it
      * the very old synthetic schema -> drop + recreate (that data can't be
        (Student_ID, Previous_GPA ...)   used by the model; re-upload needed)
      * the previous full AURA schema -> MIGRATE IN PLACE: keep student_id,
        (attendance, LMS, total_classes,  program, year_level, first_sem_gwa,
        annual GWA ... columns)           target_class and upload_id; drop the
                                          rest. Predictions are cleared so the
                                          new model re-classifies everyone on
                                          the next page load.
    """
    global _schema_checked
    if _schema_checked:
        return
    engine = get_engine()
    insp = inspect(engine)
    dialect = dialect_name()

    if not insp.has_table("students"):
        with engine.begin() as conn:
            _create_students_table(conn, dialect)
        _schema_checked = True
        return

    cols = {c["name"].lower() for c in insp.get_columns("students")}
    core = {"student_id", "program", "year_level", "first_sem_gwa", "predicted_class"}

    if not core <= cols or "previous_gpa" in cols:
        # unusable old schema -> start over
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE students"))
            _create_students_table(conn, dialect)
            conn.execute(text("UPDATE uploads SET is_active = FALSE"))
    elif cols & set(LEGACY_STUDENT_COLUMNS):
        # previous full schema -> keep the rows, keep only the model's columns
        keep = ["upload_id"] + [c for c in SCHEMA_COLUMNS if c in cols]
        with engine.begin() as conn:
            old = pd.read_sql(text(f"SELECT {', '.join(keep)} FROM students ORDER BY id"), conn)
            conn.execute(text("DROP TABLE students"))
            _create_students_table(conn, dialect)
            if not old.empty:
                old = old.astype(object).where(old.notna(), None)
                old.to_sql("students", conn, if_exists="append", index=False)
    _schema_checked = True


# --------------------------------------------------------------------------
# Loading / saving - scoped to ONE teacher via students.upload_id ->
# uploads.uploaded_by, so a teacher never gets another teacher's students.
# --------------------------------------------------------------------------

def has_dataset(teacher_id) -> bool:
    ensure_students_schema()
    engine = get_engine()
    with engine.connect() as conn:
        count = conn.execute(
            text(
                "SELECT COUNT(*) FROM students s "
                "JOIN uploads u ON s.upload_id = u.id "
                "WHERE u.uploaded_by = :tid"
            ),
            {"tid": teacher_id},
        ).scalar()
    return bool(count)


def _coerce_types(df):
    for col in NUMERIC_COLUMNS + ["prediction_confidence", "prob_at_risk"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    for col in ("student_id", "program", "target_class", "predicted_class"):
        if col in df.columns:
            df[col] = df[col].astype(object).where(df[col].notna(), None)
    return df


def _with_status(df):
    """status = the model's prediction; falls back to the recorded label."""
    if df is None:
        return None
    pred = df["predicted_class"] if "predicted_class" in df.columns else pd.Series(None, index=df.index)
    rec = df[TARGET_COLUMN] if TARGET_COLUMN in df.columns else pd.Series(None, index=df.index)
    df[STATUS_COLUMN] = pred.where(pred.notna(), rec)
    return df


def load_dataset(teacher_id, predict_fn=None):
    """
    Returns a DataFrame of just this teacher's students, or None if empty.
    If predict_fn (model_service.predict_batch) is given and some rows have
    no stored prediction yet (e.g. uploaded before the model was added),
    they are predicted on the fly.
    """
    ensure_students_schema()
    engine = get_engine()
    try:
        df = pd.read_sql(
            text(
                "SELECT s.* FROM students s "
                "JOIN uploads u ON s.upload_id = u.id "
                "WHERE u.uploaded_by = :tid ORDER BY s.id"
            ),
            engine,
            params={"tid": teacher_id},
        )
    except Exception:
        return None
    if df.empty:
        return None
    df.columns = [c.lower() for c in df.columns]
    df = _coerce_types(df.drop(columns=["id", "upload_id"], errors="ignore"))
    if predict_fn is not None and df["predicted_class"].isna().any():
        missing = df["predicted_class"].isna()
        res = predict_fn(df[missing])
        df.loc[missing, "predicted_class"] = res["labels"]
        df.loc[missing, "prediction_confidence"] = res["confidences"]
        df.loc[missing, "prob_at_risk"] = res["prob_at_risk"]
        # save them, so they're computed once (e.g. right after a migration)
        _store_predictions(teacher_id, df.loc[missing, ID_COLUMN], res)
    return _with_status(df)


def _store_predictions(teacher_id, student_ids, res):
    rows = [
        {"label": l, "conf": c, "par": p, "sid": sid, "tid": teacher_id}
        for l, c, p, sid in zip(res["labels"], res["confidences"], res["prob_at_risk"], student_ids)
        if l is not None
    ]
    if not rows:
        return 0
    try:
        with get_engine().begin() as conn:
            conn.execute(
                text(
                    "UPDATE students SET predicted_class = :label, prediction_confidence = :conf, "
                    "prob_at_risk = :par WHERE student_id = :sid "
                    "AND upload_id IN (SELECT id FROM uploads WHERE uploaded_by = :tid)"
                ),
                rows,
            )
    except Exception:
        return 0
    return len(rows)


def _read_upload(file_storage, filename, ext):
    data = file_storage.read()
    if ext in (".xlsx", ".xls"):
        try:
            xls = pd.ExcelFile(io.BytesIO(data))
        except ImportError:
            raise UnsupportedFileType(
                f'"{filename}" is an old .xls file this server cannot read - save it as .xlsx and upload again.'
            )
        sheet = next((s for s in xls.sheet_names if s.strip().lower() == PREFERRED_SHEET), xls.sheet_names[0])
        return pd.read_excel(xls, sheet_name=sheet), sheet
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc), None
        except UnicodeDecodeError:
            continue
    raise UnsupportedFileType(f'"{filename}" could not be read as a CSV file.')


def _apply_predictions(df, predict_fn):
    if predict_fn is None or df.empty:
        for c in PREDICTION_COLUMNS:
            df[c] = None
        return df
    res = predict_fn(df)
    df["predicted_class"] = res["labels"]
    df["prediction_confidence"] = res["confidences"]
    df["prob_at_risk"] = res["prob_at_risk"]
    return df


def save_dataset(file_storage, uploaded_by, predict_fn=None):
    """
    Validates (.csv/.xlsx/.xls + required columns), cleans, predicts every
    student in one batch (if predict_fn is given), then replaces THIS
    TEACHER'S OWN rows in `students` and logs the upload.
    Returns (df, report).
    """
    ensure_students_schema()
    filename = file_storage.filename or "dataset"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileType(
            f'"{filename}" is not applicable — wrong file type. Please upload a .csv or .xlsx file.'
        )

    raw_df, sheet = _read_upload(file_storage, filename, ext)
    missing = missing_required_columns(raw_df.columns)
    if missing:
        raise MissingRequiredColumn(
            f'"{filename}" is missing required column(s): {", ".join(missing)}. '
            f"The file needs student_id, program, year_level and first_sem_gwa — the only inputs "
            f"the model uses. Download the template on this page to see the exact format."
        )

    df, report = clean_dataset(raw_df)
    report["sheet"] = sheet
    if df.empty:
        raise MissingRequiredColumn(f'"{filename}" has no usable student records after cleaning.')

    df = _apply_predictions(df, predict_fn)
    df_to_insert = df[[c for c in KNOWN_STUDENT_COLUMNS if c in df.columns]].copy()
    if predict_fn is not None:
        report["predicted_counts"] = {
            k: int(v) for k, v in pd.Series(df["predicted_class"]).value_counts().items()
        }

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE uploads SET is_active = FALSE WHERE uploaded_by = :tid"),
            {"tid": uploaded_by},
        )
        params = {
            "filename": filename,
            "file_type": ext.lstrip("."),
            "row_count": len(df_to_insert),
            "uploaded_by": uploaded_by,
            "clean_report": json.dumps(report, default=str),
        }
        sql = (
            "INSERT INTO uploads (filename, file_type, row_count, uploaded_by, is_active, clean_report) "
            "VALUES (:filename, :file_type, :row_count, :uploaded_by, TRUE, :clean_report)"
        )
        if is_postgres():
            upload_id = conn.execute(text(sql + " RETURNING id"), params).scalar()
        else:
            upload_id = conn.execute(text(sql), params).lastrowid

        conn.execute(
            text(
                "DELETE FROM students WHERE upload_id IN "
                "(SELECT id FROM uploads WHERE uploaded_by = :tid AND id != :new_upload_id)"
            ),
            {"tid": uploaded_by, "new_upload_id": upload_id},
        )
        df_to_insert["upload_id"] = upload_id
        df_to_insert.to_sql("students", conn, if_exists="append", index=False)

    return _with_status(df), report


def refresh_predictions(teacher_id, predict_fn, student_id=None):
    """
    Re-runs the model over this teacher's stored students (or one student)
    and saves the new predictions - used after a record is edited and after
    the model is reloaded. Returns the number of rows updated.
    """
    if predict_fn is None:
        return 0
    df = load_dataset(teacher_id)
    if df is None:
        return 0
    if student_id is not None:
        df = df[df[ID_COLUMN].astype(str) == str(student_id)]
        if df.empty:
            return 0
    res = predict_fn(df)
    return _store_predictions(teacher_id, df[ID_COLUMN], res)


def get_upload_history(teacher_id, limit=10):
    """This teacher's uploaded files, most recent first."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT u.filename, u.file_type, u.row_count, u.is_active, u.uploaded_at,
                       us.full_name AS uploaded_by_name
                FROM uploads u
                LEFT JOIN users us ON u.uploaded_by = us.id
                WHERE u.uploaded_by = :tid
                ORDER BY u.uploaded_at DESC, u.id DESC
                LIMIT :limit
                """
            ),
            {"tid": teacher_id, "limit": limit},
        ).mappings().all()
    out = []
    for r in rows:
        item = dict(r)
        if isinstance(item.get("uploaded_at"), str):
            try:
                item["uploaded_at"] = dt.datetime.fromisoformat(item["uploaded_at"])
            except ValueError:
                item["uploaded_at"] = None
        out.append(item)
    return out


def get_last_clean_report(teacher_id):
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT filename, clean_report, uploaded_at FROM uploads "
                "WHERE is_active = TRUE AND uploaded_by = :tid "
                "ORDER BY uploaded_at DESC, id DESC LIMIT 1"
            ),
            {"tid": teacher_id},
        ).mappings().first()
    if not row or not row["clean_report"]:
        return None
    report = row["clean_report"]
    if isinstance(report, str):
        report = json.loads(report)
    report = dict(report)
    report["filename"] = row["filename"]
    uploaded_at = row["uploaded_at"]
    if isinstance(uploaded_at, str):
        report["cleaned_at"] = uploaded_at[:16]
    else:
        report["cleaned_at"] = uploaded_at.strftime("%Y-%m-%d %H:%M") if uploaded_at else ""
    return report


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _num(v):
    try:
        v = float(v)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


def gwa_value(row):
    """The GWA shown in tables: the 1st-semester GWA (the model's main input)."""
    v = _num(row.get("first_sem_gwa")) if "first_sem_gwa" in row.index else None
    return round(v, 2) if v is not None else None


def gwa_series(df):
    if "first_sem_gwa" in df.columns:
        return pd.to_numeric(df["first_sem_gwa"], errors="coerce")
    return None


def gwa_band(value):
    """Where a 1st-semester GWA falls on the study's operational cut-offs."""
    v = _num(value)
    if v is None:
        return None
    if v <= 82:
        return "82 or below (the At-Risk range)"
    if v >= 90:
        return "90 or above (the High-Performing range)"
    return "between 82 and 90 (the Stable range)"


def status_badge(label):
    """Maps a class label (or None) to a (text, css_class) pair."""
    if label == HIGH:
        return (HIGH, "badge-success")
    if label == STABLE:
        return (STABLE, "badge-info")
    if label == AT_RISK:
        return (AT_RISK, "badge-danger")
    return ("Unclassified", "badge-warning")


# --------------------------------------------------------------------------
# Dashboard
# --------------------------------------------------------------------------

def compute_dashboard_stats(df):
    if df is None:
        return {"demo": True, "total": 281, "high_performing": 163, "stable": 106, "at_risk": 12,
                "source": "demo"}
    counts = df[STATUS_COLUMN].value_counts()
    has_pred = "predicted_class" in df.columns and df["predicted_class"].notna().any()
    return {
        "demo": False,
        "total": len(df),
        "high_performing": int(counts.get(HIGH, 0)),
        "stable": int(counts.get(STABLE, 0)),
        "at_risk": int(counts.get(AT_RISK, 0)),
        "has_labels": TARGET_COLUMN in df.columns and df[TARGET_COLUMN].notna().any(),
        "source": "prediction" if has_pred else ("recorded" if counts.sum() else "none"),
    }


def get_recent_students(df, n=5):
    if df is None:
        return None
    return list(reversed([_row_to_summary(row) for _, row in df.tail(n).iterrows()]))


def _row_to_summary(row):
    badge_text, badge_class = status_badge(row.get(STATUS_COLUMN))
    return {
        "id": row.get(ID_COLUMN, "N/A"),
        "program": row.get("program") or "N/A",
        "year_level": row.get("year_level", "N/A"),
        "gwa": gwa_value(row),
        "average": gwa_value(row),
        "confidence": _num(row.get("prediction_confidence")),
        "prob_at_risk": _num(row.get("prob_at_risk")),
        "status_text": badge_text,
        "status_class": badge_class,
    }


# --------------------------------------------------------------------------
# Students list / search
# --------------------------------------------------------------------------

def get_filter_options(df):
    if df is None:
        return {"programs": [], "years": [], "statuses": CLASS_ORDER}
    programs = sorted(df["program"].dropna().astype(str).unique().tolist()) if "program" in df.columns else []
    years = sorted(int(y) for y in pd.to_numeric(df["year_level"], errors="coerce").dropna().unique()) \
        if "year_level" in df.columns else []
    return {"programs": programs, "years": years, "statuses": CLASS_ORDER}


def get_students_table(df, search="", program="", year="", status="", min_avg=""):
    filtered = filter_dataframe(df, search=search, program=program, year=year, status=status, min_avg=min_avg)
    if filtered is None:
        return None
    return [_row_to_summary(row) for _, row in filtered.iterrows()]


def get_student_row(df, student_id):
    if df is None:
        return None
    matches = df[df[ID_COLUMN].astype(str) == str(student_id)]
    if matches.empty:
        return None
    return matches.iloc[0]


# Columns a teacher may hand-edit (student_id is the lookup key; the
# prediction columns are computed by the model, never hand-typed).
EDITABLE_COLUMNS = [c for c in SCHEMA_COLUMNS if c != ID_COLUMN]

FIELD_METADATA = {
    "program": {"label": "Program", "type": "select", "options": ["BSCS", "BSIT", "BMMA"]},
    "year_level": {"label": "Year Level", "type": "select", "options": ["1", "2", "3", "4"]},
    "first_sem_gwa": {"label": "1st-Semester GWA", "type": "number", "step": "0.01"},
    "target_class": {"label": "Recorded Class (optional, not used by the model)", "type": "select",
                     "options": CLASS_ORDER},
}


def get_edit_fields(row):
    """Ordered list of {name, label, type, value, options, step} for the edit form."""
    fields = []
    for col in EDITABLE_COLUMNS:
        meta = FIELD_METADATA.get(col, {"label": col.replace("_", " ").title(), "type": "text"})
        value = row.get(col, "")
        if value is None or (isinstance(value, float) and pd.isna(value)) or value is pd.NaT:
            value = ""
        elif col == "year_level":
            value = str(int(float(value)))
        elif isinstance(value, (dt.date, pd.Timestamp)):
            value = value.strftime("%Y-%m-%d")
        fields.append({
            "name": col,
            "label": meta["label"],
            "type": meta["type"],
            "options": meta.get("options"),
            "step": meta.get("step"),
            "value": value,
        })
    return fields


def _convert_update_value(col, value):
    if value is None or value == "":
        return None
    if col in INTEGER_COLUMNS:
        return int(round(float(value)))
    if col in NUMERIC_COLUMNS:
        return float(value)
    if col in DATE_COLUMNS:
        return dt.date.fromisoformat(value)
    if col == "program":
        return value.strip().upper()
    return value.strip()


def update_student(teacher_id, student_id, updates: dict):
    """
    Updates one student's fields, but ONLY if that student belongs to an
    upload owned by teacher_id (enforced in the WHERE clause). Returns True
    if a row was updated.
    """
    clean = {}
    for k, v in updates.items():
        if k not in EDITABLE_COLUMNS:
            continue
        try:
            clean[k] = _convert_update_value(k, v)
        except (TypeError, ValueError):
            continue
    if clean.get("target_class") not in (None, *CLASS_ORDER):
        clean.pop("target_class")
    if not clean:
        return False

    set_clause = ", ".join(f"{col} = :{col}" for col in clean)
    params = dict(clean)
    params["student_id"] = student_id
    params["tid"] = teacher_id
    engine = get_engine()
    with engine.begin() as conn:
        result = conn.execute(
            text(
                f"""
                UPDATE students SET {set_clause}
                WHERE student_id = :student_id
                AND upload_id IN (SELECT id FROM uploads WHERE uploaded_by = :tid)
                """
            ),
            params,
        )
        return result.rowcount > 0


def build_student_summary(row, df=None):
    """Everything the insights page needs about one student."""
    badge_text, badge_class = status_badge(row.get(STATUS_COLUMN))
    rec_text, rec_class = status_badge(row.get(TARGET_COLUMN))
    first = _num(row.get("first_sem_gwa"))
    prob = _num(row.get("prob_at_risk"))
    return {
        "id": row.get(ID_COLUMN, "N/A"),
        "program": row.get("program") or "N/A",
        "year_level": row.get("year_level", "N/A"),
        "gwa": gwa_value(row),
        "average": gwa_value(row),
        "first_sem_gwa": first,
        "gwa_band": gwa_band(first),
        "gwa_bar_class": None if first is None else ("green" if first >= 90 else ("yellow" if first > 82 else "red")),
        "prob_at_risk": prob,
        "status_text": badge_text,
        "status_class": badge_class,
        "recorded_text": rec_text if row.get(TARGET_COLUMN) else None,
        "recorded_class": rec_class,
        "raw": row,
    }


# --------------------------------------------------------------------------
# Analytics
# --------------------------------------------------------------------------

def _class_crosstab(df, by):
    tab = pd.crosstab(by, df[STATUS_COLUMN])
    out = {c: [int(tab.loc[i, c]) if c in tab.columns else 0 for i in tab.index] for c in CLASS_ORDER}
    return tab, out


def build_analytics(df):
    if df is None:
        return None

    gwa = gwa_series(df)
    avg_grade = round(float(gwa.mean()), 2) if gwa is not None and gwa.notna().any() else None
    prob = pd.to_numeric(df["prob_at_risk"], errors="coerce") if "prob_at_risk" in df.columns else None
    avg_prob_at_risk = round(float(prob.mean()), 1) if prob is not None and prob.notna().any() else None

    counts = df[STATUS_COLUMN].value_counts()
    performance_counts = {c: int(counts.get(c, 0)) for c in CLASS_ORDER if counts.get(c, 0)}
    total = len(df)
    ai_summary = []
    if performance_counts and total:
        ai_summary.append(
            f"{round(performance_counts.get(HIGH, 0) / total * 100, 1)}% of students are classified "
            f"High-Performing and {round(performance_counts.get(STABLE, 0) / total * 100, 1)}% Stable."
        )
        if performance_counts.get(AT_RISK):
            ai_summary.append(
                f"{performance_counts[AT_RISK]} student(s) are classified At-Risk and may need intervention."
            )
    if "predicted_class" in df.columns and TARGET_COLUMN in df.columns:
        both = df[df["predicted_class"].notna() & df[TARGET_COLUMN].notna()]
        if len(both):
            agree = (both["predicted_class"] == both[TARGET_COLUMN]).mean() * 100
            ai_summary.append(
                f"The AI prediction matches the recorded class for {agree:.1f}% of the {len(both)} labeled "
                f"students. If these students were part of the training data this is NOT a test score - "
                f"see the Model Card for the held-out and cross-validated results."
            )
    if gwa is not None and "program" in df.columns:
        by_prog = gwa.groupby(df["program"]).mean().dropna()
        if len(by_prog) > 1:
            ai_summary.append(
                f"Average 1st-semester GWA is highest in {by_prog.idxmax()} ({by_prog.max():.2f}) and lowest in "
                f"{by_prog.idxmin()} ({by_prog.min():.2f})."
            )
    if prob is not None and STATUS_COLUMN in df.columns:
        watch = int(((df[STATUS_COLUMN] == STABLE) & (prob >= EARLY_WARNING_PROB)).sum())
        if watch:
            ai_summary.append(
                f"{watch} Stable student(s) still have an At-Risk probability of {EARLY_WARNING_PROB:g}% or more "
                f"- worth a light check-in (see Recommendations)."
            )

    gwa_by_program = {}
    if gwa is not None and "program" in df.columns:
        grouped = gwa.groupby(df["program"]).mean().round(2)
        gwa_by_program = {str(k): float(v) for k, v in grouped.items() if pd.notna(v)}

    class_by_program = {}
    if "program" in df.columns:
        tab, class_by_program = _class_crosstab(df, df["program"])
        class_by_program["_programs"] = [str(p) for p in tab.index]

    class_by_year = {}
    if "year_level" in df.columns:
        years = pd.to_numeric(df["year_level"], errors="coerce")
        tab, class_by_year = _class_crosstab(df, years)
        class_by_year["_years"] = [f"Year {int(y)}" for y in tab.index]

    return {
        "avg_grade": avg_grade,
        "avg_prob_at_risk": avg_prob_at_risk,
        "total": total,
        "at_risk_count": performance_counts.get(AT_RISK, 0),
        "performance_counts": performance_counts,
        "performance_colors": [LABEL_COLORS[c] for c in performance_counts],
        "gwa_by_program": gwa_by_program,
        "class_by_program": class_by_program,
        "class_by_year": class_by_year,
        "ai_summary": ai_summary,
    }


# --------------------------------------------------------------------------
# Recommendations
# --------------------------------------------------------------------------

EARLY_WARNING_PROB = 20.0   # Stable students with >= 20% At-Risk probability


def build_risk_buckets(df):
    if df is None:
        return None
    at_risk = df[df[STATUS_COLUMN] == AT_RISK]
    stable = df[df[STATUS_COLUMN] == STABLE]
    high = df[df[STATUS_COLUMN] == HIGH]
    if "prob_at_risk" in df.columns:
        at_risk = at_risk.sort_values("prob_at_risk", ascending=False, na_position="last")
        stable = stable.sort_values("prob_at_risk", ascending=False, na_position="last")
        early = stable[pd.to_numeric(stable["prob_at_risk"], errors="coerce") >= EARLY_WARNING_PROB]
    else:
        early = stable.iloc[0:0]
    return {
        "high_risk": [_row_to_summary(r) for _, r in at_risk.head(15).iterrows()],
        "monitor": [_row_to_summary(r) for _, r in early.head(15).iterrows()],
        "early_warning_count": len(early),
        "early_warning_threshold": EARLY_WARNING_PROB,
        "low_risk_count": len(high),
        "counts": (len(at_risk), len(stable), len(high)),
    }


# --------------------------------------------------------------------------
# Downloads / exports
# --------------------------------------------------------------------------

def build_enriched_dataframe(df, model_predict_batch_fn=None):
    """
    Copy of df with the AI's predicted class, confidence and At-Risk
    probability (fresh, if a batch prediction function is given) plus the
    GWA band each student falls into.
    """
    if df is None:
        return None
    export_df = df.drop(columns=[STATUS_COLUMN], errors="ignore").copy()
    if "first_sem_gwa" in export_df.columns:
        export_df["first_sem_gwa_band"] = export_df["first_sem_gwa"].map(
            lambda v: (gwa_band(v) or "").split(" (")[0] or None)
    if model_predict_batch_fn is not None and not export_df.empty:
        res = model_predict_batch_fn(export_df)
        export_df["predicted_class"] = res["labels"]
        export_df["prediction_confidence"] = res["confidences"]
        export_df["prob_at_risk"] = res["prob_at_risk"]
    return export_df


def filter_dataframe(df, search="", program="", year="", status="", min_avg=""):
    """Same filters as the Students page, returning the DataFrame (for exports)."""
    if df is None:
        return None
    filtered = df.copy()
    if search:
        s = search.strip().lower()
        filtered = filtered[filtered[ID_COLUMN].astype(str).str.lower().str.contains(s, regex=False)]
    if program and "program" in filtered.columns:
        filtered = filtered[filtered["program"].astype(str) == program]
    if year and "year_level" in filtered.columns:
        filtered = filtered[pd.to_numeric(filtered["year_level"], errors="coerce").astype("Int64").astype(str) == str(year)]
    if status:
        filtered = filtered[filtered[STATUS_COLUMN] == status]
    if min_avg:
        try:
            threshold = float(min_avg)
            g = gwa_series(filtered)
            if g is not None:
                filtered = filtered[g >= threshold]
        except ValueError:
            pass
    return filtered
