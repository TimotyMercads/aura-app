"""
data_service.py
----------------
Everything related to loading, storing, and reading the student dataset.

The expected schema is the AURA AY 2025-2026 dataset (sheet "dataset" of
the Final-AURA_Dataset workbook; see its Data_Dictionary sheet):

    student_id, academic_year, semester, program, year_level, gwa,
    first_sem_gwa, assignment_average, second_sem_gwa, total_classes,
    classes_attended, absences, attendance_rate, absenteeism_rate,
    lms_login_count, lms_resource_views, lms_assignment_submissions,
    on_time_submissions, late_submissions, lms_activity_count,
    avg_weekly_lms_logins, avg_daily_activity, first_activity_date,
    last_activity_date, target_class (optional: At-Risk / Stable /
    High-Performing)

target_class is OPTIONAL - the system's purpose is to classify UNLABELED
students, so every upload is run through the selected model and its
prediction (predicted_class) is the status shown across the app. When a
file does carry target_class, it is shown alongside as the "recorded"
class (computed by the registrar-GWA rule: At-Risk <= 82,
High-Performing >= 90).

No demographic variables are stored (study scope).

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
    "academic_year": "VARCHAR(20) NULL",
    "semester": "VARCHAR(60) NULL",
    "program": "VARCHAR(20) NULL",
    "first_activity_date": "DATE NULL",
    "last_activity_date": "DATE NULL",
    "target_class": "VARCHAR(20) NULL",
    "predicted_class": "VARCHAR(20) NULL",
    "prediction_confidence": "FLOAT NULL",
    "prob_at_risk": "FLOAT NULL",
}


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


def ensure_students_schema():
    """
    The previous version of AURA stored a different (synthetic) schema in
    `students` (Student_ID, Previous_GPA, Sex, ...). That data cannot be used
    by the new model, so if the old table is detected it is dropped and
    recreated with the current schema - automatically, once. Only the
    students table is touched; users, upload history, intervention checklists
    and action plans are kept.
    """
    global _schema_checked
    if _schema_checked:
        return
    engine = get_engine()
    insp = inspect(engine)
    if not insp.has_table("students"):
        recreate = True
    else:
        cols = {c["name"].lower() for c in insp.get_columns("students")}
        recreate = not {"first_sem_gwa", "target_class", "predicted_class", "student_id"} <= cols \
            or "previous_gpa" in cols
    if recreate:
        dialect = dialect_name()
        with engine.begin() as conn:
            if insp.has_table("students"):
                conn.execute(text("DROP TABLE students"))
            conn.execute(text(students_table_ddl(dialect)))
            conn.execute(text("CREATE INDEX idx_student_id ON students (student_id)"))
            conn.execute(text("CREATE INDEX idx_students_upload ON students (upload_id)"))
            # the old rows are gone, so no upload should still claim to be active
            conn.execute(text("UPDATE uploads SET is_active = FALSE"))
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
    return _with_status(df)


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
            f"The model needs these to classify each student — see the expected columns on this page."
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
    rows = [
        {"label": l, "conf": c, "par": p, "sid": sid, "tid": teacher_id}
        for l, c, p, sid in zip(res["labels"], res["confidences"], res["prob_at_risk"], df[ID_COLUMN])
    ]
    engine = get_engine()
    with engine.begin() as conn:
        for r in rows:
            conn.execute(
                text(
                    "UPDATE students SET predicted_class = :label, prediction_confidence = :conf, "
                    "prob_at_risk = :par WHERE student_id = :sid "
                    "AND upload_id IN (SELECT id FROM uploads WHERE uploaded_by = :tid)"
                ),
                r,
            )
    return len(rows)


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
    """The GWA shown in tables: annual GWA if available, else 1st-semester GWA."""
    for col in ("gwa", "first_sem_gwa"):
        v = _num(row.get(col)) if col in row.index else None
        if v is not None:
            return round(v, 2)
    return None


def gwa_series(df):
    if "gwa" in df.columns:
        s = pd.to_numeric(df["gwa"], errors="coerce")
        if "first_sem_gwa" in df.columns:
            s = s.fillna(pd.to_numeric(df["first_sem_gwa"], errors="coerce"))
        return s
    if "first_sem_gwa" in df.columns:
        return pd.to_numeric(df["first_sem_gwa"], errors="coerce")
    return None


def attendance_pct(value):
    v = _num(value)
    if v is None:
        return None
    return round(v * 100, 1) if v <= 1.5 else round(v, 1)


def status_badge(label):
    """Maps a class label (or None) to a (text, css_class) pair."""
    if label == HIGH:
        return (HIGH, "badge-success")
    if label == STABLE:
        return (STABLE, "badge-info")
    if label == AT_RISK:
        return (AT_RISK, "badge-danger")
    return ("Unclassified", "badge-warning")


def lms_activity_level(value, df=None):
    """Low / Medium / High from lms_login_count using dataset tertiles."""
    v = _num(value)
    if v is None:
        return "Unknown"
    if df is not None and "lms_login_count" in df.columns and len(df) >= 3:
        low_cut, high_cut = pd.to_numeric(df["lms_login_count"], errors="coerce").quantile([0.33, 0.66])
    else:
        low_cut, high_cut = 110, 190
    if v <= low_cut:
        return "Low"
    if v <= high_cut:
        return "Medium"
    return "High"


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
    "academic_year": {"label": "Academic Year", "type": "text"},
    "semester": {"label": "Semester", "type": "text"},
    "program": {"label": "Program", "type": "select", "options": ["BSCS", "BSIT", "BMMA"]},
    "year_level": {"label": "Year Level", "type": "select", "options": ["1", "2", "3", "4"]},
    "gwa": {"label": "Annual GWA", "type": "number", "step": "0.01"},
    "first_sem_gwa": {"label": "1st-Semester GWA (Previous GWA)", "type": "number", "step": "0.01"},
    "assignment_average": {"label": "Assignment Average", "type": "number", "step": "0.01"},
    "second_sem_gwa": {"label": "2nd-Semester GWA", "type": "number", "step": "0.01"},
    "total_classes": {"label": "Total Classes", "type": "number", "step": "1"},
    "classes_attended": {"label": "Classes Attended", "type": "number", "step": "1"},
    "absences": {"label": "Absences", "type": "number", "step": "1"},
    "attendance_rate": {"label": "Attendance Rate (0–1)", "type": "number", "step": "0.0001"},
    "absenteeism_rate": {"label": "Absenteeism Rate (0–1)", "type": "number", "step": "0.0001"},
    "lms_login_count": {"label": "LMS Login Count", "type": "number", "step": "1"},
    "lms_resource_views": {"label": "LMS Resource Views", "type": "number", "step": "1"},
    "lms_assignment_submissions": {"label": "LMS Assignment Submissions", "type": "number", "step": "1"},
    "on_time_submissions": {"label": "On-Time Submissions", "type": "number", "step": "1"},
    "late_submissions": {"label": "Late Submissions", "type": "number", "step": "1"},
    "lms_activity_count": {"label": "LMS Activity Count", "type": "number", "step": "1"},
    "avg_weekly_lms_logins": {"label": "Avg. Weekly LMS Logins", "type": "number", "step": "0.01"},
    "avg_daily_activity": {"label": "Avg. Daily LMS Activity", "type": "number", "step": "0.01"},
    "first_activity_date": {"label": "First LMS Activity Date", "type": "date"},
    "last_activity_date": {"label": "Last LMS Activity Date", "type": "date"},
    "target_class": {"label": "Recorded Class (target_class)", "type": "select", "options": CLASS_ORDER},
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

    components = []
    for col, label_name in (("first_sem_gwa", "1st-Semester GWA"), ("second_sem_gwa", "2nd-Semester GWA"),
                            ("gwa", "Annual GWA"), ("assignment_average", "Assignment Average")):
        val = _num(row.get(col)) if col in row.index else None
        if val is not None:
            components.append({
                "label": label_name,
                "value": round(val, 2),
                "bar_class": "green" if val >= 90 else ("yellow" if val > 82 else "red"),
            })

    lms_logins = row.get("lms_login_count", None)
    return {
        "id": row.get(ID_COLUMN, "N/A"),
        "program": row.get("program") or "N/A",
        "year_level": row.get("year_level", "N/A"),
        "gwa": gwa_value(row),
        "average": gwa_value(row),
        "first_sem_gwa": _num(row.get("first_sem_gwa")),
        "second_sem_gwa": _num(row.get("second_sem_gwa")),
        "attendance": attendance_pct(row.get("attendance_rate")),
        "absences": _num(row.get("absences")),
        "total_classes": _num(row.get("total_classes")),
        "lms_logins": _num(lms_logins),
        "lms_activity": lms_activity_level(lms_logins, df),
        "status_text": badge_text,
        "status_class": badge_class,
        "recorded_text": rec_text if row.get(TARGET_COLUMN) else None,
        "recorded_class": rec_class,
        "components": components,
        "raw": row,
    }


# --------------------------------------------------------------------------
# Analytics
# --------------------------------------------------------------------------

def build_analytics(df):
    if df is None:
        return None

    gwa = gwa_series(df)
    avg_grade = round(float(gwa.mean()), 2) if gwa is not None and gwa.notna().any() else None
    avg_attendance = None
    if "attendance_rate" in df.columns:
        att = pd.to_numeric(df["attendance_rate"], errors="coerce").map(attendance_pct)
        avg_attendance = round(float(att.mean()), 1) if att.notna().any() else None

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
                f"students. The deployed model was refit on the full training data, so this is NOT a test "
                f"score — see the Model Card for the held-out results."
            )
    if gwa is not None and "program" in df.columns:
        by_prog = gwa.groupby(df["program"]).mean().dropna()
        if len(by_prog) > 1:
            ai_summary.append(
                f"Average GWA is highest in {by_prog.idxmax()} ({by_prog.max():.2f}) and lowest in "
                f"{by_prog.idxmin()} ({by_prog.min():.2f})."
            )
    if "first_sem_gwa" in df.columns and "second_sem_gwa" in df.columns:
        change = (pd.to_numeric(df["second_sem_gwa"], errors="coerce")
                  - pd.to_numeric(df["first_sem_gwa"], errors="coerce")).dropna()
        if len(change):
            ai_summary.append(
                f"{(change < 0).mean() * 100:.1f}% of students had a lower 2nd-semester GWA than their "
                f"1st-semester GWA — an observed pattern, not a proven cause."
            )

    attendance_by_year = {}
    if "year_level" in df.columns and "attendance_rate" in df.columns:
        att = pd.to_numeric(df["attendance_rate"], errors="coerce").map(attendance_pct)
        grouped = att.groupby(pd.to_numeric(df["year_level"], errors="coerce")).mean().round(1)
        attendance_by_year = {f"Year {int(k)}": float(v) for k, v in grouped.items() if pd.notna(v)}

    gwa_by_program = {}
    if gwa is not None and "program" in df.columns:
        grouped = gwa.groupby(df["program"]).mean().round(2)
        gwa_by_program = {str(k): float(v) for k, v in grouped.items() if pd.notna(v)}

    class_by_program = {}
    if "program" in df.columns:
        tab = pd.crosstab(df["program"], df[STATUS_COLUMN])
        for c in CLASS_ORDER:
            class_by_program[c] = [int(tab.loc[p, c]) if c in tab.columns else 0 for p in tab.index]
        class_by_program["_programs"] = [str(p) for p in tab.index]

    return {
        "avg_grade": avg_grade,
        "avg_attendance": avg_attendance,
        "at_risk_count": performance_counts.get(AT_RISK, 0),
        "performance_counts": performance_counts,
        "performance_colors": [LABEL_COLORS[c] for c in performance_counts],
        "attendance_by_year": attendance_by_year,
        "gwa_by_program": gwa_by_program,
        "class_by_program": class_by_program,
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
    Copy of df with computed columns: GWA, attendance %, LMS activity level,
    and - if a batch prediction function is given - fresh predictions.
    """
    if df is None:
        return None
    export_df = df.drop(columns=[STATUS_COLUMN], errors="ignore").copy()
    gwa = gwa_series(export_df)
    if gwa is not None:
        export_df["display_gwa"] = gwa.round(2)
    if "attendance_rate" in export_df.columns:
        export_df["attendance_pct"] = export_df["attendance_rate"].map(attendance_pct)
    if "lms_login_count" in export_df.columns:
        export_df["lms_activity_level"] = export_df["lms_login_count"].apply(
            lambda v: lms_activity_level(v, export_df)
        )
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
