"""
AURA - Flask web app
---------------------
Academic at-risk / high-performing student prediction dashboard.

Run locally:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000

Requires MySQL or Postgres - see XAMPP_SETUP.md for local setup via
XAMPP, or DEPLOYMENT.md for cloud hosting (e.g. Vercel + a cloud
database). Import schema.sql (MySQL) or schema_postgres.sql (Postgres)
once before first run.

Multi-tenancy: every teacher only ever sees/edits the students from
datasets THEY uploaded - see data_service.py for how this is enforced
at the query level, not just in these routes.

Predictions come from ONE file produced by the Dataset_Training notebook:
    models/AURA_model_bundle.pkl   (selected model: KMeans + Random Forest)
plus aura_transformers.py next to this file (the notebook's custom pipeline
classes, needed to unpickle the bundle). Explanations use SHAP's
TreeExplainer (manuscript 3.3.2 Step 5).
"""

import os
import io
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user, login_required, current_user,
)
from sqlalchemy.exc import OperationalError

from services import data_service, model_service, auth_service, intervention_service, action_plan_service
from utils import intervention_policy, plain_language
from utils.cleaning import total_changes, CRITICAL_COLUMNS, SCHEMA_COLUMNS

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB upload limit


# --------------------------------------------------------------------------
# Auth setup
# --------------------------------------------------------------------------

class User(UserMixin):
    def __init__(self, row):
        self.id = str(row["id"])
        self.full_name = row["full_name"]
        self.email = row["email"]
        self.role = row.get("role", "teacher")


login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.login_message = "Please log in to continue."
login_manager.login_message_category = "error"
login_manager.init_app(app)


@login_manager.user_loader
def load_user(user_id):
    try:
        row = auth_service.get_user_by_id(int(user_id))
    except auth_service.DatabaseUnavailable:
        return None
    return User(row) if row else None


# --------------------------------------------------------------------------
# Friendly handling when the database isn't running
# --------------------------------------------------------------------------

@app.errorhandler(OperationalError)
@app.errorhandler(auth_service.DatabaseUnavailable)
def handle_db_unavailable(exc):
    return render_template(
        "db_error.html",
        active_page=None, has_dataset=False, model_ready=False, show_status_banner=False,
    ), 503


def _predict_fn():
    return model_service.predict_batch if model_service.is_ready() else None


def _load(teacher_id):
    """This teacher's students, with live predictions filled in where missing."""
    return data_service.load_dataset(teacher_id, predict_fn=_predict_fn())


def _common_context(active_page):
    teacher_id = int(current_user.id) if current_user.is_authenticated else None
    has_dataset = data_service.has_dataset(teacher_id) if teacher_id else False
    model_ready = model_service.is_ready()
    alert_count = action_plan_service.upcoming_alert_count(teacher_id) if teacher_id else 0
    info = model_service.model_info() if model_ready else None
    return {
        "active_page": active_page,
        "has_dataset": has_dataset,
        "model_ready": model_ready,
        "model_name": info["selected_model"] if info else None,
        "show_status_banner": (not has_dataset) or (not model_ready),
        "action_plan_alerts": alert_count,
    }


# --------------------------------------------------------------------------
# Auth routes
# --------------------------------------------------------------------------

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not full_name or not email or not password:
            flash("Please fill in all fields.", "error")
            return redirect(url_for("signup"))
        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return redirect(url_for("signup"))
        if len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
            return redirect(url_for("signup"))

        try:
            auth_service.create_user(full_name, email, password)
        except auth_service.EmailAlreadyExists:
            flash("An account with that email already exists. Try logging in instead.", "error")
            return redirect(url_for("signup"))

        flash("Account created — please log in.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        user_row = auth_service.get_user_by_email(email)
        if not user_row or not auth_service.verify_password(user_row, password):
            flash("Invalid email or password.", "error")
            return redirect(url_for("login"))

        login_user(User(user_row))
        flash(f"Welcome back, {user_row['full_name']}!", "success")
        next_page = request.args.get("next")
        return redirect(next_page or url_for("dashboard"))

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))


# --------------------------------------------------------------------------
# Core pages
# --------------------------------------------------------------------------

@app.route("/")
@login_required
def index():
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
@login_required
def dashboard():
    teacher_id = int(current_user.id)
    df = _load(teacher_id)
    stats = data_service.compute_dashboard_stats(df)
    students = data_service.get_recent_students(df, n=5)
    return render_template(
        "dashboard.html",
        stats=stats,
        students=students,
        **_common_context("dashboard"),
    )


@app.route("/students")
@login_required
def students():
    teacher_id = int(current_user.id)
    df = _load(teacher_id)
    query = request.args.get("q", "").strip()
    program = request.args.get("program", "").strip()
    year = request.args.get("year", "").strip()
    status = request.args.get("status", "").strip()
    min_avg = request.args.get("min_avg", "").strip()

    students_list = data_service.get_students_table(
        df, search=query, program=program, year=year, status=status, min_avg=min_avg
    )
    filter_options = data_service.get_filter_options(df)

    return render_template(
        "students.html",
        students=students_list,
        query=query,
        selected_program=program,
        selected_year=year,
        selected_status=status,
        min_avg=min_avg,
        filter_options=filter_options,
        **_common_context("students"),
    )


@app.route("/students/<student_id>")
@login_required
def student_insight(student_id):
    teacher_id = int(current_user.id)
    df = _load(teacher_id)
    row = data_service.get_student_row(df, student_id) if df is not None else None

    if row is None:
        flash(f'Student "{student_id}" was not found in your uploaded data.', "error")
        return redirect(url_for("students"))

    student = data_service.build_student_summary(row, df=df)
    prediction = model_service.predict_student(row) if model_service.is_ready() else None
    explanation = model_service.explain_student(row) if model_service.is_ready() else None

    # The AI prediction is the student's status (the system classifies
    # unlabeled students); the recorded class is shown alongside if present.
    display_label = prediction["label"] if prediction and not prediction.get("error") else row.get("status")

    stage_num = intervention_policy.determine_stage(row, display_label)
    stage_info = intervention_policy.STAGES.get(stage_num) if stage_num else None
    factor_actions = intervention_policy.explanation_recommendations(display_label, explanation)
    general_suggestions = intervention_policy.SUGGESTIONS_BY_LABEL.get(display_label, [])

    return render_template(
        "insights.html",
        student=student,
        prediction=prediction,
        explanation=explanation,
        display_label=display_label,
        stage_info=stage_info,
        factor_actions=factor_actions,
        general_suggestions=general_suggestions,
        **_common_context("students"),
    )


@app.route("/students/<student_id>/edit", methods=["GET", "POST"])
@login_required
def student_edit(student_id):
    teacher_id = int(current_user.id)
    df = _load(teacher_id)
    row = data_service.get_student_row(df, student_id) if df is not None else None

    if row is None:
        # Either the student doesn't exist, or (just as important) it
        # belongs to a dataset THIS teacher didn't upload - either way,
        # they don't get to see or edit it.
        flash(f'Student "{student_id}" was not found in your uploaded data.', "error")
        return redirect(url_for("students"))

    if request.method == "POST":
        updates = {}
        for col in data_service.EDITABLE_COLUMNS:
            if col in request.form:
                val = request.form.get(col, "").strip()
                updates[col] = val if val != "" else None
        ok = data_service.update_student(teacher_id, student_id, updates)
        if ok:
            # re-classify with the edited values so every page stays consistent
            data_service.refresh_predictions(teacher_id, _predict_fn(), student_id=student_id)
            flash(f"Student {student_id} updated and re-classified.", "success")
        else:
            flash("Update failed — the student may not belong to your uploaded data.", "error")
        return redirect(url_for("student_insight", student_id=student_id))

    student = data_service.build_student_summary(row, df=df)
    edit_fields = data_service.get_edit_fields(row)
    return render_template(
        "student_edit.html",
        student=student,
        edit_fields=edit_fields,
        **_common_context("students"),
    )


@app.route("/students/<student_id>/explanation")
@login_required
def student_explanation(student_id):
    teacher_id = int(current_user.id)
    df = _load(teacher_id)
    row = data_service.get_student_row(df, student_id) if df is not None else None

    if row is None:
        flash(f'Student "{student_id}" was not found in your uploaded data.', "error")
        return redirect(url_for("students"))

    student = data_service.build_student_summary(row, df=df)
    prediction = model_service.predict_student(row) if model_service.is_ready() else None
    explanation = model_service.explain_student(row) if model_service.is_ready() else None

    narrative = None
    if explanation and explanation.get("top_features"):
        narrative = plain_language.narrative_paragraph(
            student["id"], explanation["predicted_label"],
            prediction.get("confidence") if prediction else None,
            prediction.get("probabilities") if prediction else None,
            explanation["top_features"],
            include_intro=False,        # the page banner above already states the class
        )

    return render_template(
        "explanation.html",
        student=student,
        prediction=prediction,
        explanation=explanation,
        narrative=narrative,
        **_common_context("students"),
    )


@app.route("/students/<student_id>/recommendations")
@login_required
def student_recommendations(student_id):
    teacher_id = int(current_user.id)
    df = _load(teacher_id)
    row = data_service.get_student_row(df, student_id) if df is not None else None

    if row is None:
        flash(f'Student "{student_id}" was not found in your uploaded data.', "error")
        return redirect(url_for("students"))

    student = data_service.build_student_summary(row, df=df)
    prediction = model_service.predict_student(row) if model_service.is_ready() else None
    explanation = model_service.explain_student(row) if model_service.is_ready() else None
    display_label = prediction["label"] if prediction and not prediction.get("error") else row.get("status")

    stage_num = intervention_policy.determine_stage(row, display_label)
    completed_actions = intervention_service.get_actions_for_student(student_id)

    return render_template(
        "interventions.html",
        student=student,
        display_label=display_label,
        current_stage=stage_num,
        stages=[intervention_policy.STAGE_1, intervention_policy.STAGE_2, intervention_policy.STAGE_3],
        completed_actions=completed_actions,
        factor_actions=intervention_policy.explanation_recommendations(display_label, explanation, limit=5),
        label_suggestions=intervention_policy.SUGGESTIONS_BY_LABEL.get(display_label, []),
        at_risk_suggestions=intervention_policy.AT_RISK_SUGGESTIONS,
        high_performing_suggestions=intervention_policy.HIGH_PERFORMING_SUGGESTIONS,
        policy_reminders=intervention_policy.POLICY_REMINDERS,
        **_common_context("students"),
    )


@app.route("/students/<student_id>/intervention/toggle", methods=["POST"])
@login_required
def toggle_intervention_action(student_id):
    teacher_id = int(current_user.id)
    # Ownership check: only allow toggling checklist items for a student
    # that actually belongs to this teacher's uploaded data.
    df = _load(teacher_id)
    if data_service.get_student_row(df, student_id) is None:
        flash(f'Student "{student_id}" was not found in your uploaded data.', "error")
        return redirect(url_for("students"))

    stage = request.form.get("stage", type=int)
    action_key = request.form.get("action_key", "")
    if stage and action_key:
        intervention_service.toggle_action(student_id, stage, action_key, teacher_id)
    return redirect(url_for("student_recommendations", student_id=student_id))


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    teacher_id = int(current_user.id)
    if request.method == "POST":
        file = request.files.get("dataset")
        if not file or file.filename == "":
            flash("Please choose a file before uploading.", "error")
            return redirect(url_for("upload"))
        try:
            df, report = data_service.save_dataset(file, uploaded_by=teacher_id, predict_fn=_predict_fn())
            changes = total_changes(report)
            msg = f"Uploaded {file.filename} — {len(df):,} student records loaded"
            if report.get("predicted_counts"):
                pc = report["predicted_counts"]
                msg += (f" and classified ({pc.get('At-Risk', 0)} At-Risk, {pc.get('Stable', 0)} Stable, "
                        f"{pc.get('High-Performing', 0)} High-Performing)")
            msg += "."
            if changes:
                msg += f" Cleaning made {changes:,} change(s); see the report below."
            flash(msg, "success")
        except (data_service.UnsupportedFileType, data_service.MissingRequiredColumn) as exc:
            flash(str(exc), "error")
        except Exception as exc:
            flash(f"Upload failed: {exc}", "error")
        return redirect(url_for("upload"))

    history = data_service.get_upload_history(teacher_id)
    clean_report = data_service.get_last_clean_report(teacher_id)
    return render_template(
        "upload.html",
        history=history,
        model_files_present=model_service.file_status(),
        model_info=model_service.model_info(),
        model_error=model_service.load_error(),
        clean_report=clean_report,
        required_columns=["student_id"] + CRITICAL_COLUMNS,
        schema_columns=SCHEMA_COLUMNS,
        **_common_context("upload"),
    )


@app.route("/download/dataset")
@login_required
def download_dataset():
    teacher_id = int(current_user.id)
    df = _load(teacher_id)
    if df is None:
        flash("No dataset to download yet.", "error")
        return redirect(url_for("upload"))
    buffer = io.BytesIO()
    df.drop(columns=["status"], errors="ignore").to_csv(buffer, index=False)
    buffer.seek(0)
    return send_file(
        buffer, mimetype="text/csv", as_attachment=True,
        download_name="aura_dataset.csv",
    )


@app.route("/download/dataset/enriched")
@login_required
def download_dataset_enriched():
    teacher_id = int(current_user.id)
    df = _load(teacher_id)
    if df is None:
        flash("No dataset to download yet.", "error")
        return redirect(url_for("upload"))
    export_df = data_service.build_enriched_dataframe(df, model_predict_batch_fn=_predict_fn())
    buffer = io.BytesIO()
    export_df.to_csv(buffer, index=False)
    buffer.seek(0)
    return send_file(
        buffer, mimetype="text/csv", as_attachment=True,
        download_name="aura_dataset_enriched.csv",
    )


@app.route("/download/students")
@login_required
def download_students_filtered():
    teacher_id = int(current_user.id)
    df = _load(teacher_id)
    if df is None:
        flash("No dataset to download yet.", "error")
        return redirect(url_for("students"))

    query = request.args.get("q", "").strip()
    program = request.args.get("program", "").strip()
    year = request.args.get("year", "").strip()
    status = request.args.get("status", "").strip()
    min_avg = request.args.get("min_avg", "").strip()

    filtered = data_service.filter_dataframe(
        df, search=query, program=program, year=year, status=status, min_avg=min_avg
    )
    export_df = data_service.build_enriched_dataframe(filtered, model_predict_batch_fn=_predict_fn())

    buffer = io.BytesIO()
    export_df.to_csv(buffer, index=False)
    buffer.seek(0)
    return send_file(
        buffer, mimetype="text/csv", as_attachment=True,
        download_name="aura_students_filtered.csv",
    )


@app.route("/analytics")
@login_required
def analytics():
    teacher_id = int(current_user.id)
    df = _load(teacher_id)
    charts = data_service.build_analytics(df)
    top_features = model_service.global_importance(df) if model_service.is_ready() else None
    return render_template(
        "analytics.html",
        charts=charts,
        top_features=top_features,
        model_info=model_service.model_info(),
        **_common_context("analytics"),
    )


@app.route("/recommendations")
@login_required
def recommendations():
    teacher_id = int(current_user.id)
    df = _load(teacher_id)
    buckets = data_service.build_risk_buckets(df)
    action_items = action_plan_service.list_items(teacher_id)
    return render_template(
        "recommendations.html",
        buckets=buckets,
        stages=[intervention_policy.STAGE_1, intervention_policy.STAGE_2, intervention_policy.STAGE_3],
        stable_suggestions=intervention_policy.STABLE_SUGGESTIONS,
        high_performing_suggestions=intervention_policy.HIGH_PERFORMING_SUGGESTIONS,
        action_items=action_items,
        statuses=action_plan_service.STATUSES,
        priorities=action_plan_service.PRIORITIES,
        **_common_context("recommendations"),
    )


@app.route("/action-plan/create", methods=["POST"])
@login_required
def action_plan_create():
    teacher_id = int(current_user.id)
    title = request.form.get("title", "")
    student_id = request.form.get("student_id", "").strip() or None
    notes = request.form.get("notes", "")
    priority = request.form.get("priority", "Medium")
    due_date = request.form.get("due_date", "").strip() or None
    status = request.form.get("status", "Pending")

    ok = action_plan_service.create_item(
        teacher_id, title, student_id=student_id, notes=notes,
        priority=priority, due_date=due_date, status=status,
    )
    flash("Action item added." if ok else "Please enter a title for the action item.", "success" if ok else "error")
    return redirect(url_for("recommendations"))


@app.route("/action-plan/<int:item_id>/status", methods=["POST"])
@login_required
def action_plan_update_status(item_id):
    teacher_id = int(current_user.id)
    status = request.form.get("status", "")
    ok = action_plan_service.update_status(teacher_id, item_id, status)
    if not ok:
        flash("Could not update that action item.", "error")
    return redirect(url_for("recommendations"))


@app.route("/action-plan/<int:item_id>/delete", methods=["POST"])
@login_required
def action_plan_delete(item_id):
    teacher_id = int(current_user.id)
    action_plan_service.delete_item(teacher_id, item_id)
    return redirect(url_for("recommendations"))


@app.route("/reload-model", methods=["POST"])
@login_required
def reload_model():
    """After copying a new AURA_model_bundle.pkl into /models, this reloads it
    without a restart and re-classifies this teacher's students."""
    model_service.reload_all()
    teacher_id = int(current_user.id)
    if model_service.is_ready():
        n = data_service.refresh_predictions(teacher_id, _predict_fn())
        flash(f"Model reloaded — {n:,} student(s) re-classified with the current AURA_model_bundle.pkl.", "success")
    else:
        flash("Model reloaded, but AURA_model_bundle.pkl could not be loaded — see Model Files below.", "error")
    return redirect(url_for("upload"))


# Vercel (and most serverless hosts) import this `app` object directly as
# the WSGI callable and never execute this block - app.run() is only for
# local development.
if __name__ == "__main__":
    app.run(debug=True)
