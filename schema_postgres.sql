-- ============================================================
-- AURA database schema (PostgreSQL)
-- Use this instead of schema.sql when your database is Postgres
-- (e.g. a cloud Postgres add-on used alongside a Vercel deployment
-- - Neon, Supabase, Railway, Render Postgres, etc).
--
-- Run it once against your database, e.g.:
--   psql "$DATABASE_URL" -f schema_postgres.sql
-- or paste it into your host's SQL console (most cloud Postgres
-- dashboards have one).
--
-- Column names are the AURA dataset's lowercase snake_case names
-- (student_id, first_sem_gwa, ...), so no quoting is needed.
--
-- This does NOT include "CREATE DATABASE" - most managed Postgres
-- hosts give you an already-created database and only let you
-- connect to that one, so just run this against it directly.
-- ============================================================

-- ------------------------------------------------------------
-- Users (login / signup)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    full_name VARCHAR(150) NOT NULL,
    email VARCHAR(150) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(30) NOT NULL DEFAULT 'teacher',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ------------------------------------------------------------
-- Upload history (every file a teacher has uploaded)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS uploads (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    file_type VARCHAR(10) NOT NULL,
    row_count INT NOT NULL DEFAULT 0,
    uploaded_by INT NULL REFERENCES users(id) ON DELETE SET NULL,
    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    clean_report JSON NULL,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_uploads_owner ON uploads (uploaded_by);

-- ------------------------------------------------------------
-- Students. Each row belongs to one upload (upload_id), and each
-- upload belongs to one teacher (uploads.uploaded_by) - that chain
-- is how "a teacher can only see/edit what they uploaded" is
-- enforced. Re-uploading only replaces THAT teacher's own rows
-- (see data_service.save_dataset) - other teachers' students are
-- untouched.
--
-- Columns = the AURA AY 2025-2026 dataset sheet (no demographics) plus
-- the model's outputs (predicted_class, prediction_confidence,
-- prob_at_risk). target_class is optional (unlabeled students are
-- classified by the model).
--
-- UPGRADING from the previous AURA version? Nothing to do by hand: on
-- first start the app detects the old students table (Student_ID,
-- Previous_GPA, ...) and recreates it with these columns. Users, upload
-- history, checklists and action plans are kept.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS students (
    id SERIAL PRIMARY KEY,
    upload_id INT NULL REFERENCES uploads(id) ON DELETE SET NULL,
    student_id VARCHAR(50) NOT NULL,
    academic_year VARCHAR(20) NULL,
    semester VARCHAR(60) NULL,
    program VARCHAR(20) NULL,
    year_level INT NULL,
    gwa FLOAT NULL,
    first_sem_gwa FLOAT NULL,
    assignment_average FLOAT NULL,
    second_sem_gwa FLOAT NULL,
    total_classes INT NULL,
    classes_attended INT NULL,
    absences INT NULL,
    attendance_rate FLOAT NULL,
    absenteeism_rate FLOAT NULL,
    lms_login_count INT NULL,
    lms_resource_views INT NULL,
    lms_assignment_submissions INT NULL,
    on_time_submissions INT NULL,
    late_submissions INT NULL,
    lms_activity_count INT NULL,
    avg_weekly_lms_logins FLOAT NULL,
    avg_daily_activity FLOAT NULL,
    first_activity_date DATE NULL,
    last_activity_date DATE NULL,
    target_class VARCHAR(20) NULL,
    predicted_class VARCHAR(20) NULL,
    prediction_confidence FLOAT NULL,
    prob_at_risk FLOAT NULL
);

CREATE INDEX IF NOT EXISTS idx_student_id ON students (student_id);
CREATE INDEX IF NOT EXISTS idx_students_upload ON students (upload_id);

-- ------------------------------------------------------------
-- Intervention log - the AUF Early Intervention Policy checklist,
-- tracked PER STUDENT_ID (not a foreign key to students.id, since
-- a teacher's students rows get replaced on re-upload) - this table
-- is deliberately independent so a teacher's tracked progress on a
-- student survives a dataset re-upload, as long as the same
-- student_id reappears.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS intervention_log (
    id SERIAL PRIMARY KEY,
    student_id VARCHAR(50) NOT NULL,
    stage SMALLINT NOT NULL,
    action_key VARCHAR(100) NOT NULL,
    completed BOOLEAN NOT NULL DEFAULT FALSE,
    completed_at TIMESTAMP NULL,
    completed_by INT NULL REFERENCES users(id) ON DELETE SET NULL,
    notes TEXT NULL,
    CONSTRAINT uniq_student_action UNIQUE (student_id, action_key)
);

CREATE INDEX IF NOT EXISTS idx_intervention_student ON intervention_log (student_id);

-- ------------------------------------------------------------
-- Action plan - the Teacher Action Plan / reminders & scheduling
-- feature. Each item belongs to one teacher (created_by) and
-- optionally references one student. status is one of:
-- 'Pending', 'Scheduled', 'Ongoing', 'Finished'.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS action_plans (
    id SERIAL PRIMARY KEY,
    created_by INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    student_id VARCHAR(50) NULL,
    title VARCHAR(255) NOT NULL,
    notes TEXT NULL,
    priority VARCHAR(10) NOT NULL DEFAULT 'Medium',
    due_date DATE NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'Pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_action_plans_owner ON action_plans (created_by);
