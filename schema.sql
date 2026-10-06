-- ============================================================
-- AURA database schema (MySQL / MariaDB / XAMPP)
-- Run this once in phpMyAdmin (Import tab) against a database
-- named aura_db, or via the mysql CLI:
--   mysql -u root aura_db < schema.sql
--
-- Deploying to a cloud MySQL instead of XAMPP? This same file
-- works unchanged - just run it against that database instead.
-- Deploying to Postgres (e.g. for Vercel + a cloud Postgres)?
-- Use schema_postgres.sql instead.
-- ============================================================

CREATE DATABASE IF NOT EXISTS aura_db;
USE aura_db;

-- ------------------------------------------------------------
-- Users (login / signup)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    full_name VARCHAR(150) NOT NULL,
    email VARCHAR(150) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(30) NOT NULL DEFAULT 'teacher',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ------------------------------------------------------------
-- Upload history (every file a teacher has uploaded)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS uploads (
    id INT AUTO_INCREMENT PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    file_type VARCHAR(10) NOT NULL,
    row_count INT NOT NULL DEFAULT 0,
    uploaded_by INT NULL,
    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    clean_report JSON NULL,
    uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_uploads_owner (uploaded_by),
    CONSTRAINT fk_uploads_user FOREIGN KEY (uploaded_by) REFERENCES users(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

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
    id INT AUTO_INCREMENT PRIMARY KEY,
    upload_id INT NULL,
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
    prob_at_risk FLOAT NULL,
    INDEX idx_student_id (student_id),
    INDEX idx_students_upload (upload_id),
    CONSTRAINT fk_students_upload FOREIGN KEY (upload_id) REFERENCES uploads(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ------------------------------------------------------------
-- Intervention log - the AUF Early Intervention Policy checklist,
-- tracked PER STUDENT_ID (not a foreign key to students.id, since
-- a teacher's students rows get replaced on re-upload) - this table
-- is deliberately independent so a teacher's tracked progress on a
-- student survives a dataset re-upload, as long as the same
-- student_id reappears.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS intervention_log (
    id INT AUTO_INCREMENT PRIMARY KEY,
    student_id VARCHAR(50) NOT NULL,
    stage TINYINT NOT NULL,
    action_key VARCHAR(100) NOT NULL,
    completed BOOLEAN NOT NULL DEFAULT FALSE,
    completed_at DATETIME NULL,
    completed_by INT NULL,
    notes TEXT NULL,
    UNIQUE KEY uniq_student_action (student_id, action_key),
    INDEX idx_intervention_student (student_id),
    CONSTRAINT fk_intervention_user FOREIGN KEY (completed_by) REFERENCES users(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ------------------------------------------------------------
-- Action plan - the Teacher Action Plan / reminders & scheduling
-- feature. Each item belongs to one teacher (created_by) and
-- optionally references one student. status is one of:
-- 'Pending', 'Scheduled', 'Ongoing', 'Finished'.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS action_plans (
    id INT AUTO_INCREMENT PRIMARY KEY,
    created_by INT NOT NULL,
    student_id VARCHAR(50) NULL,
    title VARCHAR(255) NOT NULL,
    notes TEXT NULL,
    priority VARCHAR(10) NOT NULL DEFAULT 'Medium',
    due_date DATE NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'Pending',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_action_plans_owner (created_by),
    CONSTRAINT fk_action_plans_user FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
