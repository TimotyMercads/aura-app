-- ============================================================
-- AURA - panel revision migration (October 2026)
--
-- OPTIONAL. The app does this automatically the first time it starts
-- with the new code (services/data_service.ensure_students_schema), so
-- you only need this file if you prefer to migrate by hand.
--
-- What it does: keeps every student row, but removes the columns the
-- panel-revised model no longer uses (annual / 2nd-semester GWA,
-- attendance, LMS, total classes, dates), and clears the old model's
-- predictions so the new XGBoost model re-classifies everyone on the
-- next page load. Users, uploads, checklists and action plans are kept.
--
-- BACK UP FIRST (phpMyAdmin -> Export, or pg_dump).
-- ============================================================
-- XAMPP (MariaDB) / MySQL. "DROP COLUMN IF EXISTS" needs MariaDB 10.0+
-- (what XAMPP ships). On Oracle MySQL 8, remove the words "IF EXISTS".
USE aura_db;

ALTER TABLE students
    DROP COLUMN IF EXISTS academic_year,
    DROP COLUMN IF EXISTS semester,
    DROP COLUMN IF EXISTS gwa,
    DROP COLUMN IF EXISTS assignment_average,
    DROP COLUMN IF EXISTS second_sem_gwa,
    DROP COLUMN IF EXISTS total_classes,
    DROP COLUMN IF EXISTS classes_attended,
    DROP COLUMN IF EXISTS absences,
    DROP COLUMN IF EXISTS attendance_rate,
    DROP COLUMN IF EXISTS absenteeism_rate,
    DROP COLUMN IF EXISTS lms_login_count,
    DROP COLUMN IF EXISTS lms_resource_views,
    DROP COLUMN IF EXISTS lms_assignment_submissions,
    DROP COLUMN IF EXISTS on_time_submissions,
    DROP COLUMN IF EXISTS late_submissions,
    DROP COLUMN IF EXISTS lms_activity_count,
    DROP COLUMN IF EXISTS avg_weekly_lms_logins,
    DROP COLUMN IF EXISTS avg_daily_activity,
    DROP COLUMN IF EXISTS first_activity_date,
    DROP COLUMN IF EXISTS last_activity_date;

UPDATE students SET predicted_class = NULL, prediction_confidence = NULL, prob_at_risk = NULL;
