# AURA — Academic At-Risk & High-Performing Student Dashboard

Flask web app for the "Implementation and Evaluation of a Supervised Machine
Learning Model with Explainable AI for Predicting At-Risk and High-Performing
College Students" thesis project.

## What changed in this version (October 2026)

- **New model:** one file, `models/AURA_model_bundle.pkl`, exported by
  `Dataset_Training.ipynb` - the notebook's selected **KMeans + Random
  Forest** pipeline (tied with the top repeated-CV macro-F1, highest
  At-Risk recall among tied models, exact SHAP). It replaces the old
  Best_Model/Scaler/TargetEncoder/Categorical_Encoders files.
  `aura_transformers.py` (next to `app.py`) is required to load it.
- **New data schema:** the AURA AY 2025-2026 dataset (24 columns +
  optional `target_class`, no demographics). The xlsx `dataset` sheet is
  read automatically, and `first_sem_gwa(previous_gwa)` is recognized.
- **Three classes:** At-Risk / Stable / High-Performing everywhere.
  The model's prediction is each student's status, so unlabeled files
  work; a recorded `target_class` is shown alongside when present.
- **Database:** the `students` table now uses the dataset's own column
  names. Existing installs migrate automatically on first start (only the
  old students rows are replaced; accounts, uploads history, checklists
  and action plans are kept).
- **Manuscript alignment:** exclusion criteria (missing critical inputs,
  >20% missing), plain-language explanations, non-binding /
  human-in-the-loop notes, remediation + enrichment + monitoring
  pathways and explanation-driven suggestions (Sections 2.7.2-2.7.4),
  and a Model Card with the held-out test metrics.
- **UI fixes:** the password show/hide toggle is now a simple eye icon
  inside the field, and the Upload page shows the chosen file's name
  (and a "Currently loaded" indicator after uploading).

## First-time setup

This version requires **MySQL** (via XAMPP) or a **cloud MySQL/Postgres**
database. For local development, follow **`XAMPP_SETUP.md`** — a short,
mostly point-and-click guide. To publish a live version on Vercel with a
cloud database instead, follow **`DEPLOYMENT.md`**. Same codebase either
way — just a different environment variable.

## Multi-teacher accounts

Every signed-up teacher only ever sees and edits the students from
datasets **they** uploaded — not everyone shares one dataset anymore.
Uploading replaces only your own previous data; other teachers' rosters
are untouched. This is enforced at the database query level (see
`data_service.py`), not just hidden in the UI — even a direct URL to
another teacher's student returns "not found," not their data. Each
student now also has an **Edit** page for hand-correcting a record.

## Performance

Two real bottlenecks were fixed this round, both measured before/after,
not just assumed:
- **Bulk predictions** (on upload, and on CSV export) now run in one
  vectorized batch instead of one model call per row — **~266x faster**
  on a 1,000-row test (from ~2 minutes down to well under a second for
  the model-inference part).
- **The SHAP explainer** is now cached per loaded model instead of being
  rebuilt on every single request — the first explanation after a model
  loads takes its normal ~10 seconds, but every one after that is
  **~650x faster** (tens of milliseconds).

## Plain-language explanations

The "Why This Prediction?" page shows no raw SHAP numbers up front - every
factor gets a plain name, a naturally formatted value (a GWA, "378 class
meetings", "Year 2"), a short note on what it generally means, and a
Major/Moderate/Minor bar. ⚠️ factors pull toward At-Risk and ✅ factors
toward High-Performing (computed as SHAP(High-Performing) - SHAP(At-Risk),
which keeps a 3-class model readable on one scale). One-hot columns are
combined into one factor (Program, Student Group). Advisers can expand a
"Technical detail" table with the exact SHAP values for the predicted class.

## Teacher Action Plan (reminders & scheduling)

The Recommendations page's "Teacher Action Plan" is no longer a static
example table — add your own reminders (optionally tied to a student),
set a priority and due date, and track status through **Pending →
Scheduled → Ongoing → Finished**. Items due within 7 days or overdue get
an in-app "Due Soon"/"Overdue" badge, and a small notification banner
appears on the Dashboard summarizing how many need attention. (This is
in-app only — no email/push notifications are configured.)

## Design

The interface was redesigned around AUF's own seal rather than a generic
dashboard template — every color is drawn from the crest itself:

- **Ink Navy / Deep Harbor** — background and panels (the crest's blue field)
- **Torchlight Gold** — primary actions and active states (the crest's laurel/flame)
- **Ember Maroon** — At-Risk status and secondary actions (the crest's red glyphs)
- **Harbor Blue** — Stable status and informational accents (policy stage badges)
- **Sage** — High-Performing status

Numeric readouts (grades, percentages, counts) use IBM Plex Mono instead
of the body font, for an "instrument panel" feel appropriate to a
monitoring dashboard. A soft radial gold glow — echoing the torch at the
center of the seal — appears behind the login card and the dashboard's
welcome banner as a consistent, understated signature element.

The AUF seal itself appears on the Login and Sign Up pages, and in the
top-right corner of every page once logged in (`static/img/auf_logo.png`).

**Charts** were a specific complaint in an earlier pass — Analytics
previously stacked full-width charts with no height limit, requiring a
lot of scrolling. Every chart now sits in a fixed-height container
(`.chart-wrap`) paired with Chart.js's `maintainAspectRatio: false`, and
the Attendance-by-Year-Level / Academic-Component-Breakdown charts were
moved into a 2-column grid instead of stacking — the whole Analytics page
now fits in roughly one screen instead of four separate full-width chart
blocks.

## What's wired up right now

- **Login / Sign Up / Logout** — every page requires a teacher account.
  Sign up with a name, email, and password (hashed, never stored in plain
  text); log in to reach the dashboard; log out clears your session. If
  MySQL isn't running, you'll see a friendly "can't reach the database"
  page instead of a crash, telling you to start it in XAMPP.
- **Dashboard** — total / High-Performing / Stable / At-Risk counts and a
  distribution chart, from the model's predictions for your students.
- **Students** — searchable table (Student ID, Program, Year Level,
  minimum GWA, status) with each prediction's confidence, plus
  **Download Results** for exactly the filtered rows.
- **Student Insights** — GWA / 1st-semester GWA / attendance / LMS
  activity, the live prediction with all three class probabilities (and
  the recorded class if the file had one), Key Factors bars sized by SHAP
  per factor group, and Recommended Actions.
  - **Explanation** → plain-language factor cards + technical SHAP table.
  - **More** → the AUF policy stages with a trackable checklist (At-Risk),
    or the monitoring / enrichment pathway (Stable / High-Performing),
    plus suggestions driven by the student's own explanation.
  - **Edit** → change any field; the student is re-classified on save.
- **Upload Dataset** — .csv / .xlsx / .xls only; required columns
  `student_id, program, year_level, first_sem_gwa, total_classes`.
  Cleaning (`utils/cleaning.py`) maps header aliases, removes demographic
  columns, drops empty/duplicate rows and duplicate IDs, coerces and clips
  values, converts percent rates to 0-1, applies the manuscript's
  exclusion criteria, and imputes the rest (never the label) - all shown
  in the Cleaning Report. Every student is then classified in one batch.
- **Analytics** — average GWA, attendance, class distribution,
  attendance by year level, classification by program, top predictors
  (mean |SHAP|), and a **Model Card** with the notebook's held-out test
  metrics.
- **Recommendations** — At-Risk (intervene) / Stable (monitor) /
  High-Performing (enrich) counts, the three AUF policy stages, the
  monitoring and enrichment pathways, the At-Risk list sorted by
  probability, an early-warning list of Stable students with ≥20%
  At-Risk probability, and the Teacher Action Plan.

## Interventions are based on AUF's actual approved policy

The Recommended Actions / Explanation / Recommendations pages no longer
show generic placeholder suggestions — they're built from AUF's approved
**"Early Intervention and Academic Support for Students at Risk"** policy
(Office of the VP for Academic Affairs, implemented January 25, 2024),
encoded in `utils/intervention_policy.py`:

- **Stage 1** — student fails/misses the first two quizzes or requirements
  during the midterm period (Identify → Notify → Respond → Engage → Refer).
- **Stage 2** — student obtains a Midterm grade other than "Passed"
  (Performance Debrief → Parent Engagement → Parent-Teacher Meeting →
  Continuous Monitoring).
- **Stage 3** — student obtains a Final grade other than "Passed"
  (retention policy / alternative career options discussion).

Each stage shows its real checklist from the policy's Appendix C, and a
teacher can **check items off per student** — this is saved to the
database (`intervention_log` table) and persists across dataset
re-uploads (as long as the same student_id reappears), with who completed
it and when.

**Which stage a student is shown is a heuristic**, not a literal
implementation of the policy's real triggers — the dataset has semester
GWAs rather than quiz/requirement failures, so `determine_stage()` uses:
Stage 3 if the annual or 2nd-semester GWA is below 75 (AUF passing grade),
Stage 2 if the 1st-semester GWA is below 75, otherwise Stage 1. This is
disclosed in the UI. Stable and High-Performing students see clearly
labeled monitoring / enrichment suggestions from the manuscript instead —
the real policy only covers at-risk students.

## Project structure

```
flaskapp/
├── app.py                       Routes, auth, multi-tenant scoping, DB-error handling
├── aura_transformers.py         The notebook's custom pipeline classes (needed by the model)
├── schema.sql                   MySQL schema (import via phpMyAdmin)
├── schema_postgres.sql          Postgres equivalent (for cloud deployment)
├── vercel.json                  Vercel deployment config
├── XAMPP_SETUP.md               Step-by-step local XAMPP/MySQL setup guide
├── DEPLOYMENT.md                Cloud database + Vercel deployment guide
├── .env.example                 DB config template (copy to .env if needed)
├── requirements.txt
├── services/
│   ├── auth_service.py          Signup/login logic (password hashing)
│   ├── data_service.py          Dataset loading/saving - DB-backed, per-teacher scoped
│   ├── model_service.py         Loads AURA_model_bundle.pkl, batch predicts, SHAP explanations
│   ├── intervention_service.py  Tracks the policy checklist per student
│   └── action_plan_service.py   Teacher Action Plan reminders/scheduling
├── utils/
│   ├── db.py                    SQLAlchemy engine - works with MySQL or Postgres
│   ├── cleaning.py              Upload cleaning pipeline
│   ├── intervention_policy.py   The AUF policy content itself
│   └── plain_language.py        Parent-friendly explanation content
├── templates/                   Jinja templates (base.html + one per page)
├── static/
│   ├── css/style.css            AUF-branded design
│   ├── js/password-toggle.js    Show/hide password eye icon
│   └── img/auf_logo.png         AUF seal
└── models/
    └── AURA_model_bundle.pkl    The selected model (see models/README.md)
```

## Running it

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Make sure **MySQL is running in XAMPP first** (see `XAMPP_SETUP.md`).
Then open **http://127.0.0.1:5000** — it'll send you to the login page.
Sign up for your first account, then log in.

## Getting real data & predictions flowing

1. The model is already in place (`models/AURA_model_bundle.pkl` +
   `aura_transformers.py`). The Upload page's **Model Files** section
   confirms it loaded and shows the selected model.
2. Log in, go to **Upload Dataset** and upload the AURA workbook (or any
   CSV/XLSX with the same columns). Every student is classified
   immediately and every page switches from demo numbers to real ones.
3. After replacing the bundle with a retrained one, click **Reload
   Model** - your students are re-classified without a restart.

Nothing crashes if the model is missing - every page shows a status
banner and safe fallback content until it's added.

## What's not done yet (flagged for later)

- Role-based permissions — every signed-up account currently has equal
  access; there's no admin/teacher distinction, and no way for one
  teacher to intentionally share a student with another (by design, per
  this round's changes, they're fully isolated instead).
- "Forgot password" — not implemented; a teacher would need a new account
  or a manual database fix today.
- No way to add a brand-new student by hand outside of a CSV/XLSX upload
  (existing students can be edited).
- Exact SHAP needs the `shap` package (in requirements.txt). If it isn't
  installed, the app still explains predictions with a tree-path
  decomposition (Saabas method) that approximates SHAP, and labels it as such.
- Email/push notifications for the Teacher Action Plan — the "Notification
  basing on status" request was interpreted as in-app badges + a dashboard
  banner, not actual emails or push notifications, since no email service
  is configured. Worth a follow-up if real notifications (not just in-app)
  are wanted.
- Vercel deployment is documented and the code is compatible (see
  `DEPLOYMENT.md`), but hasn't been deployed to a live Vercel project as
  part of this work — only tested against local MySQL and local Postgres
  servers standing in for their cloud equivalents.
