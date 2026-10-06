"""
intervention_policy.py
-----------------------
Encodes Angeles University Foundation's approved policy:
"Early Intervention and Academic Support for Students at Risk"
(Office of the VP for Academic Affairs, implemented January 25, 2024).

This is real institutional policy content, not a generic placeholder -
every step and checklist item below is taken directly from the approved
policy document and its appendices. Nothing here is invented.

The policy defines three stages:

  STAGE 1 - When a student fails/misses the first two quizzes or
            requirements during the midterm period.
  STAGE 2 - When a student obtains a Midterm grade other than "Passed".
  STAGE 3 - When a student obtains a Final grade other than "Passed".

Appendix C's monitoring checklist is reproduced as ACTION items so a
teacher can track exactly which interventions have been carried out per
student, matching how Program Chairs/Deans are expected to document
compliance in the real policy.

NOTE ON STAGE DETECTION: the source policy triggers stages from actual
quiz/requirement failures and official midterm/final grades. The AURA
AY 2025-2026 dataset has semester-level GWAs instead (1st-semester,
2nd-semester and annual, on AUF's percentage scale where 75 is passing).
determine_stage() below maps those to the closest stage as a heuristic -
it is disclosed as such everywhere it's used, not presented as if it were
the literal policy trigger.

The manuscript's decision-support framework (Sections 2.7.2-2.7.4) adds:
  - remediation strategies for At-Risk students,
  - enrichment strategies for High-Performing students,
  - prescriptive, explanation-driven suggestions (what the SHAP factors
    point to), see explanation_recommendations().
Those are the study's own NON-BINDING suggestions and are always shown
separately from the official AUF policy content.
"""

STAGE_1 = {
    "number": 1,
    "title": "Stage 1 — Missed First Two Quizzes/Requirements (Midterm Period)",
    "trigger": "Student fails or misses the first two consecutive quizzes/requirements in a subject.",
    "steps": [
        {
            "title": "Identify",
            "detail": "The teacher identifies the student as \"at risk\" in that subject.",
        },
        {
            "title": "Notify",
            "detail": (
                "The teacher promptly informs the student (via MyClass Inbox) about the "
                "failed or missed quizzes and requires the student to submit an Academic "
                "Self-Reflection Report (ASR) in PDF."
            ),
        },
        {
            "title": "Respond",
            "detail": (
                "Depending on the content of the ASR, the teacher schedules a meeting or "
                "initiates further communication with the student (email, virtual meeting, "
                "or in person)."
            ),
        },
        {
            "title": "Engage",
            "detail": (
                "The teacher offers guidance: relevant resources on study techniques, "
                "self-regulation, and test-taking skills; and/or peer-facilitated mentoring "
                "or referral to peer tutoring services within the college."
            ),
        },
        {
            "title": "Refer (if needed)",
            "detail": (
                "If challenges extend beyond academics, the teacher refers the student to the "
                "Guidance and Counseling Center (GCC) via a formal email (Program Chair cc'd), "
                "attaching the student's completed ASR."
            ),
        },
    ],
    "actions": [
        ("informed_failed_quizzes", "Informed the student about the failed quizzes/missed requirements"),
        ("required_asr", "Required the student to submit an Academic Self-Reflection Report (ASR)"),
        ("followup_after_asr", "Initiated further communication with the student after ASR submission and offered relevant resources"),
        ("scheduled_remedial", "Scheduled remedial/tutorial classes for at-risk students"),
        ("peer_mentoring", "Designed and implemented a peer-facilitated mentoring activity in class"),
        ("referred_peer_tutoring", "Referred student to peer tutoring services offered within the college"),
        ("referred_gcc", "Referred student to the Guidance and Counseling Center (GCC)"),
        ("maintained_records", "Maintained thorough records of all communications, meetings, and interventions"),
        ("ensured_confidentiality", "Ensured that all information regarding the at-risk student is kept confidential"),
    ],
}

STAGE_2 = {
    "number": 2,
    "title": "Stage 2 — Midterm Grade Other Than \"Passed\"",
    "trigger": "Student obtains any grade other than \"Passed\" in the Midterm period.",
    "steps": [
        {
            "title": "Post-Intervention Assessment",
            "detail": (
                "The teacher schedules a Student Performance Debrief to discuss and evaluate "
                "initial intervention strategies and identify new challenges. If the student "
                "was not previously flagged at-risk, a debrief is still conducted and an ASR "
                "is formally required afterward."
            ),
        },
        {
            "title": "Initial Parent Engagement",
            "detail": (
                "Parents/guardians may be informed of the student's midterm academic "
                "performance, per the Unit/College's criteria for parent notification "
                "(e.g. failing 30% or more of total units). The Program Chair sends a formal "
                "notification email (Dean cc'd) and requires prompt parent feedback."
            ),
        },
        {
            "title": "Parent-Teacher/Chair Meeting",
            "detail": (
                "Depending on the parent's response, the Program Chair arranges a "
                "parent-teacher meeting (virtual or in-person) to discuss the student's "
                "challenges and ongoing support plan."
            ),
        },
        {
            "title": "Continuous Monitoring",
            "detail": (
                "The teacher and Program Chair maintain ongoing communication with the "
                "student and parents, continuously monitoring progress and adjusting support "
                "as needed — remedial classes or peer tutoring may continue."
            ),
        },
    ],
    "actions": [
        ("performance_debrief", "Conducted Student Performance Debrief after release of Midterm Grades"),
        ("updated_parent", "Updated the parent or guardian about the student's academic performance"),
        ("parent_teacher_meeting", "Arranged/attended a parent-teacher meeting"),
        ("maintained_records", "Maintained thorough records of all communications, meetings, and interventions"),
        ("ensured_confidentiality", "Ensured that all information regarding the at-risk student is kept confidential"),
    ],
}

STAGE_3 = {
    "number": 3,
    "title": "Stage 3 — Final Grade Other Than \"Passed\"",
    "trigger": "Student obtains a Final grade other than \"Passed\".",
    "steps": [
        {
            "title": "Retention Policies / Alternative Career Options",
            "detail": (
                "If academic challenges persist and the current course or program appears "
                "unsuitable for the student, the Program Chair notifies the student and "
                "parents to reiterate retention policies and discuss alternative academic or "
                "career options."
            ),
        },
    ],
    "actions": [
        ("notified_retention_policy", "Notified student and parents of retention policy / alternative options"),
        ("maintained_records", "Maintained thorough records of all communications, meetings, and interventions"),
        ("ensured_confidentiality", "Ensured that all information regarding the at-risk student is kept confidential"),
    ],
}

STAGES = {1: STAGE_1, 2: STAGE_2, 3: STAGE_3}

# General reminders that apply across every stage (from the policy's
# "Other Reminders" section).
POLICY_REMINDERS = [
    "Documentation: maintain thorough records of all communications, meetings, and interventions for tracking and documentation purposes.",
    "Confidentiality: information regarding at-risk students is shared only with relevant school personnel and the student's parents or guardians.",
]

# --------------------------------------------------------------------------
# Non-binding suggestions from the manuscript (NOT part of the AUF policy)
# --------------------------------------------------------------------------

PASSING_GRADE = 75          # AUF passing grade (dataset Validation sheet)
AT_RISK_CUTOFF = 82         # At-Risk: annual GWA <= 82 (operational definition)
HIGH_PERFORMING_CUTOFF = 90 # High-Performing: annual GWA >= 90

# Manuscript 2.7.2 - common remediation strategies in higher education
AT_RISK_SUGGESTIONS = [
    "Structured tutoring program in the subjects pulling the GWA down.",
    "Supplemental instruction sessions for difficult core courses.",
    "Academic advising consultation to review the study plan and course load.",
    "Peer mentoring with an upper-year or high-performing student.",
    "Counseling and behavioral support services (GCC) if challenges go beyond academics.",
]

# Manuscript 2.7.3 - enrichment strategies for high-performing students
HIGH_PERFORMING_SUGGESTIONS = [
    "Honors or accelerated coursework opportunities.",
    "Research assistantship opportunities with faculty.",
    "Academic leadership programs (e.g. peer tutor or mentor role, org officer).",
    "Advanced certification tracks related to the program.",
    "Competitive scholarship preparation programs and Dean's List recognition.",
]

# Stable students - neither remediation nor enrichment by default: monitor.
STABLE_SUGGESTIONS = [
    "Keep regular monitoring — re-check the classification when the next term's grades are released.",
    "Offer optional study-skills resources and access to peer tutoring.",
    "Encourage participation in academic organizations to stay engaged.",
    "If the At-Risk probability is elevated, schedule a light check-in with the adviser.",
]

SUGGESTIONS_BY_LABEL = {
    "At-Risk": AT_RISK_SUGGESTIONS,
    "Stable": STABLE_SUGGESTIONS,
    "High-Performing": HIGH_PERFORMING_SUGGESTIONS,
}

# Prescriptive mapping (manuscript 2.7.4): which factor drives the
# prediction -> which action to consider. "risk" texts are used when the
# factor works AGAINST a strong outcome, "support" texts when it works FOR it.
FACTOR_ACTIONS = {
    "first_sem_gwa": {
        "risk": "1st-semester GWA is the main factor: arrange structured tutoring or supplemental instruction and an academic advising consultation.",
        "support": "Strong 1st-semester GWA: consider honors/accelerated coursework or competitive scholarship preparation.",
    },
    "gwa": {
        "risk": "Annual GWA is a key factor: review subject-level grades with the student and plan remediation for the weakest subjects.",
        "support": "Strong annual GWA: recommend for Dean's List recognition or research assistantship.",
    },
    "second_sem_gwa": {
        "risk": "2nd-semester GWA is a key factor: debrief with the student on what changed and adjust the support plan.",
        "support": "Strong 2nd-semester GWA: consider leadership or peer-mentor roles.",
    },
    "total_classes": {
        "risk": "Course load is contributing: review with the academic adviser whether the current number of subjects is manageable.",
        "support": "Course load is in a favorable range: no load adjustment needed.",
    },
    "year_level": {
        "risk": "Year-level pattern is contributing: provide transition support (orientation, peer mentoring with upper-year students).",
        "support": "Year-level pattern is favorable: consider year-appropriate enrichment (e.g. advanced electives).",
    },
    "program": {
        "risk": "Program pattern is contributing: coordinate with the Program Chair on program-specific supplemental instruction.",
        "support": "Program pattern is favorable: encourage program-related competitions or certification tracks.",
    },
    "student_group": {
        "risk": "The student's K-Means peer group leans weaker: consider group study sessions or peer mentoring.",
        "support": "The student's K-Means peer group leans stronger: invite the student to lead a study group.",
    },
    "attendance_rate": {
        "risk": "Attendance is contributing: start attendance monitoring and check in about barriers to attending.",
        "support": "Good attendance: acknowledge the consistency.",
    },
    "absences": {
        "risk": "Absences are contributing: start attendance monitoring and check in about barriers to attending.",
        "support": "Few absences: acknowledge the consistency.",
    },
    "lms_login_count": {
        "risk": "Low LMS engagement is contributing: set up structured engagement monitoring or guided study sessions.",
        "support": "High LMS engagement: point the student to advanced learning resources.",
    },
    "assignment_average": {
        "risk": "Assignment performance is contributing: review submission quality and offer guided practice.",
        "support": "Strong assignment performance: offer challenge tasks or project-based enrichment.",
    },
}


def explanation_recommendations(label, explanation, limit=3):
    """
    Prescriptive, explanation-driven suggestions (manuscript 2.7.4): for the
    top factors in this student's explanation, return the matching action.
    At-Risk / Stable students get actions for the factors working AGAINST
    them; High-Performing students get actions for the factors working FOR
    them. Returns a list of strings (may be empty).
    """
    if not explanation or not explanation.get("top_features"):
        return []
    want = "support" if label == "High-Performing" else "risk"
    out = []
    for f in explanation["top_features"]:
        if f["plain"]["direction"] != want:
            continue
        action = FACTOR_ACTIONS.get(f["feature"], {}).get(want)
        if action and action not in out:
            out.append(action)
        if len(out) >= limit:
            break
    return out


def determine_stage(row, prediction_label=None):
    """
    Heuristically maps a student's data to the closest matching policy
    stage. Returns None if the student isn't on an at-risk track.

    Uses the semester GWAs (AUF percentage scale, 75 = passing):
      Stage 3 - the year-end grade is not passing (annual or 2nd-sem GWA < 75)
      Stage 2 - the 1st-semester (earlier-term) GWA is not passing (< 75)
      Stage 1 - classified At-Risk without a failing term GWA (early warning)

    A heuristic from the columns the dataset actually has - not the policy's
    literal triggers (specific quiz failures, official midterm/final grades).
    Always disclosed as such in the UI.
    """
    label = prediction_label or row.get("target_class")
    if label != "At-Risk":
        return None
    gwa = _safe_float(row.get("gwa"))
    second = _safe_float(row.get("second_sem_gwa"))
    first = _safe_float(row.get("first_sem_gwa"))
    if (gwa is not None and gwa < PASSING_GRADE) or (second is not None and second < PASSING_GRADE):
        return 3
    if first is not None and first < PASSING_GRADE:
        return 2
    return 1


def _safe_float(v):
    try:
        v = float(v)
        return None if v != v else v
    except (TypeError, ValueError):
        return None
