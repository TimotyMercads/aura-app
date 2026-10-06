"""
plain_language.py
------------------
Turns a technical SHAP explanation into something a teacher can read
aloud to a parent who has never heard of "SHAP," "confidence scores," or
machine learning at all (manuscript 3.2.1: outputs use "clear, plain, and
non-technical language").

Three things happen here that the raw SHAP output does NOT give you:
  1. Every factor gets a plain English name and a natural way to state its
     value (a GWA, "378 class meetings", "Year 2").
  2. Every factor gets a short sentence explaining what a weak/strong value
     generally means for a student.
  3. The SHAP magnitude (a decimal like 0.0523) becomes a 3-level rating -
     Major / Moderate / Minor - shown as a bar, not a number.

Nothing here changes the prediction or the SHAP math (that's in
model_service.py) - this only changes how it's DESCRIBED.

The model's factors are the columns of the AURA AY 2025-2026 dataset
(see the Data_Dictionary sheet). One-hot columns are shown as ONE factor:
program_BSCS/program_BSIT/program_BMMA -> "Program", and the K-Means
cluster_0/cluster_1 columns -> "Student Group".
"""


def _gwa(v):
    return f"{float(v):.2f}"


def _int(unit):
    return lambda v: f"{float(v):,.0f} {unit}"


def _pct_from_rate(v):
    v = float(v)
    return f"{v * 100:.1f}%" if v <= 1.5 else f"{v:.1f}%"


# plain_name, group (for the Key Factors bars), value format, and two short
# phrases - one for when the factor works AGAINST a strong outcome
# (risk_note) and one for when it works FOR one (support_note).
FEATURE_INFO = {
    # ---- Academic performance (REAL / DERIVED from registrar transcripts) ----
    "first_sem_gwa": {
        "plain_name": "1st-Semester GWA",
        "group": "Academic Performance",
        "format": _gwa,
        "risk_note": "A lower GWA in the first semester is the clearest early sign that the student may struggle for the rest of the year.",
        "support_note": "A strong first-semester GWA shows a solid academic foundation going into the rest of the year.",
    },
    "gwa": {
        "plain_name": "Annual GWA",
        "group": "Academic Performance",
        "format": _gwa,
        "risk_note": "A lower GWA for the whole year points to ongoing difficulty across subjects.",
        "support_note": "A strong GWA for the whole year shows consistent performance across subjects.",
    },
    "second_sem_gwa": {
        "plain_name": "2nd-Semester GWA",
        "group": "Academic Performance",
        "format": _gwa,
        "risk_note": "A lower second-semester GWA suggests difficulties have continued or grown.",
        "support_note": "A strong second-semester GWA shows the student kept performing well.",
    },
    "sem_gwa_change": {
        "plain_name": "Grade Change (2nd vs 1st Semester)",
        "group": "Academic Performance",
        "format": lambda v: f"{float(v):+.2f} pts",
        "risk_note": "Grades going down from one semester to the next is an early warning worth checking in about.",
        "support_note": "Grades holding steady or improving between semesters is a good sign.",
    },
    "assignment_average": {
        "plain_name": "Assignment Average",
        "group": "Academic Performance",
        "format": lambda v: f"{float(v):.1f}",
        "risk_note": "Lower assignment scores can point to rushed work, missed instructions, or needing more time.",
        "support_note": "Strong assignment scores show consistent effort on take-home work.",
    },
    # ---- Course load / attendance ----
    "total_classes": {
        "plain_name": "Scheduled Class Meetings (Course Load)",
        "group": "Course Load",
        "format": _int("class meetings"),
        "risk_note": "The number of class meetings in the student's load is linked to a weaker outcome here - a heavy or unusual load can make it harder to keep up with every subject.",
        "support_note": "The student's course load is in a range associated with stronger outcomes in this dataset.",
    },
    "classes_attended": {
        "plain_name": "Classes Attended",
        "group": "Attendance",
        "format": _int("classes"),
        "risk_note": "Attending fewer classes means more missed lessons to catch up on.",
        "support_note": "Attending most classes gives the best chance to catch everything taught.",
    },
    "absences": {
        "plain_name": "Absences",
        "group": "Attendance",
        "format": _int("absence(s)"),
        "risk_note": "More absences means more missed lessons and instructions to catch up on.",
        "support_note": "Few absences means the student was present for most of the material.",
    },
    "attendance_rate": {
        "plain_name": "Attendance Rate",
        "group": "Attendance",
        "format": _pct_from_rate,
        "risk_note": "Missing class regularly makes it much harder to keep up with new material.",
        "support_note": "Showing up consistently gives the best chance to catch everything taught.",
    },
    "absenteeism_rate": {
        "plain_name": "Absenteeism Rate",
        "group": "Attendance",
        "format": _pct_from_rate,
        "risk_note": "A higher share of missed classes is a common warning sign.",
        "support_note": "A low share of missed classes supports steady progress.",
    },
    # ---- LMS engagement ----
    "lms_login_count": {
        "plain_name": "LMS Logins",
        "group": "LMS Engagement",
        "format": _int("logins"),
        "risk_note": "Logging into the LMS less often can mean missing announcements or materials.",
        "support_note": "Frequent LMS logins show consistent engagement with the course.",
    },
    "lms_resource_views": {
        "plain_name": "LMS Resource Views",
        "group": "LMS Engagement",
        "format": _int("views"),
        "risk_note": "Viewing fewer learning materials can mean gaps in the assigned content.",
        "support_note": "Regularly opening learning materials shows active use of course resources.",
    },
    "lms_assignment_submissions": {
        "plain_name": "Assignments Submitted on the LMS",
        "group": "LMS Engagement",
        "format": _int("submissions"),
        "risk_note": "Fewer submitted assignments directly affects grades and understanding.",
        "support_note": "Consistently submitting work keeps grades and understanding on track.",
    },
    "on_time_submissions": {
        "plain_name": "On-Time Submissions",
        "group": "LMS Engagement",
        "format": _int("on time"),
        "risk_note": "Fewer on-time submissions can signal time-management difficulties.",
        "support_note": "Submitting work on time shows good planning habits.",
    },
    "late_submissions": {
        "plain_name": "Late Submissions",
        "group": "LMS Engagement",
        "format": _int("late"),
        "risk_note": "Frequent late work can signal the student is falling behind.",
        "support_note": "Rarely submitting late shows the student keeps up with deadlines.",
    },
    "lms_activity_count": {
        "plain_name": "Total LMS Activity",
        "group": "LMS Engagement",
        "format": _int("actions"),
        "risk_note": "Lower overall LMS activity can mean less contact with course content.",
        "support_note": "High overall LMS activity shows active engagement.",
    },
    "avg_weekly_lms_logins": {
        "plain_name": "Average Weekly LMS Logins",
        "group": "LMS Engagement",
        "format": lambda v: f"{float(v):.1f} per week",
        "risk_note": "Logging in only a few times a week can mean falling out of touch with the course.",
        "support_note": "Logging in regularly every week shows steady engagement.",
    },
    "avg_daily_activity": {
        "plain_name": "Average Daily LMS Activity",
        "group": "LMS Engagement",
        "format": lambda v: f"{float(v):.1f} per day",
        "risk_note": "Low day-to-day activity can mean less practice with course material.",
        "support_note": "Steady day-to-day activity supports consistent learning.",
    },
    "on_time_ratio": {
        "plain_name": "Share of Work Submitted On Time",
        "group": "LMS Engagement",
        "format": _pct_from_rate,
        "risk_note": "A lower share of on-time work can signal time-management difficulties.",
        "support_note": "Most work submitted on time shows good planning habits.",
    },
    "activity_per_login": {
        "plain_name": "LMS Activity per Login",
        "group": "LMS Engagement",
        "format": lambda v: f"{float(v):.1f}",
        "risk_note": "Doing little per LMS visit can mean only skimming the material.",
        "support_note": "Doing a lot per LMS visit shows focused study sessions.",
    },
    "lms_active_days": {
        "plain_name": "Days Active on the LMS",
        "group": "LMS Engagement",
        "format": _int("days"),
        "risk_note": "A shorter active period on the LMS can mean starting late or stopping early.",
        "support_note": "Staying active on the LMS across the whole year is a good sign.",
    },
    # ---- Program / year level / K-Means group ----
    "year_level": {
        "plain_name": "Year Level",
        "group": "Program & Year Level",
        "format": lambda v: f"Year {float(v):.0f}",
        "risk_note": "Students at this year level showed a slightly higher chance of difficulty in this dataset. This describes a group pattern, not the student personally.",
        "support_note": "Students at this year level generally did well in this dataset. This describes a group pattern, not the student personally.",
    },
    "program": {
        "plain_name": "Program",
        "group": "Program & Year Level",
        "format": lambda v: str(v),
        "risk_note": "The program's grading and workload pattern leaned slightly toward a weaker outcome in this dataset. This describes a group pattern, not the student personally.",
        "support_note": "The program's grading and workload pattern leaned slightly toward a stronger outcome in this dataset. This describes a group pattern, not the student personally.",
    },
    "student_group": {
        "plain_name": "Student Group (K-Means)",
        "group": "Student Group (K-Means)",
        "format": lambda v: str(v),
        "risk_note": "The clustering step placed the student with peers whose overall profile is linked to weaker results.",
        "support_note": "The clustering step placed the student with peers whose overall profile is linked to stronger results.",
    },
}


def _fallback_info(feature_key):
    return {
        "plain_name": feature_key.replace("_", " ").title(),
        "group": "Other",
        "format": lambda v: f"{v}",
        "risk_note": "This is one of the factors the model weighed for this student.",
        "support_note": "This is one of the factors the model weighed for this student.",
    }


def _info(feature_key):
    if feature_key in FEATURE_INFO:
        return FEATURE_INFO[feature_key]
    if feature_key.startswith("program_"):
        return FEATURE_INFO["program"]
    if feature_key.startswith("cluster_"):
        return FEATURE_INFO["student_group"]
    return _fallback_info(feature_key)


def friendly_name(feature_key):
    return _info(feature_key)["plain_name"]


def factor_group(feature_key):
    return _info(feature_key)["group"]


def format_value(feature_key, raw_value):
    if raw_value is None or raw_value == "":
        return "—"
    try:
        if raw_value != raw_value:     # NaN
            return "—"
    except Exception:
        pass
    try:
        return _info(feature_key)["format"](raw_value)
    except (TypeError, ValueError):
        return str(raw_value)


def magnitude_level(shap_value, max_abs_shap):
    """
    Converts a raw SHAP number into a 3-level plain rating. Returns
    (level: 'Major'|'Moderate'|'Minor', pct: 0-100 for a visual bar).
    """
    if max_abs_shap <= 0:
        return "Minor", 10
    ratio = abs(shap_value) / max_abs_shap
    pct = max(round(ratio * 100), 8)
    if ratio >= 0.66:
        return "Major", pct
    if ratio >= 0.33:
        return "Moderate", pct
    return "Minor", pct


def describe_feature(feature_key, raw_value, direction, shap_value, max_abs_shap):
    """
    One plain-language explanation card for a single factor - no SHAP
    numbers, no jargon.

    Returns {plain_name, formatted_value, note, level, level_pct, direction}
    where direction is "risk" (working against a strong outcome) or
    "support" (working in the student's favor).
    """
    info = _info(feature_key)
    is_risk = direction == "increases_risk"
    level, level_pct = magnitude_level(shap_value, max_abs_shap)
    return {
        "plain_name": info["plain_name"],
        "formatted_value": format_value(feature_key, raw_value),
        "note": info["risk_note"] if is_risk else info["support_note"],
        "level": level,
        "level_pct": level_pct,
        "direction": "risk" if is_risk else "support",
    }


# --------------------------------------------------------------------------
# One readable paragraph per student (shown first on the Explanation page)
# --------------------------------------------------------------------------

# How each factor reads inside a sentence ({v} = its formatted value).
_PARAGRAPH_PHRASES = {
    "first_sem_gwa": "the 1st-semester GWA of {v}",
    "gwa": "the annual GWA of {v}",
    "second_sem_gwa": "the 2nd-semester GWA of {v}",
    "total_classes": "a course load of {v}",
    "year_level": "being in {v}",
    "program": "being in the {v} program",
    "student_group": "being placed in {v} by the clustering step",
    "attendance_rate": "an attendance rate of {v}",
    "absences": "{v} of absences",
    "lms_login_count": "{v} on the LMS",
}


def _phrase(f):
    value = f["plain"]["formatted_value"]
    template = _PARAGRAPH_PHRASES.get(f["feature"])
    if template:
        return template.format(v=value)
    return f"{f['plain']['plain_name']} ({value})"


def _join(items):
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _cap(text):
    return text[:1].upper() + text[1:]


_CLOSING = {
    "At-Risk": "A short check-in with the student, and the suggested actions on the student's page, would be a sensible next step.",
    "Stable": "The student is not flagged, but it is worth re-checking when the next term's grades are released.",
    "High-Performing": "The student may be a good candidate for the enrichment opportunities listed in the recommendations.",
}


def narrative_paragraph(student_id, label, confidence, probabilities, top_features, include_intro=True):
    """
    Turns the ranked factor list into ONE plain paragraph a teacher can read
    aloud: the class and confidence, the biggest influence and what it means,
    the other factors (pulling toward a weaker or stronger result), and a
    short next step with the non-binding reminder. No SHAP numbers.
    Returns None if there are no factors.
    """
    if not top_features:
        return None

    first = ""
    if include_intro:
        first = f"AURA classified {student_id} as {label}"
        if confidence:
            first += f" with {confidence:g}% confidence"
        first += "."
    if probabilities:
        others = sorted(((c, p) for c, p in probabilities.items() if c != label),
                        key=lambda kv: kv[1], reverse=True)
        if others and others[0][1] >= 20:
            first = (first + " " if first else "") + (
                f"The model also gave {others[0][0]} a {others[0][1]:g}% chance, so this is not a clear-cut case.")

    main = top_features[0]
    second = f"The biggest influence was {_phrase(main)}. {main['plain']['note']}"

    rest = top_features[1:]
    risk = [f for f in rest if f["plain"]["direction"] == "risk"]
    support = [f for f in rest if f["plain"]["direction"] == "support"]
    clauses = []
    if risk:
        slight = "slightly " if all(f["plain"]["level"] == "Minor" for f in risk) else ""
        clauses.append(f"{_cap(_join([_phrase(f) for f in risk]))} also {slight}pulled toward a weaker result")
    if support:
        slight = "slightly " if all(f["plain"]["level"] == "Minor" for f in support) else ""
        text = f"{_join([_phrase(f) for f in support])} {slight}worked in the student's favor"
        clauses.append(text if not clauses else text)
    third = ""
    if clauses:
        third = (", while ".join(clauses) if len(clauses) == 2 else clauses[0]) + "."
        if len(clauses) == 2 and not risk:
            third = _cap(third)

    closing = _CLOSING.get(label, "")
    caveat = "These are patterns the model relied on, not proven causes, so please weigh them with your own knowledge of the student."
    return " ".join(x for x in (first, second, third, closing, caveat) if x)
