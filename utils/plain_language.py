"""
plain_language.py
------------------
Turns a technical SHAP explanation into something a teacher can read
aloud to a parent who has never heard of "SHAP," "confidence scores," or
machine learning at all (manuscript 3.2.1: outputs use "clear, plain, and
non-technical language").

Three things happen here that the raw SHAP output does NOT give you:
  1. Every factor gets a plain English name and a natural way to state its
     value (a GWA, "Year 2", "BSIT").
  2. Every factor gets a short sentence explaining what a weak/strong value
     generally means for a student.
  3. The SHAP magnitude (a decimal like 0.0523) becomes a 3-level rating -
     Major / Moderate / Minor - shown as a bar, not a number.

Nothing here changes the prediction or the SHAP math (that's in
model_service.py) - this only changes how it's DESCRIBED.

PANEL REVISION: the final model (XGBoost) uses three factors only -
1st-semester GWA, year level and program. The one-hot program columns
(program_BSCS / program_BSIT / program_BMMA) are shown as ONE factor,
"Program". Descriptions of the older K-Means / course-load factors are kept
only so an older model bundle would still display sensibly.
"""


def _gwa(v):
    return f"{float(v):.2f}"


def _int(unit):
    return lambda v: f"{float(v):,.0f} {unit}"


# plain_name, group (for the Key Factors bars), value format, and two short
# phrases - one for when the factor works AGAINST a strong outcome
# (risk_note) and one for when it works FOR one (support_note).
FEATURE_INFO = {
    # ---- the three inputs of the final model ----
    "first_sem_gwa": {
        "plain_name": "1st-Semester GWA",
        "group": "1st-Semester GWA",
        "format": _gwa,
        "risk_note": "A lower GWA in the first semester is the clearest early sign that the student may struggle for the rest of the year.",
        "support_note": "A strong first-semester GWA shows a solid academic foundation going into the rest of the year.",
    },
    "year_level": {
        "plain_name": "Year Level",
        "group": "Year Level",
        "format": lambda v: f"Year {float(v):.0f}",
        "risk_note": "Students at this year level showed a slightly higher chance of difficulty in this dataset. This describes a group pattern, not the student personally.",
        "support_note": "Students at this year level generally did well in this dataset. This describes a group pattern, not the student personally.",
    },
    "program": {
        "plain_name": "Program",
        "group": "Program",
        "format": lambda v: str(v),
        "risk_note": "The program's grading and workload pattern leaned slightly toward a weaker outcome in this dataset. This describes a group pattern, not the student personally.",
        "support_note": "The program's grading and workload pattern leaned slightly toward a stronger outcome in this dataset. This describes a group pattern, not the student personally.",
    },
    # ---- only used if an OLDER model bundle is loaded ----
    "total_classes": {
        "plain_name": "Scheduled Class Meetings (Course Load)",
        "group": "Course Load",
        "format": _int("class meetings"),
        "risk_note": "The student's course load is linked to a weaker outcome in this dataset.",
        "support_note": "The student's course load is in a range associated with stronger outcomes in this dataset.",
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
    "year_level": "being in {v}",
    "program": "being in the {v} program",
    "total_classes": "a course load of {v}",
    "student_group": "being placed in {v} by the clustering step",
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
