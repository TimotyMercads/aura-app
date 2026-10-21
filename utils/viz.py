"""
viz.py
------
Builds the configuration for every switchable visualization panel in AURA
(rendered by templates/_viz.html + static/js/aura-viz.js).

Each panel lets the viewer choose HOW to see the same numbers:
    Chart       - pick a chart type (bar, horizontal bar, doughnut, pie, line,
                  radar, stacked ...), limited to the types that make sense
                  for that data
    Statistics  - a table of the exact values (+ share of total, totals,
                  highest / lowest / average)
    Both        - chart and table together

Colors (validated for the dark navy panel with the dataviz palette checker:
lightness band, chroma, colorblind separation and 3:1 contrast all pass):
    classes  At-Risk #C9567A · Stable #5B8CC4 · High-Performing #4FA374
    metrics  #C98500 · #9085E9 · #D95926 · #199E70   (fixed order, never cycled)
    single series  #C98500 (amber)
A class always keeps its color, whatever chart type or filter is chosen.
"""

import numpy as np
import pandas as pd

CLASS_ORDER = ["At-Risk", "Stable", "High-Performing"]
CLASS_COLORS = {"At-Risk": "#C9567A", "Stable": "#5B8CC4", "High-Performing": "#4FA374"}
METRIC_COLORS = ["#C98500", "#9085E9", "#D95926", "#199E70"]
SINGLE = "#C98500"
TOWARD, AWAY = "#C98500", "#9085E9"     # SHAP: pushes toward / away from the predicted class

# chart-type sets that make sense per data shape
PART_TO_WHOLE = ["bar", "hbar", "doughnut", "pie", "polarArea"]
CATEGORY_VALUES = ["bar", "hbar", "line", "radar"]
MULTI_SERIES = ["stacked", "grouped", "hstacked", "line", "radar"]
SCORES = ["bar", "hbar", "radar", "line"]


def _clean(values, decimals=None):
    out = []
    for v in values:
        try:
            f = float(v)
            if np.isnan(f):
                out.append(None)
                continue
            out.append(round(f, decimals) if decimals is not None else f)
        except (TypeError, ValueError):
            out.append(None)
    return out


def panel(pid, labels, series, types, default=None, *, title=None, unit="", decimals=0,
          percent=False, total_label="Total", value_label="Value", max_value=None, min_value=None,
          note=None, default_view="chart", stats_summary=True, signed=False, extra_stats=None,
          height="sm"):
    """
    pid       unique id (also remembers the viewer's choice in their browser)
    labels    category labels (x axis / slices)
    series    [{"name", "data", "color" | "colors"}]  - "colors" = one per point
    types     allowed chart types, first = default unless `default` is given
    percent   show "% of total" in Statistics (part-to-whole data)
    signed    values can be negative (diverging bars, no pie)
    extra_stats  [(label, value_text), ...] shown above the table
    """
    labels = [str(l) for l in labels]
    clean_series = []
    for s in series:
        d = {"name": s.get("name", ""), "data": _clean(s["data"], decimals if decimals else None)}
        if s.get("colors"):
            d["colors"] = s["colors"]
        d["color"] = s.get("color", SINGLE)
        if s.get("errors"):
            d["errors"] = _clean(s["errors"], 3)
        clean_series.append(d)
    if signed or len(clean_series) > 1:
        types = [t for t in types if t not in ("pie", "doughnut", "polarArea")] or ["bar"]
    if len(labels) < 3:
        types = [t for t in types if t != "radar"] or ["bar"]
    return {
        "id": pid, "title": title, "labels": labels, "series": clean_series,
        "types": types, "default": default if default in types else types[0],
        "unit": unit, "decimals": decimals, "percent": percent, "totalLabel": total_label,
        "valueLabel": value_label, "max": max_value, "min": min_value, "note": note,
        "defaultView": default_view, "summary": stats_summary, "signed": signed,
        "extra": [[a, b] for a, b in (extra_stats or [])], "height": height,
    }


# --------------------------------------------------------------------------
# Class distributions
# --------------------------------------------------------------------------

def class_distribution(pid, counts, title="Students per class", default="bar"):
    counts = counts or {}
    labels = [c for c in CLASS_ORDER]
    return panel(pid, labels, [{"name": "Students", "data": [counts.get(c, 0) for c in labels],
                                "colors": [CLASS_COLORS[c] for c in labels]}],
                 PART_TO_WHOLE, default, title=title, unit=" students", percent=True,
                 value_label="Students")


def class_crosstab(pid, cross, label_key, title, default="stacked"):
    """cross = {"At-Risk": [...], "Stable": [...], "High-Performing": [...], label_key: [...]}"""
    if not cross or not cross.get(label_key):
        return None
    series = [{"name": c, "data": cross.get(c, []), "color": CLASS_COLORS[c]} for c in CLASS_ORDER]
    return panel(pid, cross[label_key], series, MULTI_SERIES, default, title=title,
                 unit=" students", percent=True, value_label="Students")


# --------------------------------------------------------------------------
# GWA
# --------------------------------------------------------------------------

def gwa_by_group(pid, mapping, title):
    if not mapping:
        return None
    return panel(pid, list(mapping.keys()), [{"name": "Average 1st-Sem GWA", "data": list(mapping.values())}],
                 CATEGORY_VALUES, "bar", title=title, decimals=2, min_value=75, max_value=100,
                 value_label="Average GWA", stats_summary=True)


GWA_BINS = [0, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 101]


def gwa_histogram(pid, df, status_col="status"):
    """How many students fall in each 1st-semester GWA band, split by AI class."""
    if df is None or "first_sem_gwa" not in df.columns:
        return None
    g = pd.to_numeric(df["first_sem_gwa"], errors="coerce")
    if g.notna().sum() == 0:
        return None
    labels = []
    for lo, hi in zip(GWA_BINS[:-1], GWA_BINS[1:]):
        labels.append(f"< {hi}" if lo == 0 else (f"≥ {lo}" if hi == 101 else f"{lo}–{hi}"))
    bins = pd.cut(g, GWA_BINS, right=False, labels=labels)
    series = []
    for c in CLASS_ORDER:
        mask = df[status_col] == c if status_col in df.columns else pd.Series(False, index=df.index)
        counts = bins[mask].value_counts().reindex(labels, fill_value=0)
        series.append({"name": c, "data": counts.tolist(), "color": CLASS_COLORS[c]})
    return panel(pid, labels, series, ["stacked", "grouped", "line", "hstacked"], "stacked",
                 title="1st-semester GWA distribution", unit=" students", percent=True,
                 value_label="Students",
                 note="Bands of the 1st-semester GWA; the study's cut-offs are 82 (At-Risk) and 90 (High-Performing).")


def probability_histogram(pid, df):
    """Spread of the model's At-Risk probability across students."""
    if df is None or "prob_at_risk" not in df.columns:
        return None
    p = pd.to_numeric(df["prob_at_risk"], errors="coerce").dropna()
    if p.empty:
        return None
    edges = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100.001]
    labels = [f"{a}–{int(round(b))}%" for a, b in zip(edges[:-1], edges[1:])]
    counts = pd.cut(p, edges, right=False, labels=labels).value_counts().reindex(labels, fill_value=0)
    return panel(pid, labels, [{"name": "Students", "data": counts.tolist()}],
                 ["bar", "hbar", "line"], "bar", title="At-Risk probability across your students",
                 unit=" students", percent=True, value_label="Students",
                 note="Most students sit near 0%; anyone in the upper bands is worth a closer look.")


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def feature_importance(pid, top_features, title="Top predictors"):
    """top_features = [(friendly_name, mean|SHAP|), ...]"""
    if not top_features:
        return None
    return panel(pid, [n for n, _ in top_features], [{"name": "Mean |SHAP|", "data": [v for _, v in top_features]}],
                 ["hbar", "bar", "doughnut", "pie", "polarArea"], "hbar", title=title, decimals=4,
                 percent=True, value_label="Mean |SHAP|",
                 note="Share = this factor's part of the total influence across your students.")


TEST_KEYS = ["Accuracy", "F1 (macro)", "MCC", "ROC-AUC (OvR macro)", "Recall (macro)",
             "Specificity (macro)", "At-Risk Recall", "At-Risk Precision"]


def test_metrics(pid, metrics, n_at_risk=None):
    if not metrics:
        return None
    keys = [k for k in TEST_KEYS if metrics.get(k) is not None]
    return panel(pid, keys, [{"name": "Held-out test", "data": [metrics[k] for k in keys]}],
                 SCORES, "bar", title="Held-out test set", decimals=3, min_value=0, max_value=1,
                 value_label="Score", stats_summary=False,
                 note=(f"Only {n_at_risk} At-Risk students were in the test set, so At-Risk recall/precision "
                       f"move in big steps." if n_at_risk else None))


def cv_metrics(pid, cv):
    if not cv:
        return None
    keys = [k for k in ["F1 (macro)", "MCC", "Accuracy", "At-Risk Recall", "At-Risk Precision"] if cv.get(k)]
    return panel(pid, keys, [{"name": "Mean of 50 folds", "data": [cv[k][0] for k in keys],
                              "errors": [cv[k][1] for k in keys]}],
                 SCORES, "bar", title="Repeated cross-validation", decimals=3, min_value=0, max_value=1,
                 value_label="Mean", stats_summary=False,
                 note="Whiskers / ± = standard deviation across the 5 folds × 10 repeats.")


COMPARE_KEYS = ["F1 (macro)", "At-Risk Recall", "At-Risk Precision", "Accuracy"]


def model_comparison(pid, repeated_cv, selected=None, rule_name=None):
    if not repeated_cv:
        return None
    names = sorted(repeated_cv, key=lambda n: repeated_cv[n]["F1 (macro)"][0], reverse=True)
    labels = []
    for n in names:
        tag = " ★ selected" if n == selected else (" (baseline)" if n == rule_name else "")
        labels.append(n.replace("Rule: 1st-sem GWA 82/90", "Rule 82/90") + tag)
    series = [{"name": k, "data": [repeated_cv[n][k][0] for n in names], "color": METRIC_COLORS[i]}
              for i, k in enumerate(COMPARE_KEYS)]
    return panel(pid, labels, series, ["grouped", "hgrouped", "radar", "line"], "grouped",
                 title="Models compared (repeated CV)", decimals=3, min_value=0, max_value=1,
                 value_label="Score", stats_summary=False, height="md")


# --------------------------------------------------------------------------
# One student
# --------------------------------------------------------------------------

def class_probabilities(pid, probabilities):
    if not probabilities:
        return None
    labels = [c for c in CLASS_ORDER if c in probabilities]
    return panel(pid, labels, [{"name": "Probability", "data": [probabilities[c] for c in labels],
                                "colors": [CLASS_COLORS[c] for c in labels]}],
                 ["hbar", "bar", "doughnut", "pie"], "hbar", title="Class probabilities", unit="%",
                 decimals=1, max_value=100, min_value=0, value_label="Probability", stats_summary=False,
                 height="xs")


def factor_shares(pid, explanation):
    """Share of this student's prediction explained by each input (|SHAP| for the predicted class)."""
    if not explanation or not explanation.get("technical"):
        return None
    rows = explanation["technical"]
    total = sum(abs(r["shap"]) for r in rows) or 1
    return panel(pid, [r["label"] for r in rows],
                 [{"name": "Share of influence", "data": [abs(r["shap"]) / total * 100 for r in rows]}],
                 ["hbar", "bar", "doughnut", "pie", "polarArea"], "hbar",
                 title="How much each input influenced this prediction", unit="%", decimals=1,
                 value_label="Share of influence", stats_summary=False, height="xs")


def signed_contributions(pid, explanation):
    """Signed SHAP per factor: + pushes toward the predicted class, − pushes away."""
    if not explanation or not explanation.get("technical"):
        return None
    rows = explanation["technical"]
    label = explanation.get("predicted_label", "the class")
    return panel(pid, [r["label"] for r in rows],
                 [{"name": f"SHAP toward {label}", "data": [r["shap"] for r in rows],
                   "colors": [TOWARD if r["shap"] >= 0 else AWAY for r in rows]}],
                 ["hbar", "bar"], "hbar", title=f"Contribution toward {label}", decimals=3, signed=True,
                 value_label="SHAP", stats_summary=False, height="xs",
                 note=f"Amber = pushes toward {label}; violet = pushes away from it.")


# --------------------------------------------------------------------------
# Upload cleaning
# --------------------------------------------------------------------------

def cleaning_flow(pid, report):
    """Where the uploaded rows went: kept vs each removal reason."""
    if not report or not report.get("original_rows"):
        return None
    reasons = [
        ("Kept (classified)", report.get("final_rows", 0)),
        ("Blank rows", report.get("empty_rows_removed", 0)),
        ("Exact duplicates", report.get("duplicate_rows_removed", 0)),
        ("Duplicate Student IDs", report.get("duplicate_ids_removed", 0)),
        ("Missing a model input", report.get("excluded_missing_critical", 0)),
    ]
    reasons = [(k, v) for k, v in reasons if v]
    colors = ["#4FA374"] + [SINGLE] * (len(reasons) - 1)
    return panel(pid, [k for k, _ in reasons], [{"name": "Rows", "data": [v for _, v in reasons], "colors": colors}],
                 ["hbar", "bar", "doughnut", "pie"], "hbar", title="Where the uploaded rows went",
                 unit=" rows", percent=True, value_label="Rows", height="xs",
                 extra_stats=[("Rows in the file", f"{report.get('original_rows', 0):,}"),
                              ("Rows kept", f"{report.get('final_rows', 0):,}")])
