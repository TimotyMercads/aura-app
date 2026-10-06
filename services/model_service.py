"""
model_service.py
-----------------
Loads the deployable AURA model produced by the Dataset_Training notebook
("Deployable Model" / "Export .pkl files" cells):

    models/AURA_model_bundle.pkl   -> dict with the fitted pipeline + metadata
    aura_transformers.py           -> the notebook's custom transformer classes
                                      (FeatureBuilder, CorrelationFilter,
                                      ClusterFeatureAdder, ...). The pickle only
                                      stores a REFERENCE to these classes, so the
                                      file must be importable (it sits next to
                                      app.py).

Selected model (chosen by the notebook, manuscript 3.3.2 Step 4):
    KMeans + Random Forest - statistically tied with the best repeated-CV
    macro-F1, the highest At-Risk recall among the tied models, and explained
    EXACTLY by SHAP's TreeExplainer.

The bundle's pipeline is self-contained: it takes RAW dataset rows (the same
columns as the AURA dataset sheet) and does all feature building, imputation,
one-hot encoding, K-Means cluster assignment and classification itself. So
there are no separate scaler / encoder files anymore.

Everything here degrades gracefully: if the bundle is missing or cannot be
loaded, is_ready() is False and the pages fall back to a "demo mode" banner.
"""

import os
import sys
import pickle
import hashlib
import warnings

import numpy as np
import pandas as pd

from utils import plain_language

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# The pickle references the module "aura_transformers" by name - make sure the
# project root (where aura_transformers.py lives) is importable no matter how
# the app was started (python app.py, gunicorn, flask run, ...).
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

BUNDLE_FILE = "AURA_model_bundle.pkl"
TRANSFORMERS_FILE = "aura_transformers.py"

CLASS_ORDER_DEFAULT = ["At-Risk", "Stable", "High-Performing"]
AT_RISK = "At-Risk"
HIGH_PERFORMING = "High-Performing"

_cache = {}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def _bundle_path():
    return os.path.join(MODEL_DIR, BUNDLE_FILE)


def _transformers_path():
    return os.path.join(BASE_DIR, TRANSFORMERS_FILE)


def _sklearn_compat_patch(obj, _seen=None):
    """
    The bundle was trained with scikit-learn 1.6.1 (see bundle['versions']).
    requirements.txt pins that version, but if a different one is installed a
    few private attributes can be missing on unpickled SimpleImputers. This
    walks the pipeline and fills them in so predictions still work. It never
    changes fitted values - only adds attributes newer versions expect.
    """
    from sklearn.impute import SimpleImputer
    _seen = _seen if _seen is not None else set()
    if id(obj) in _seen:
        return
    _seen.add(id(obj))
    if isinstance(obj, SimpleImputer) and not hasattr(obj, "_fill_dtype"):
        obj._fill_dtype = getattr(obj, "_fit_dtype", None)
    for value in list(getattr(obj, "__dict__", {}).values()):
        if isinstance(value, (list, tuple)):
            for item in value:
                for sub in (item if isinstance(item, tuple) else (item,)):
                    if hasattr(sub, "get_params"):
                        _sklearn_compat_patch(sub, _seen)
        elif hasattr(value, "get_params"):
            _sklearn_compat_patch(value, _seen)


def _load_bundle():
    if "bundle" in _cache:
        return _cache["bundle"]
    path = _bundle_path()
    if not os.path.exists(path):
        _cache["bundle"] = None
        _cache["load_error"] = None
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # sklearn InconsistentVersionWarning
            try:
                import joblib
                bundle = joblib.load(path)
            except Exception:
                with open(path, "rb") as fh:
                    bundle = pickle.load(fh)
        _sklearn_compat_patch(bundle["pipeline"])
        _cache["bundle"] = bundle
        _cache["load_error"] = None
    except Exception as exc:
        # Most common causes: aura_transformers.py missing, or an incompatible
        # library version. Treat as "not ready" rather than crashing every page.
        _cache["bundle"] = None
        _cache["load_error"] = f"{type(exc).__name__}: {exc}"
    return _cache["bundle"]


def reload_all():
    """Clears the in-memory cache so a freshly copied bundle gets picked up."""
    _cache.clear()


def is_ready() -> bool:
    return _load_bundle() is not None


def load_error():
    _load_bundle()
    return _cache.get("load_error")


def shap_available() -> bool:
    """
    True if the shap library is installed. Uses find_spec so it does NOT import
    shap (which pulls in numba/llvmlite and ~150+ MB of RAM) on every page load -
    the real import happens only when an explanation is first generated.
    Set the env var AURA_DISABLE_SHAP=1 to force the lighter tree-path fallback
    (useful on small hosts, e.g. a 512 MB free tier, if the app runs out of memory).
    """
    if os.environ.get("AURA_DISABLE_SHAP", "").strip().lower() in ("1", "true", "yes"):
        return False
    import importlib.util
    try:
        return importlib.util.find_spec("shap") is not None
    except Exception:
        return False


def file_status():
    """Presence of the files the app needs, for the Upload page."""
    return {
        "bundle": os.path.exists(_bundle_path()),
        "transformers": os.path.exists(_transformers_path()),
        "shap": shap_available(),
    }


def _pipeline():
    b = _load_bundle()
    return b["pipeline"] if b else None


def class_order():
    b = _load_bundle()
    return list(b.get("class_order", CLASS_ORDER_DEFAULT)) if b else list(CLASS_ORDER_DEFAULT)


def raw_input_columns():
    b = _load_bundle()
    return list(b.get("raw_input_columns", [])) if b else []


def model_features():
    """Names of the features the classifier actually sees (after the pipeline's
    own feature building / filtering / one-hot / clustering)."""
    if "features" in _cache:
        return _cache["features"]
    pipe = _pipeline()
    feats = []
    if pipe is not None:
        clf = pipe[-1]
        names = getattr(clf, "feature_names_in_", None)
        feats = list(names) if names is not None else []
    _cache["features"] = feats
    return feats


def required_input_columns():
    """Raw columns that feed the model's features (what an upload must contain
    for a meaningful prediction). Derived from the fitted pipeline itself."""
    feats = model_features()
    raw = set()
    for f in feats:
        if f.startswith("cluster_"):
            continue
        if f.startswith("program_"):
            raw.add("program")
        else:
            raw.add(f)
    order = ["program", "year_level", "first_sem_gwa", "total_classes"]
    return [c for c in order if c in raw] + sorted(raw - set(order))


def model_info():
    """Model card shown on the Analytics / Upload pages."""
    b = _load_bundle()
    if not b:
        return None
    import sklearn
    trained_with = b.get("versions", {}) or {}
    return {
        "selected_model": b.get("selected_model", "Unknown"),
        "explained_with": b.get("explained_with", b.get("selected_model")),
        "class_order": class_order(),
        "test_metrics": b.get("test_metrics", {}),
        "at_risk_threshold": b.get("at_risk_threshold"),
        "features": model_features(),
        "friendly_features": [plain_language.friendly_name(f) for f in model_features()],
        "required_columns": required_input_columns(),
        "trained_sklearn": trained_with.get("sklearn"),
        "installed_sklearn": sklearn.__version__,
        "version_mismatch": bool(trained_with.get("sklearn")) and trained_with.get("sklearn") != sklearn.__version__,
        "explainer": "SHAP TreeExplainer (exact)" if shap_available() else "Tree path attribution (SHAP library not installed)",
    }


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------

def _to_raw_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rows in exactly the raw schema the pipeline was fitted on."""
    cols = raw_input_columns()
    X = df.reindex(columns=cols).copy()
    if "program" in X.columns:
        X["program"] = X["program"].astype(object).where(X["program"].notna(), None)
    return X


def _labels_from_proba(proba):
    order = class_order()
    b = _load_bundle()
    threshold = b.get("at_risk_threshold") if b else None
    idx = proba.argmax(axis=1)
    if threshold is not None and AT_RISK in order:
        ar = order.index(AT_RISK)
        idx = np.where(proba[:, ar] >= float(threshold), ar, idx)
    return [order[i] for i in idx], idx


def predict_proba_frame(df: pd.DataFrame):
    """Returns an (n, n_classes) probability array, or None if not ready."""
    pipe = _pipeline()
    if pipe is None or df is None or len(df) == 0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pipe.predict_proba(_to_raw_frame(df))


def predict_batch(df: pd.DataFrame):
    """
    Predicts every row of df in ONE vectorized pass.
    Returns {"labels": [...], "confidences": [...], "prob_at_risk": [...]}
    (same order/length as df); all-None lists if the model isn't ready.
    """
    n = 0 if df is None else len(df)
    empty = {"labels": [None] * n, "confidences": [None] * n, "prob_at_risk": [None] * n}
    if n == 0 or not is_ready():
        return empty
    try:
        proba = predict_proba_frame(df)
        labels, idx = _labels_from_proba(proba)
        order = class_order()
        ar = order.index(AT_RISK) if AT_RISK in order else 0
        return {
            "labels": labels,
            "confidences": [round(float(proba[i, k]) * 100, 1) for i, k in enumerate(idx)],
            "prob_at_risk": [round(float(p) * 100, 1) for p in proba[:, ar]],
        }
    except Exception:
        return empty


def predict_student(row: pd.Series):
    """
    Single-student prediction for the Student Insights page.
    Returns {label, confidence, probabilities{class: pct}, model_used}
    or {error: ...}; None if the model isn't loaded.
    """
    if not is_ready():
        return None
    try:
        proba = predict_proba_frame(pd.DataFrame([row]))
        labels, idx = _labels_from_proba(proba)
        order = class_order()
        return {
            "label": labels[0],
            "confidence": round(float(proba[0, idx[0]]) * 100, 1),
            "probabilities": {c: round(float(p) * 100, 1) for c, p in zip(order, proba[0])},
            "model_used": _load_bundle().get("selected_model", "AURA model"),
        }
    except Exception as exc:
        return {"error": str(exc)}


# --------------------------------------------------------------------------
# Explainability (SHAP) - manuscript 3.3.2 Step 5 / 3.3.4
# --------------------------------------------------------------------------

def _transform(df: pd.DataFrame) -> pd.DataFrame:
    """Raw rows -> the exact feature table the classifier sees."""
    pipe = _pipeline()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pipe[:-1].transform(_to_raw_frame(df))


def _to_3d(sv, n, f, c):
    arr = np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)
    if arr.shape == (n, f, c):
        return arr
    if arr.shape == (c, n, f):
        return np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2 and arr.shape == (n, f) and c == 1:
        return arr[:, :, None]
    raise ValueError(f"Unexpected SHAP output shape {arr.shape}")


def _tree_path_attribution(clf, X):
    """
    Fallback used ONLY when the shap library isn't installed: decomposes each
    tree's probability along the decision path (Saabas method), averaged over
    the forest. Contributions + base value add up exactly to predict_proba, so
    the ranking is faithful to the model, but it is an approximation of SHAP.
    """
    Xv = X.to_numpy(dtype=np.float32)
    n, f = Xv.shape
    n_classes = len(clf.classes_)
    contrib = np.zeros((n, f, n_classes))
    base = np.zeros(n_classes)
    estimators = getattr(clf, "estimators_", [clf])
    for est in estimators:
        tree = est.tree_
        val = tree.value[:, 0, :].astype(float)
        val = val / np.clip(val.sum(axis=1, keepdims=True), 1e-12, None)
        parent = np.full(tree.node_count, -1)
        for node in range(tree.node_count):
            for child in (tree.children_left[node], tree.children_right[node]):
                if child != -1:
                    parent[child] = node
        path = est.decision_path(Xv)
        rows = np.repeat(np.arange(n), np.diff(path.indptr))
        nodes = path.indices
        keep = nodes != 0
        rows, nodes = rows[keep], nodes[keep]
        par = parent[nodes]
        np.add.at(contrib, (rows, tree.feature[par]), val[nodes] - val[par])
        base += val[0]
    k = len(estimators)
    return contrib / k, base / k


def _shap_3d(Xt: pd.DataFrame):
    """(n, features, classes) attributions + expected values + method name."""
    clf = _pipeline()[-1]
    n, f, c = len(Xt), Xt.shape[1], len(class_order())
    if shap_available():
        try:
            import shap
            explainer = _cache.get("explainer")
            if explainer is None:
                explainer = shap.TreeExplainer(clf)
                _cache["explainer"] = explainer
            sv = explainer.shap_values(Xt, check_additivity=False)
            ev = np.atleast_1d(np.asarray(explainer.expected_value, dtype=float))
            return _to_3d(sv, n, f, c), ev, "SHAP TreeExplainer"
        except Exception:
            pass
    contrib, base = _tree_path_attribution(clf, Xt)
    return contrib, base, "Tree path attribution"


def _factor_key(feature):
    """Collapses one-hot columns into one human factor (SHAP is additive, so
    summing a one-hot group's values is exact)."""
    if feature.startswith("program_"):
        return "program"
    if feature.startswith("cluster_"):
        return "student_group"
    return feature


def _factor_value(factor, row, xt_row):
    if factor == "program":
        return row.get("program")
    if factor == "student_group":
        hits = [c for c in xt_row.index if c.startswith("cluster_") and xt_row[c] >= 0.5]
        return f"Group {hits[0].split('_', 1)[1]}" if hits else None
    if factor in row.index and pd.notna(row.get(factor)):
        return row.get(factor)
    return xt_row.get(factor)


def explain_student(row: pd.Series, top_n=6):
    """
    Per-student explanation of the selected model.

    Returns:
      {
        "predicted_label": "At-Risk" | "Stable" | "High-Performing",
        "method": "SHAP TreeExplainer" | "Tree path attribution",
        "groups": {"Academic Performance": pct, ...},      # Key Factors bars
        "top_features": [ {feature, label, raw_value, plain{...}}, ... ],
        "technical": [ {feature, label, value, shap, effect}, ... ],
        "base_value": float,
      }
    or None if the model isn't ready / the explanation fails.
    """
    if not is_ready():
        return None
    try:
        single = pd.DataFrame([row])
        Xt = _transform(single)
        sv3, ev, method = _shap_3d(Xt)
    except Exception:
        return None

    order = class_order()
    proba = predict_proba_frame(single)
    labels, idx = _labels_from_proba(proba)
    c = int(idx[0])
    xt_row = Xt.iloc[0]
    features = list(Xt.columns)
    sv = sv3[0]                                  # (features, classes)

    # Key Factors bars: share of |SHAP| for the predicted class, per factor group
    group_scores = {}
    for j, feat in enumerate(features):
        g = plain_language.factor_group(_factor_key(feat))
        group_scores[g] = group_scores.get(g, 0.0) + abs(sv[j, c])
    top_score = max(group_scores.values()) if group_scores else 0
    groups = {
        g: (max(round(s / top_score * 100, 1), 4) if top_score > 0 else 4)
        for g, s in sorted(group_scores.items(), key=lambda kv: kv[1], reverse=True)
    }

    # Plain-language factors. Direction = does the factor push toward a STRONG
    # outcome (High-Performing) or toward RISK (At-Risk)? Using
    # SHAP(High-Performing) - SHAP(At-Risk) keeps the 3-class model readable
    # with one "working for / working against" scale.
    ar = order.index(AT_RISK) if AT_RISK in order else 0
    hp = order.index(HIGH_PERFORMING) if HIGH_PERFORMING in order else len(order) - 1
    outcome, toward_pred = {}, {}
    for j, feat in enumerate(features):
        key = _factor_key(feat)
        outcome[key] = outcome.get(key, 0.0) + float(sv[j, hp] - sv[j, ar])
        toward_pred[key] = toward_pred.get(key, 0.0) + float(sv[j, c])

    ranked = sorted(outcome.items(), key=lambda kv: abs(kv[1]), reverse=True)
    max_abs = abs(ranked[0][1]) if ranked else 0
    top_features = []
    for key, val in ranked[:top_n]:
        direction = "decreases_risk" if val >= 0 else "increases_risk"
        raw_value = _factor_value(key, row, xt_row)
        top_features.append({
            "feature": key,
            "label": plain_language.friendly_name(key),
            "raw_value": raw_value,
            "plain": plain_language.describe_feature(key, raw_value, direction, val, max_abs),
        })

    technical = []
    for key, val in sorted(toward_pred.items(), key=lambda kv: abs(kv[1]), reverse=True):
        technical.append({
            "feature": key,
            "label": plain_language.friendly_name(key),
            "value": plain_language.format_value(key, _factor_value(key, row, xt_row)),
            "shap": round(val, 4),
            "effect": f"pushes toward {order[c]}" if val >= 0 else f"pushes away from {order[c]}",
        })

    return {
        "predicted_label": labels[0],
        "method": method,
        "groups": groups,
        "top_features": top_features,
        "technical": technical,
        "base_value": round(float(ev[c] if len(ev) > c else ev[0]), 4),
    }


def global_importance(df: pd.DataFrame = None, top_n=10, sample=200):
    """
    Global importance for the Analytics page: mean |SHAP| over the teacher's
    students (all classes, one-hot groups combined) - the same measure as the
    notebook's global SHAP bar chart. Falls back to the forest's own
    impurity-based importance if no data is given.
    Returns [(friendly_name, value), ...] or None.
    """
    if not is_ready():
        return None
    try:
        if df is not None and len(df):
            data = df.sample(min(sample, len(df)), random_state=42) if len(df) > sample else df
            Xt = _transform(data)
            key = hashlib.md5(pd.util.hash_pandas_object(Xt, index=False).values.tobytes()).hexdigest()
            cached = _cache.get("global_importance")
            if cached and cached[0] == key:
                return cached[1][:top_n]
            sv3, _, _ = _shap_3d(Xt)
            per_feature = np.abs(sv3).mean(axis=(0, 2))
            names = list(Xt.columns)
        else:
            key = None
            clf = _pipeline()[-1]
            per_feature = clf.feature_importances_
            names = model_features()
        agg = {}
        for name, v in zip(names, per_feature):
            k = _factor_key(name)
            agg[k] = agg.get(k, 0.0) + float(v)
        result = [(plain_language.friendly_name(k), round(v, 4))
                  for k, v in sorted(agg.items(), key=lambda kv: kv[1], reverse=True)]
        if key:
            _cache["global_importance"] = (key, result)
        return result[:top_n]
    except Exception:
        return None
