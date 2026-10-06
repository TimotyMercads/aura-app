import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ID_COLS = ["student_id"]
CONTEXT_COLS = ["academic_year", "semester"]
DATE_COLS = ["first_activity_date", "last_activity_date"]
CATEGORICAL_COLS = ["program"]


def _num(s):
    return pd.to_numeric(s, errors="coerce")


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Raw schema rows -> model-ready feature table (no fitting needed)."""

    def __init__(self, drop_cols=()):
        self.drop_cols = drop_cols

    def fit(self, X, y=None):
        self.is_fitted_ = True      # stateless, but lets sklearn treat pipeline slices as fitted
        return self

    def transform(self, X):
        X = X.copy()
        drop = set(self.drop_cols)
        for c in DATE_COLS:
            if c in X.columns:
                X[c] = pd.to_datetime(X[c], errors="coerce")

        def ok(*cols):
            return all(c in X.columns and c not in drop for c in cols)

        new = {}
        if ok("first_sem_gwa", "second_sem_gwa"):      # 2nd-semester GWA minus 1st-semester GWA = grade trend
            new["sem_gwa_change"] = _num(X["second_sem_gwa"]) - _num(X["first_sem_gwa"])
        if ok("on_time_submissions", "late_submissions"):
            on_time, late = _num(X["on_time_submissions"]), _num(X["late_submissions"])
            total = on_time + late
            new["on_time_ratio"] = on_time / total.where(total > 0)
        if ok("lms_activity_count", "lms_login_count"):
            logins = _num(X["lms_login_count"])
            new["activity_per_login"] = _num(X["lms_activity_count"]) / logins.where(logins > 0)
        if ok(*DATE_COLS):
            new["lms_active_days"] = (X[DATE_COLS[1]] - X[DATE_COLS[0]]).dt.days

        X = X.drop(columns=[c for c in ID_COLS + CONTEXT_COLS + DATE_COLS + list(drop) if c in X.columns])
        for c in X.columns:
            X[c] = X[c].astype(object) if c in CATEGORICAL_COLS else _num(X[c])
        for k, v in new.items():
            X[k] = v
        return X


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drops numeric features that are (almost) copies of another feature. Fitted on training data only.
    Columns listed in `protect` are never dropped (and are considered first)."""

    def __init__(self, threshold=0.95, protect=()):
        self.threshold = threshold
        self.protect = protect

    def fit(self, X, y=None):
        num = X.select_dtypes(include=[np.number])
        corr = num.corr().abs()
        order = [c for c in num.columns if c in self.protect] + [c for c in num.columns if c not in self.protect]
        kept, dropped = [], {}
        for c in order:
            if c in self.protect:
                kept.append(c)
                continue
            partner = next((k for k in kept if corr.loc[c, k] > self.threshold), None)
            if partner is None:
                kept.append(c)
            else:
                dropped[c] = partner
        self.dropped_ = dropped
        return self

    def transform(self, X):
        return X.drop(columns=[c for c in self.dropped_ if c in X.columns])


def select_categorical(df):
    return [c for c in df.columns if c in CATEGORICAL_COLS]


def select_numeric(df):
    return [c for c in df.columns if c not in CATEGORICAL_COLS]


def make_preprocessor(scale=False):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    ct = ColumnTransformer([("num", Pipeline(num_steps), select_numeric),
                            ("cat", cat, select_categorical)],
                           verbose_feature_names_out=False)
    ct.set_output(transform="pandas")
    return ct


class NameCleaner(BaseEstimator, TransformerMixin):
    """XGBoost rejects feature names containing [ ] < > ; make every name safe."""

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = X.copy()
        X.columns = [str(c).replace("[", "(").replace("]", ")").replace("<", "lt").replace(">", "gt")
                     for c in X.columns]
        return X


class ClusterFeatureAdder(BaseEstimator, TransformerMixin):
    """Fits K-Means on training data and appends one-hot cluster membership (3.3.2, Step 1)."""

    def __init__(self, n_clusters=4, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y=None):
        self.scaler_ = StandardScaler().fit(X)
        self.km_ = KMeans(n_clusters=self.n_clusters, n_init=10,
                          random_state=self.random_state).fit(self.scaler_.transform(X))
        return self

    def transform(self, X):
        labels = self.km_.predict(self.scaler_.transform(X))
        out = X.copy()
        for k in range(self.n_clusters):
            out[f"cluster_{k}"] = (labels == k).astype(int)
        return out
