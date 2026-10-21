# Model files

AURA uses ONE model file, exported by `AURA-Final-Model-Training.ipynb`
("Deployable Model" cell), plus an optional report for the Model Card:

| File | Where | Purpose |
|---|---|---|
| `AURA_model_bundle.joblib` | this folder | The selected, fitted pipeline + metadata (class order, held-out test metrics, SHAP background, library versions). A `.pkl` with the same name is also accepted. |
| `AURA_model_report.json` | this folder | Optional. Repeated-CV, rule-baseline and ablation results shown on the Analytics Model Card (export cell at the end of the notebook). |
| `aura_transformers.py` | project root, next to `app.py` | The notebook's custom pipeline classes. The bundle only stores a *reference* to them, so it cannot load without this file. |

**Selected model: XGBoost** (panel revision). In the notebook's repeated
5-fold CV on the training students it is statistically tied with the best
macro-F1 (0.815 vs 0.818 for the RF+XGBoost hybrid), has the same At-Risk
recall (0.77), and is explained exactly by TreeSHAP. On 5-fold x 10 CV over
all 281 students: macro-F1 0.754, At-Risk recall 0.657.

The model uses **three inputs: 1st-semester GWA, year level and program.**
K-Means and total scheduled classes were removed after the panel review (the
ablation showed no significant effect); annual / 2nd-semester GWA are
excluded to avoid label leakage, and the simulated attendance / LMS columns
are excluded because no institutional source exists.

## Replacing the model
1. Re-run the notebook; download `AURA_model_bundle.joblib` (and
   `AURA_model_report.json` from the export cell).
2. Copy them here (and `aura_transformers.py` to the project root if you
   changed it). Delete any old `AURA_model_bundle.pkl`.
3. Click **Reload Model** on the Upload page (or restart the app). Your
   students are re-classified automatically.

Keep `scikit-learn==1.6.1` (requirements.txt) unless you retrain with a newer
version - pickled scikit-learn models are version-sensitive. Pin `xgboost-cpu`
to the version Colab used (`import xgboost; print(xgboost.__version__)`).
