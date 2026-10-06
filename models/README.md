# Model files

AURA uses ONE model file, exported by `Dataset_Training.ipynb`
("Deployable Model" + "Export .pkl files" cells):

| File | Where | Purpose |
|---|---|---|
| `AURA_model_bundle.pkl` | this folder | The selected, fitted pipeline + metadata (class order, test metrics, SHAP background, library versions) |
| `aura_transformers.py` | project root, next to `app.py` | The notebook's custom pipeline classes. The pickle only stores a *reference* to them, so the bundle cannot load without this file. |

**Selected model: KMeans + Random Forest.** In the notebook's repeated 5-fold
CV it is statistically tied with the top macro-F1 (0.833 vs 0.834 for the
RF+XGBoost hybrid), has the highest At-Risk recall among the tied models
(0.88), and is explained exactly by SHAP's TreeExplainer.

The pipeline is self-contained: it takes raw rows of the AURA dataset and does
feature building, imputation, one-hot encoding, K-Means cluster assignment and
classification itself - there are no separate scaler/encoder files anymore.
The features it actually uses are year level, 1st-semester GWA, total classes,
program and the K-Means group (simulated attendance/LMS columns and the
label-leaking annual / 2nd-semester GWA are excluded, as in the notebook's
final early-warning run).

The other files in `AURA_pkl.zip` (Random_Forest.pkl, XGBoost.pkl, ...) are the
comparison models for the manuscript; the app does not need them.

## Replacing the model
1. Re-run the notebook and export again.
2. Copy the new `AURA_model_bundle.pkl` here (and `aura_transformers.py` to the
   project root if you changed it).
3. Click **Reload Model** on the Upload page (or restart the app). Your
   students are re-classified automatically.

Keep `scikit-learn==1.6.1` (requirements.txt) unless you retrain with a newer
version - pickled scikit-learn models are version-sensitive.
