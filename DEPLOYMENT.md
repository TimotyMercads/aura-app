# Deploying AURA to the Cloud (Vercel + a cloud database)

This app now works with **either** a local XAMPP MySQL (see `XAMPP_SETUP.md`)
**or** a cloud-hosted MySQL/Postgres database — the same codebase, switched
by one environment variable. This guide covers the cloud path, for
publishing a live version on Vercel.

---

## 1. Pick a cloud database

Any of these work. Postgres options are usually the easiest to get a free
tier on:

| Provider | Type | Notes |
|---|---|---|
| [Neon](https://neon.tech) | Postgres | Generous free tier, very easy setup |
| [Supabase](https://supabase.com) | Postgres | Free tier, includes a nice SQL editor |
| [Railway](https://railway.app) | MySQL or Postgres | Simple, usage-based free credits |
| [PlanetScale](https://planetscale.com) | MySQL | MySQL-compatible, generous free tier |

Whichever you pick, you'll end up with a **connection string** that looks
like one of these:

```
postgresql://user:password@host:5432/dbname
mysql://user:password@host:3306/dbname
```

Copy it somewhere — you'll need it in Step 3.

## 2. Create the tables

Open your database provider's SQL editor (Neon, Supabase, and Railway all
have one built into their dashboard), and run the contents of:

- **`schema_postgres.sql`** if you picked a Postgres provider (Neon, Supabase, Railway Postgres)
- **`schema.sql`** if you picked a MySQL provider (PlanetScale, Railway MySQL)

Copy the whole file's contents, paste into the SQL editor, and run it. You
should see the same 5 tables as the local setup: `users`, `uploads`,
`students`, `intervention_log`, `action_plans`.

## 3. Push the code to GitHub

Vercel deploys from a GitHub (or GitLab/Bitbucket) repository. If this
project isn't already in one:

```bash
cd flaskapp
git init
git add .
git commit -m "AURA initial commit"
```

Then create a new repository on GitHub and push it there (GitHub's own
"create a new repository" page gives you the exact commands for this).

## 4. Deploy — two options

### Option A: Render (recommended for this app)

Render runs your app as a normal, always-on process rather than a
size-limited serverless function, so pandas/scikit-learn/shap being
large is simply never an issue here. This is the path to use if Vercel
gave you a "bundle size exceeds 500 MB" error — that limit doesn't
exist on Render.

1. Go to [render.com](https://render.com) and sign in with GitHub.
2. Click **New → Web Service**, and connect the same GitHub repository
   you pushed in Step 3.
3. Render auto-detects Python. Set:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
4. Under **Environment Variables**, add `DATABASE_URL` (your Neon
   connection string, ending in `?sslmode=require`) and `SECRET_KEY`
   (any long random text). Set the Start Command to
   `gunicorn app:app --workers 1 --threads 2 --timeout 120 --bind 0.0.0.0:$PORT`
   (already in `render.yaml`) — one worker keeps memory inside the
   free plan's 512 MB. If the service restarts with an out-of-memory
   message, add the variable `AURA_DISABLE_SHAP=1` (explanations then use a
   lighter tree-path method).
5. Click **Create Web Service**. First deploy takes a few minutes
   (installing pandas/scikit-learn/shap is heavier than a typical app).
6. Open the `.onrender.com` URL Render gives you — sign up and log in.

A `render.yaml` file is included in this project if you'd rather use
Render's "Blueprint" import instead of filling in the fields by hand —
same result either way.

**Database tip:** Render's own free Postgres expires 30 days after it is
created, so a free Neon database (no expiry) is the safer choice for a
thesis that has to stay online after the defense.

**One tradeoff on Render's free tier**: the service spins down after a
period of no traffic, so the very first request after a quiet stretch
takes 30-60 seconds to wake back up. Not a bug — just how free-tier
always-on hosting works. A paid tier removes this.

### Option B: Vercel

1. Go to [vercel.com](https://vercel.com) and sign in with GitHub.
2. Click **Add New → Project**, and import the repository.
3. Vercel auto-detects this as a Python/Flask project from
   `requirements.txt` and the `app` variable in `app.py` — no build
   configuration needed. **Don't add a `vercel.json` file** unless you
   have a specific reason to — an explicit `functions` block pointing
   at `app.py` actively conflicts with Vercel's zero-config Flask
   detection and will fail the build with a
   `doesn't match any Serverless Functions` error.
4. Add the same two environment variables (`DATABASE_URL`, `SECRET_KEY`)
   under **Environment Variables** before deploying.
5. Click **Deploy**.

**Honest caveat, found by actually testing this**: this app's combined
dependencies (pandas, scikit-learn, shap, plus everything else) came in
at ~845 MB in one real deployment attempt — over Vercel's 500 MB
serverless function limit, even after removing the unused `xgboost`
dependency to shrink it. Whether this fits depends on exactly which
package versions get resolved at deploy time, and Vercel's "Large
Functions" (5 GB) opt-in may or may not be available depending on your
plan. If you hit a bundle-size error, Render (Option A) sidesteps this
entirely, since it isn't a serverless platform.

## 5. Adding your trained model files

`models/AURA_model_bundle.pkl` and `aura_transformers.py` need to be
**committed to the GitHub repository** so Render/Vercel deploys them along
with the code. `.gitignore` already allows `AURA_model_bundle.pkl` (other
`.pkl` files in `/models` stay ignored). `requirements.txt` pins
`scikit-learn==1.6.1`, the version the model was trained with, and
`render.yaml` sets Python 3.11.

---

## Things worth knowing before you rely on this in production

**Cold starts and the explanation cache (Vercel specifically).** One of
the optimizations in this project caches the SHAP explainer in memory
after its first use per server instance, so repeat explanations are
fast. On a serverless host like Vercel, a fully cold start (e.g. after a
period with no traffic) pays that first-load cost again. On Render, the
process stays warm the whole time it's running, so the cache holds
until the service spins down on the free tier.

## Switching back to local XAMPP later

Just unset `DATABASE_URL` (or don't set it at all) and the app falls back
to the local `DB_HOST`/`DB_USER`/etc. defaults matching a fresh XAMPP
install — see `.env.example`. The same codebase runs both ways; nothing
needs to be changed in the code itself, only which environment variables
are set.
