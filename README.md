# ESP Field Dashboard (web app)

A Streamlit rebuild of `KPC_Field_Dashboard_V3.ipynb`: upload the two field
export ZIPs in a browser, review/edit the auto-detected tables, and get the
desktop PNG, mobile/WhatsApp PNG, Excel workbook, and WhatsApp message —
no local Python setup, no editing file paths in a notebook.

- `dashboard_lib.py` — the notebook's data-processing and chart-building
  logic, unchanged in behavior, just extracted from notebook cells into an
  importable module.
- `app.py` — the Streamlit UI: upload → review/edit tables → build outputs.
- `github_sync.py` — pushes each generated report to a GitHub repo so you
  get commit history (who ran it, when, what the numbers were).

## 1. Put this on GitHub

Create a new **private** GitHub repo (e.g. `esp-dashboard`) and push this
folder to it:

```bash
cd esp-dashboard
git init
git add .
git commit -m "Initial ESP field dashboard app"
git branch -M main
git remote add origin https://github.com/<your-org>/esp-dashboard.git
git push -u origin main
```

`.gitignore` already excludes `.streamlit/secrets.toml`, so you won't
accidentally commit a token.

## 2. Deploy to Streamlit Community Cloud (free)

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **New app**, pick your `esp-dashboard` repo, branch `main`, main
   file `app.py`.
3. Click **Deploy**. You'll get a public URL like
   `https://esp-dashboard-<random>.streamlit.app` — share that with your
   team. (It's an unlisted URL, not indexed or listed anywhere public —
   per your earlier choice, it isn't further password-protected.)

Redeploys happen automatically whenever you push to `main`.

## 3. Turn on GitHub history for the *data* (separate from step 1)

This is a second, optional repo that stores every generated report — think
of it as your run history, distinct from the app's own source code repo.
You can reuse the same repo from step 1, or create a fresh one just for
data (recommended, so the app's commit history stays small and readable).

1. Create the data repo, e.g. `esp-dashboard-data` (private is fine).
2. Create a **fine-grained personal access token**:
   GitHub → Settings → Developer settings → Personal access tokens →
   Fine-grained tokens → Generate new token.
   - Repository access: **Only** `esp-dashboard-data`.
   - Permissions: **Contents: Read and write**. Nothing else needed.
3. In your Streamlit Cloud app: **Settings → Secrets**, paste:

   ```toml
   GITHUB_TOKEN = "github_pat_xxxxxxxxxxxxxxxxxxxxxxxx"
   GITHUB_REPO = "your-org/esp-dashboard-data"
   GITHUB_BRANCH = "main"
   ```

4. Save. The app will now show "Connected" in the sidebar, and the
   **"Push this report to GitHub"** button on the Report step will commit
   the PNGs, Excel workbook, and WhatsApp message to
   `reports/<YYYY-MM-DD>/<timestamp>/` in that repo — one commit per file,
   so `git log` / GitHub's file history give you a full audit trail of
   every report your team has generated.

For local testing only, copy `.streamlit/secrets.toml.example` to
`.streamlit/secrets.toml` and fill in real values — never commit that file.

## 4. Logos (optional)

Drop `tam_logo.png` and `khalda_logo.png` into the `assets/` folder and
push them to the app's repo to have them appear by default on every
report. Anyone using the app can also upload replacement logos for a
single session from the sidebar, without touching the repo.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What changed vs. the notebook

- The notebook's `input()` prompts for reviewing/editing tables (adding a
  missed shutdown, deleting a false-positive vibration alert, etc.) are
  now editable tables in the browser (`st.data_editor`) — same
  capabilities, no typing row indices into a text prompt.
- Hardcoded local file paths are gone; you upload the ZIPs and (optionally)
  logos through the browser instead of editing a CONFIG cell.
- Every other calculation (shutdown detection, vibration glitch filtering,
  PIP trend logic, fault-code lookup, etc.) is untouched from the notebook.
