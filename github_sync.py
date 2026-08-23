"""
Push generated report files to a GitHub repo using the Contents API, so
every run is committed and the repo's commit history becomes the audit
trail (who ran it, when, what the numbers were).

Configure via Streamlit secrets (.streamlit/secrets.toml locally, or the
"Secrets" panel in Streamlit Community Cloud):

    GITHUB_TOKEN  = "ghp_xxx..."      # fine-grained PAT, Contents: Read & write, on the target repo only
    GITHUB_REPO   = "your-org/esp-dashboard-data"
    GITHUB_BRANCH = "main"            # optional, defaults to "main"

Reports are written under reports/<YYYY-MM-DD>/<timestamp>/<filename>,
one commit per file (simplest reliable approach against the Contents API).
"""
import base64
import os

import requests
import streamlit as st

API_ROOT = "https://api.github.com"


def _secret(name, default=None):
    # st.secrets raises if secrets.toml doesn't exist at all locally; fall
    # back to environment variables so the app also works outside Streamlit
    # Cloud (e.g. local testing, other hosts).
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)


def github_configured() -> bool:
    return bool(_secret("GITHUB_TOKEN")) and bool(_secret("GITHUB_REPO"))


def push_report_to_github(files: dict, ts: str, author: str = "unknown") -> str:
    """
    files: {filename: local_filepath}
    ts: timestamp string used to group this run's files in one folder
    Returns a URL to the last commit made (for display to the user).
    """
    token = _secret("GITHUB_TOKEN")
    repo = _secret("GITHUB_REPO")
    branch = _secret("GITHUB_BRANCH", "main")

    if not token or not repo:
        raise RuntimeError("GitHub is not configured (missing GITHUB_TOKEN / GITHUB_REPO).")

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    date_folder = ts[:8]  # YYYYMMDD
    date_folder = f"{date_folder[:4]}-{date_folder[4:6]}-{date_folder[6:8]}"
    last_commit_url = None

    for filename, filepath in files.items():
        with open(filepath, "rb") as f:
            content_b64 = base64.b64encode(f.read()).decode("utf-8")

        repo_path = f"reports/{date_folder}/{ts}/{filename}"
        url = f"{API_ROOT}/repos/{repo}/contents/{repo_path}"

        payload = {
            "message": f"ESP dashboard report {ts} ({filename}) \u2014 by {author}",
            "content": content_b64,
            "branch": branch,
        }

        resp = requests.put(url, headers=headers, json=payload, timeout=60)
        if resp.status_code not in (200, 201):
            raise RuntimeError(
                f"Failed to upload {filename}: {resp.status_code} {resp.text[:300]}"
            )
        last_commit_url = resp.json().get("commit", {}).get("html_url")

    return last_commit_url or f"https://github.com/{repo}/tree/{branch}/reports/{date_folder}/{ts}"
