"""
ESP Field Dashboard - Streamlit web app.

Team workflow:
  1. Upload the two ZIPs (All-wells snapshot + Single-wells time series).
  2. Review/adjust every detection threshold (all optional, defaults match
     the original notebook).
  3. Review/edit the auto-detected Shutdown Events, Vibration, PIP, Motor Temp 
     tables, and Lost Communication Wells right in the browser.
  4. Build outputs -> desktop PNG, mobile/WhatsApp PNG, Excel workbook,
     WhatsApp message text.
  5. Optionally push the generated report to a GitHub repo, so every run
     is saved with full history (who ran it, when, and what changed).
"""
import io
import os
import shutil
import tempfile
import zipfile
import requests  # <-- NEW
import re        # <-- NEW
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

import dashboard_lib as dl
from github_sync import push_report_to_github, github_configured

st.set_page_config(page_title="ESP Field Dashboard", page_icon="\U0001F4CA", layout="wide")

# ------------------------------------------------------------------
# Session state defaults
# ------------------------------------------------------------------
for key, default in [
    ("summary", None),
    ("results", None),
    ("report_date", None),
    ("stage", "upload"),  # upload -> review -> done
    ("viewing_history", None), # <-- NEW: Tracks the selected past run
]:
    if key not in st.session_state:
        st.session_state[key] = default

st.title("ESP Field Dashboard")
st.caption("Upload the field export ZIPs, review the auto-detected tables, then build the report.")

# ==================================================================
# HISTORY VIEWER MODE (Intercepts main UI)
# ==================================================================
if st.session_state.stage == "history" and st.session_state.viewing_history:
    run_data = st.session_state.viewing_history
    
    col1, col2 = st.columns([3, 1])
    col1.info("🕒 You are viewing a historical report loaded directly from GitHub.")
    if col2.button("← Exit History Mode", use_container_width=True):
        st.session_state.stage = "upload"
        st.session_state.viewing_history = None
        st.rerun()
        
    st.divider()
    st.subheader("Historical Outputs")
    
    with st.spinner("Downloading report assets from GitHub..."):
        desk_bytes = download_github_file(run_data.get('desktop')) if 'desktop' in run_data else None
        mob_bytes = download_github_file(run_data.get('mobile')) if 'mobile' in run_data else None
        txt_bytes = download_github_file(run_data.get('whatsapp')) if 'whatsapp' in run_data else None
        xls_bytes = download_github_file(run_data.get('excel')) if 'excel' in run_data else None
    
    htab1, htab2, htab3 = st.tabs(["Desktop dashboard", "Mobile / WhatsApp", "WhatsApp message"])
    
    with htab1:
        if desk_bytes:
            st.image(desk_bytes, use_container_width=True)
            st.download_button("Download desktop PNG", desk_bytes, file_name=os.path.basename(run_data['desktop']), mime="image/png")
        else:
            st.warning("Desktop image not found for this run.")
            
    with htab2:
        if mob_bytes:
            st.image(mob_bytes, width=420)
            st.download_button("Download mobile PNG", mob_bytes, file_name=os.path.basename(run_data['mobile']), mime="image/png")
        else:
            st.warning("Mobile image not found for this run.")
            
    with htab3:
        if txt_bytes:
            txt_str = txt_bytes.decode('utf-8')
            st.text_area("Message", txt_str, height=300, disabled=True)
            st.download_button("Download message .txt", txt_str, file_name=os.path.basename(run_data['whatsapp']))
        else:
            st.warning("WhatsApp text not found for this run.")
            
    if xls_bytes:
        st.download_button(
            "Download Excel workbook", xls_bytes, file_name=os.path.basename(run_data['excel']),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        
    st.stop() # Prevents Steps 1-4 from rendering while viewing history
    
# ==================================================================
# STEP 1 - UPLOAD + PROCESS 
# (Leave your existing Step 1 code right below here)
# ==================================================================

# ------------------------------------------------------------------
# GitHub History Fetcher
# ------------------------------------------------------------------
@st.cache_data(ttl=60)
def fetch_github_runs():
    """Finds all previous dashboard runs saved in the repo."""
    if not github_configured(): return {}
    
    token = st.secrets["GITHUB_TOKEN"]
    repo = st.secrets["GITHUB_REPO"]
    branch = st.secrets.get("GITHUB_BRANCH", "main")
    
    # Pull the entire repo tree recursively
    url = f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.v3+json"}
    res = requests.get(url, headers=headers)
    
    runs = {}
    if res.status_code == 200:
        for f in res.json().get('tree', []):
            # Look for files with the ESP_ prefix
            if f['type'] == 'blob' and 'ESP_' in f['path']:
                # Extract your timestamp format (YYYYMMDD_HHMMSS)
                match = re.search(r'_(\d{8}_\d{6})\.', f['path'])
                if match:
                    ts = match.group(1)
                    if ts not in runs: runs[ts] = {}
                    
                    # Group related files by timestamp
                    if "WhatsApp" in f['path']: runs[ts]['whatsapp'] = f['path']
                    elif "Summary" in f['path']: runs[ts]['excel'] = f['path']
                    elif "Mobile" in f['path']: runs[ts]['mobile'] = f['path']
                    elif "Dashboard" in f['path']: runs[ts]['desktop'] = f['path']
    return runs

@st.cache_data(ttl=3600)
def download_github_file(path):
    """Downloads raw file content securely using the GitHub API."""
    repo = st.secrets["GITHUB_REPO"]
    branch = st.secrets.get("GITHUB_BRANCH", "main")
    url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={branch}"
    headers = {"Authorization": f"Bearer {st.secrets['GITHUB_TOKEN']}", "Accept": "application/vnd.github.v3.raw"}
    res = requests.get(url, headers=headers)
    return res.content if res.status_code == 200 else None

# ------------------------------------------------------------------
# Sidebar: GitHub status & History
# ------------------------------------------------------------------
with st.sidebar:
    st.header("GitHub history")
    if github_configured():
        st.success("Connected — reports are saved to GitHub automatically.")
        
        st.divider()
        st.subheader("Load Past Report")
        runs = fetch_github_runs()
        
        if runs:
            sorted_ts = sorted(runs.keys(), reverse=True)
            # Format timestamps cleanly for the UI
            display_names = {
                ts: datetime.strptime(ts, "%Y%m%d_%H%M%S").strftime("%b %d, %Y - %H:%M") 
                for ts in sorted_ts
            }
            
            selected_ts = st.selectbox("Select a date:", sorted_ts, format_func=lambda x: display_names[x])
            
            if st.button("View Report", type="primary"):
                st.session_state.viewing_history = runs[selected_ts]
                st.session_state.stage = "history"
                st.rerun()
        else:
            st.info("No saved reports found in the repository yet.")
    else:
        st.warning(
            "Not configured. Add `GITHUB_TOKEN`, `GITHUB_REPO`, and (optionally) "
            "`GITHUB_BRANCH` in Streamlit secrets to enable history."
        )

# ==================================================================
# STEP 1 - UPLOAD + PROCESS
# ==================================================================
st.header("1. Upload field data")

col1, col2 = st.columns(2)
with col1:
    all_wells_zip = st.file_uploader(
        "All-wells snapshot ZIP (single sheet, all wells)", type=["zip"], key="all_wells_zip"
    )
with col2:
    single_wells_zip = st.file_uploader(
        "Single-well exports ZIP (one .xlsx per well)", type=["zip"], key="single_wells_zip"
    )

fallback_total = st.number_input(
    "Total wells in field (only used as a fallback if no master well-list sheet is found "
    "in the uploads)",
    min_value=0, value=0, step=1,
    help="Leave at 0 to just use the number of well files found.",
)

# ==================================================================
# STEP 2 - DETECTION THRESHOLDS (all limits/constraints, editable)
# ==================================================================
st.header("2. Detection thresholds")
st.caption(
    "These control what counts as an alert. Defaults match the original notebook \u2014 "
    "change any value below before processing if you want different sensitivity."
)

ESSENTIAL_THRESHOLDS = [
    ("PIP_RISE_THRESHOLD_PSI", "Sustained PIP rise threshold (psi)", 0.0, 1.0, False,
     "Minimum sustained PIP increase (from baseline to current level) to flag a rising-PIP trend indicating possible plugging or changes in intake. Default is 25 psi."),
    ("TEMP_RISE_THRESHOLD_F", "Sustained motor-temp rise threshold (\u00b0F)", 0.0, 0.5, False,
     "Minimum sustained motor temperature increase to flag a well, which could indicate poor cooling or motor overload. Default is 3\u00b0F."),
    ("VX_THRESHOLD_G", "High vibration alert threshold (G)", 0.0, 0.1, False,
     "Absolute vibration alert threshold. Any verified spike above this value triggers an immediate high-vibration alert."),
]

ADVANCED_THRESHOLDS = [
    ("VX_GLITCH_CHECK_G", "Vibration glitch-check threshold (G)", 0.0, 0.1, False,
     "Spikes at or above this value are cross-checked against shutdown history to eliminate false telemetry/sensor noise."),
    ("VX_GLITCH_WINDOW_BACK", "Glitch tolerance window \u2014 before spike (min)", 0.0, 1.0, True,
     "Time window before a high-vibration spike to check for a shutdown event that might explain the transient."),
    ("VX_GLITCH_WINDOW_FWD", "Glitch tolerance window \u2014 after spike (min)", 0.0, 5.0, True,
     "Time window after a high-vibration spike to verify if a shutdown event occurred, validating the transient."),
    ("VX_DOUBLE_RATIO", "Vibration doubling ratio", 1.0, 0.1, False,
     "Multiplier against the baseline vibration. Flags a well if its settled Vx reaches or exceeds this multiple of its baseline."),
    ("VX_DOUBLE_MIN_BASELINE_G", "Min. baseline Vx to check doubling (G)", 0.0, 0.01, False,
     "Floor for baseline vibration to avoid false doubling alarms caused by dividing by near-zero background noise."),
    ("VX_DOUBLE_BASELINE_WINDOW", "Vibration baseline/7AM window (min)", 5.0, 5.0, True,
     "Rolling timeframe used to establish the starting baseline and current settled Vx levels for the doubling check."),
    ("PIP_BASELINE_WINDOW", "PIP baseline/7AM window (min)", 5.0, 5.0, True,
     "Timeframe used to determine stable PIP averages, ignoring temporary dips or peaks."),
    ("TEMP_BASELINE_WINDOW", "Motor-temp baseline/7AM window (min)", 5.0, 5.0, True,
     "Timeframe to lock in the starting and current temperature levels. Also anchors to post-restart periods if a trip occurred."),
    ("TEMP_DECLINE_TOLERANCE_F", "Motor-temp cooling tolerance (\u00b0F)", 0.0, 0.5, False,
     "Degrees of cooling accepted before a well is no longer considered to have an actively 'sustained' rise (e.g. recovering from a restart)."),
    ("IMPLAUSIBLE_TEMP_F", "Implausible temperature floor (\u00b0F)", 0.0, 1.0, False,
     "Readings below this physically impossible downhole temperature cause the entire row to be dropped as a telemetry glitch."),
]

def _default_for(attr_name):
    val = getattr(dl, attr_name)
    return val.total_seconds() / 60.0 if isinstance(val, pd.Timedelta) else float(val)

threshold_values = {}
with st.expander("Essential Limits (PIP, Temp, High Vib)", expanded=True):
    t_cols = st.columns(2)
    for i, (attr, label, min_val, step, is_minutes, help_text) in enumerate(ESSENTIAL_THRESHOLDS):
        with t_cols[i % 2]:
            threshold_values[attr] = st.number_input(
                label, min_value=min_val, value=_default_for(attr), step=step,
                help=help_text, key=f"threshold_{attr}",
            )

with st.expander("Advanced Limits & Tolerances", expanded=False):
    t_cols = st.columns(2)
    for i, (attr, label, min_val, step, is_minutes, help_text) in enumerate(ADVANCED_THRESHOLDS):
        with t_cols[i % 2]:
            threshold_values[attr] = st.number_input(
                label, min_value=min_val, value=_default_for(attr), step=step,
                help=help_text, key=f"threshold_{attr}",
            )

process_clicked = st.button("Process data", type="primary", disabled=not (all_wells_zip or single_wells_zip))

if process_clicked:
    # Apply thresholds
    for attr, label, min_val, step, is_minutes, help_text in ESSENTIAL_THRESHOLDS + ADVANCED_THRESHOLDS:
        val = threshold_values[attr]
        setattr(dl, attr, pd.Timedelta(minutes=val) if is_minutes else val)

    tmp_root = tempfile.mkdtemp(prefix="esp_dash_")
    try:
        all_wells_path = None
        single_wells_path = None
        if all_wells_zip is not None:
            all_wells_path = os.path.join(tmp_root, "all_wells.zip")
            with open(all_wells_path, "wb") as f:
                f.write(all_wells_zip.getbuffer())
        if single_wells_zip is not None:
            single_wells_path = os.path.join(tmp_root, "single_wells.zip")
            with open(single_wells_path, "wb") as f:
                f.write(single_wells_zip.getbuffer())

        with st.spinner("Extracting and analyzing well files..."):
            folder = dl.extract_zips_to_temp(all_wells_path, single_wells_path, tmp_root)
            results, master_wells, master_df = dl.process_folder(folder)

        if not results:
            st.error(f"No .xlsx well files found in the uploaded ZIP(s).")
        else:
            if master_wells:
                summary = dl.build_summary(results, miscommunication_wells=master_wells, master_snapshot=master_df)
                st.info(f"Detected a field master list ({len(master_wells)} wells listed) \u2014 "
                        f"Miscommunication wells are reported by name.")
            else:
                total_expected = int(fallback_total) if fallback_total else len(results)
                summary = dl.build_summary(results, total_wells_expected=total_expected)

            dates = [r["date_range"][0] for r in results if r.get("date_range")]
            report_start = dates[0].normalize() if dates else pd.Timestamp(datetime.now().date())
            report_end = report_start + pd.Timedelta(days=1)
            report_date = (
                f"{report_start.strftime('%d-%b-%Y')} 7 AM  to  "
                f"{report_end.strftime('%d-%b-%Y')} 7 AM"
            )

            st.session_state.summary = summary
            st.session_state.results = results
            st.session_state.report_date = report_date
            st.session_state.stage = "review"
            st.success(
                f"Processed {summary['files_found']} well file(s) | Total wells: {summary['total']} | "
                f"Miscommunication: {summary['miscommunication']} | "
                f"Shutdown events: {len(summary['shutdown_df'])} | "
                f"Vx > {dl.VX_THRESHOLD_G}G wells: {len(summary['vx_df'])} | "
                f"Rising PIP wells: {len(summary['pip_df'])} | "
                f"Motor temp rise wells: {len(summary['temp_df'])}"
            )
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

# ==================================================================
# STEP 3 - REVIEW / EDIT TABLES
# ==================================================================
if st.session_state.stage in ("review", "done") and st.session_state.summary is not None:
    st.header("3. Review / edit tables")
    st.caption(
        "Add, edit, or delete rows below \u2014 mirrors the notebook's REVIEW / EDIT TABLES step. "
        "Nothing is finalized until you click 'Build outputs' below."
    )
    summary = st.session_state.summary

    # ---- Lost Communication Wells ----
    st.subheader("Lost Communication Wells (> 3 hours)")
    st.caption("Wells missing telemetry for over 3 hours. Auto-detected wells are listed below; you can manually edit or add more.")
    
    mc_wells = summary.get('miscommunication_wells', [])
    mc_df = pd.DataFrame([{"Well": w, "Notes": "Auto-detected"} for w in mc_wells])
    if mc_df.empty:
        mc_df = pd.DataFrame(columns=["Well", "Notes"])
        
    edited_mc = st.data_editor(
        mc_df,
        num_rows="dynamic",
        use_container_width=True,
        key="mc_editor",
        column_config={
            "Well": st.column_config.TextColumn("Well Name", required=True),
            "Notes": st.column_config.TextColumn("Notes / Status")
        },
    )

    # ---- Shutdown events ----
    st.subheader("Shutdown Events")
    shutdown_cols = ["Well", "Shutdown Start", "Shutdown End", "Downtime (hrs)", "Reason", "Ongoing"]
    shutdown_df = summary["shutdown_df"].copy()
    if not shutdown_df.empty:
        shutdown_df = shutdown_df.reindex(columns=shutdown_cols)
    else:
        shutdown_df = pd.DataFrame(columns=shutdown_cols)
    st.caption(
        "**Downtime (hrs) is calculated automatically** from Shutdown Start/End "
        "and can't be typed in directly \u2014 edit the Start/End times instead."
    )
    edited_shutdown = st.data_editor(
        shutdown_df,
        num_rows="dynamic",
        use_container_width=True,
        key="shutdown_editor",
        column_config={
            "Shutdown Start": st.column_config.DatetimeColumn(),
            "Shutdown End": st.column_config.DatetimeColumn(),
            "Downtime (hrs)": st.column_config.NumberColumn(
                disabled=True, help="Auto-calculated from Shutdown Start/End."
            ),
            "Ongoing": st.column_config.CheckboxColumn(),
        },
    )
    _start_ts = pd.to_datetime(edited_shutdown["Shutdown Start"], errors="coerce")
    _end_ts = pd.to_datetime(edited_shutdown["Shutdown End"], errors="coerce")
    edited_shutdown["Downtime (hrs)"] = (
        (_end_ts - _start_ts).dt.total_seconds() / 3600.0
    ).round(2)

    # ---- High vibration alerts ----
    st.subheader(f"High Vibration Alerts (Vx > {dl.VX_THRESHOLD_G}G)")
    vx_df = summary["vx_df"].copy().reset_index(drop=True)
    if vx_df.empty:
        st.caption("No wells exceeded the vibration threshold.")
        edited_vx = vx_df
    else:
        vx_df.insert(0, "Keep", True)
        edited_vx_raw = st.data_editor(
            vx_df, use_container_width=True, key="vx_editor", disabled=[c for c in vx_df.columns if c != "Keep"]
        )
        edited_vx = edited_vx_raw[edited_vx_raw["Keep"]].drop(columns=["Keep"]).reset_index(drop=True)

    # ---- Rising PIP trends ----
    st.subheader(f"Rising PIP Trends (> {dl.PIP_RISE_THRESHOLD_PSI} psi)")
    pip_df = summary["pip_df"].copy().reset_index(drop=True)
    if pip_df.empty:
        st.caption("No wells showed a sustained PIP rise.")
        edited_pip = pip_df
    else:
        pip_df.insert(0, "Keep", True)
        edited_pip_raw = st.data_editor(
            pip_df, use_container_width=True, key="pip_editor", disabled=[c for c in pip_df.columns if c != "Keep"]
        )
        edited_pip = edited_pip_raw[edited_pip_raw["Keep"]].drop(columns=["Keep"]).reset_index(drop=True)

    # ---- Sustained motor temp increase ----
    st.subheader(f"Sustained Motor Temp Increase (> {dl.TEMP_RISE_THRESHOLD_F}\u00b0F, still up at 7AM)")
    temp_df = summary["temp_df"].copy().reset_index(drop=True)
    if temp_df.empty:
        st.caption("No wells showed a sustained motor-temp rise.")
        edited_temp = temp_df
    else:
        temp_df.insert(0, "Keep", True)
        edited_temp_raw = st.data_editor(
            temp_df, use_container_width=True, key="temp_editor", disabled=[c for c in temp_df.columns if c != "Keep"]
        )
        edited_temp = edited_temp_raw[edited_temp_raw["Keep"]].drop(columns=["Keep"]).reset_index(drop=True)

    st.divider()
    build_clicked = st.button("Build outputs", type="primary")

    if build_clicked:
        # Recompute miscommunication table
        new_mc_wells = edited_mc["Well"].dropna().astype(str).tolist()
        summary["miscommunication_wells"] = new_mc_wells
        summary["miscommunication"] = len(new_mc_wells)
        summary['total'] = summary['files_found'] + summary['miscommunication']

        # Update well status rows
        ws_df = summary["well_status_df"].copy()
        ws_df = ws_df[ws_df['Status'] != 'Miscommunication']
        new_rows = []
        for _, row in edited_mc.iterrows():
            w = row['Well']
            if not w or str(w).strip() == "": continue
            notes = row['Notes'] if pd.notna(row['Notes']) and str(row['Notes']).strip() != "" else 'Manually added / No data'
            new_rows.append({
                'Well': w, 'Status': 'Miscommunication',
                'Last Freq (Hz)': None, 'Last Vx (G)': None,
                'Rows Logged': 0, 'Error': notes
            })
        if new_rows:
            summary["well_status_df"] = pd.concat([ws_df, pd.DataFrame(new_rows)], ignore_index=True)
        else:
            summary["well_status_df"] = ws_df

        # Recompute derived tables 
        df = edited_shutdown.copy()
        if not df.empty:
            df["Downtime (hrs)"] = pd.to_numeric(df["Downtime (hrs)"], errors="coerce").fillna(0.0)
            df = df.sort_values("Downtime (hrs)", ascending=False).reset_index(drop=True)
            shutdown_count_df = (
                df.groupby("Well").size().reset_index(name="Shutdown Count")
                .sort_values("Shutdown Count", ascending=False).reset_index(drop=True)
            )
            reason_counts = df["Reason"].value_counts()
        else:
            shutdown_count_df = pd.DataFrame(columns=["Well", "Shutdown Count"])
            reason_counts = pd.Series(dtype=int)

        summary["shutdown_df"] = df
        summary["shutdown_count_df"] = shutdown_count_df
        summary["reason_counts"] = reason_counts
        summary["vx_df"] = edited_vx.sort_values("Max Vx (G)", ascending=False).reset_index(drop=True) if not edited_vx.empty else edited_vx
        summary["pip_df"] = edited_pip.sort_values("Net Rise (psi)", ascending=False).reset_index(drop=True) if not edited_pip.empty else edited_pip
        summary["temp_df"] = edited_temp.sort_values("Rise (F)", ascending=False).reset_index(drop=True) if not edited_temp.empty else edited_temp
        st.session_state.summary = summary
        st.session_state.stage = "done"

# ==================================================================
# STEP 4 - OUTPUTS
# ==================================================================
if st.session_state.stage == "done" and st.session_state.summary is not None:
    st.header("4. Report")
    summary = st.session_state.summary
    results = st.session_state.results
    report_date = st.session_state.report_date

    out_dir = tempfile.mkdtemp(prefix="esp_dash_out_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_path = os.path.join(out_dir, f"ESP_Field_Dashboard_{ts}.png")
    png_mobile_path = os.path.join(out_dir, f"ESP_Field_Dashboard_Mobile_{ts}.png")
    xlsx_path = os.path.join(out_dir, f"ESP_Field_Summary_{ts}.xlsx")
    whatsapp_path = os.path.join(out_dir, f"ESP_WhatsApp_Message_{ts}.txt")

    with st.spinner("Rendering dashboard..."):
        dl.build_dashboard_figure(summary, report_date, png_path)
        dl.build_mobile_dashboard_figure(summary, report_date, png_mobile_path)
        dl.export_excel(summary, results, xlsx_path)
        whatsapp_message = dl.build_whatsapp_message(summary, report_date)
        with open(whatsapp_path, "w", encoding="utf-8") as f:
            f.write(whatsapp_message)

    tab1, tab2, tab3 = st.tabs(["Desktop dashboard", "Mobile / WhatsApp", "WhatsApp message"])
    with tab1:
        st.image(png_path, use_container_width=True)
        with open(png_path, "rb") as f:
            st.download_button("Download desktop PNG", f, file_name=os.path.basename(png_path), mime="image/png")
    with tab2:
        st.image(png_mobile_path, width=420)
        with open(png_mobile_path, "rb") as f:
            st.download_button("Download mobile PNG", f, file_name=os.path.basename(png_mobile_path), mime="image/png")
    with tab3:
        st.text_area("Message", whatsapp_message, height=300)
        st.download_button("Download message .txt", whatsapp_message, file_name=os.path.basename(whatsapp_path))

    with open(xlsx_path, "rb") as f:
        st.download_button(
            "Download Excel workbook", f, file_name=os.path.basename(xlsx_path),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    st.divider()
    st.subheader("Save to GitHub (history)")
    if github_configured():
        author = st.text_input("Your name (recorded in the commit message)", key="author_name")
        if st.button("Push this report to GitHub"):
            with st.spinner("Pushing to GitHub..."):
                try:
                    files = {
                        os.path.basename(png_path): png_path,
                        os.path.basename(png_mobile_path): png_mobile_path,
                        os.path.basename(xlsx_path): xlsx_path,
                        os.path.basename(whatsapp_path): whatsapp_path,
                    }
                    commit_url = push_report_to_github(files, ts=ts, author=author or "unknown")
                    st.success(f"Pushed to GitHub. [View commit]({commit_url})")
                except Exception as e:
                    st.error(f"GitHub push failed: {e}")
    else:
        st.info(
            "GitHub isn't configured for this app yet. See the README for how to add "
            "`GITHUB_TOKEN` / `GITHUB_REPO` to Streamlit secrets."
        )

    st.divider()
    if st.button("Start a new report"):
        for key in ("summary", "results", "report_date"):
            st.session_state[key] = None
        st.session_state.stage = "upload"
        st.rerun()
