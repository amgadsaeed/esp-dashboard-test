"""
ESP Field Dashboard - Streamlit web app.

Team workflow:
  1. Upload the two ZIPs (All-wells snapshot + Single-wells time series).
  2. Review/adjust every detection threshold (all optional, defaults match
     the original notebook).
  3. Review/edit the auto-detected Shutdown Events, Vibration, PIP, and
     Motor Temp tables right in the browser (replaces the notebook's
     input() prompts).
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
]:
    if key not in st.session_state:
        st.session_state[key] = default

st.title("ESP Field Dashboard")
st.caption("Upload the field export ZIPs, review the auto-detected tables, then build the report.")

# ------------------------------------------------------------------
# Sidebar: logos + GitHub status
# ------------------------------------------------------------------
with st.sidebar:
    st.header("Branding (optional)")
    tam_logo_file = st.file_uploader("TAM logo (PNG)", type=["png"], key="tam_logo")
    khalda_logo_file = st.file_uploader("Khalda logo (PNG)", type=["png"], key="khalda_logo")

    st.header("GitHub history")
    if github_configured():
        st.success("Connected \u2014 reports are saved to GitHub automatically.")
    else:
        st.warning(
            "Not configured. Add `GITHUB_TOKEN`, `GITHUB_REPO`, and (optionally) "
            "`GITHUB_BRANCH` in Streamlit secrets to enable history."
        )

# Persist uploaded logos to disk so dashboard_lib (which reads them by path) can use them
tmp_assets_dir = os.path.join(tempfile.gettempdir(), "esp_dashboard_assets")
os.makedirs(tmp_assets_dir, exist_ok=True)
if tam_logo_file is not None:
    p = os.path.join(tmp_assets_dir, "tam_logo.png")
    with open(p, "wb") as f:
        f.write(tam_logo_file.getbuffer())
    dl.TAM_LOGO_PATH = p
if khalda_logo_file is not None:
    p = os.path.join(tmp_assets_dir, "khalda_logo.png")
    with open(p, "wb") as f:
        f.write(khalda_logo_file.getbuffer())
    dl.KHALDA_LOGO_PATH = p

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

# (attr on dashboard_lib, label, min_value, step, is_a_time_window_in_minutes, help text)
THRESHOLD_SPEC = [
    ("VX_THRESHOLD_G", "High vibration alert threshold (G)", 0.0, 0.1, False,
     "Any single Vx reading above this is flagged as a vibration alert."),
    ("VX_GLITCH_CHECK_G", "Vibration glitch-check threshold (G)", 0.0, 0.1, False,
     "Spikes at/above this need a shutdown nearby to count as real, not sensor noise."),
    ("VX_GLITCH_WINDOW_BACK", "Glitch tolerance window \u2014 before spike (min)", 0.0, 1.0, True,
     "How far back a shutdown can start and still explain a high-Vx spike."),
    ("VX_GLITCH_WINDOW_FWD", "Glitch tolerance window \u2014 after spike (min)", 0.0, 5.0, True,
     "How far forward a shutdown can start and still explain a high-Vx spike."),
    ("VX_DOUBLE_RATIO", "Vibration doubling ratio", 1.0, 0.1, False,
     "Flag a well if its settled Vx is at least this many times its baseline Vx."),
    ("VX_DOUBLE_MIN_BASELINE_G", "Min. baseline Vx to check doubling (G)", 0.0, 0.01, False,
     "Avoids flagging near-zero baseline noise as a 'doubling'."),
    ("VX_DOUBLE_BASELINE_WINDOW", "Vibration baseline/7AM window (min)", 5.0, 5.0, True,
     "Window used to compute the starting and 7 AM Vx levels for the doubling check."),
    ("PIP_RISE_THRESHOLD_PSI", "Sustained PIP rise threshold (psi)", 0.0, 1.0, False,
     "Minimum sustained PIP increase (baseline to 7 AM) to flag a rising-PIP trend."),
    ("PIP_BASELINE_WINDOW", "PIP baseline/7AM window (min)", 5.0, 5.0, True,
     "Window used to compute the starting and 7 AM PIP levels."),
    ("TEMP_RISE_THRESHOLD_F", "Sustained motor-temp rise threshold (\u00b0F)", 0.0, 0.5, False,
     "Minimum sustained motor-temp increase (baseline to 7 AM) to flag a well."),
    ("TEMP_BASELINE_WINDOW", "Motor-temp baseline/7AM window (min)", 5.0, 5.0, True,
     "Window used to compute the starting (or post-restart) and 7 AM motor-temp levels."),
    ("TEMP_DECLINE_TOLERANCE_F", "Motor-temp cooling tolerance (\u00b0F)", 0.0, 0.5, False,
     "If the well has cooled by at least this much from its recent level, treat it as recovering, not a sustained rise."),
    ("IMPLAUSIBLE_TEMP_F", "Implausible temperature floor (\u00b0F)", 0.0, 1.0, False,
     "Temp readings below this are treated as a sensor/comm glitch, not a real reading."),
]


def _default_for(attr_name):
    val = getattr(dl, attr_name)
    return val.total_seconds() / 60.0 if isinstance(val, pd.Timedelta) else float(val)


threshold_values = {}
with st.expander("Adjust thresholds", expanded=True):
    t_cols = st.columns(2)
    for i, (attr, label, min_val, step, is_minutes, help_text) in enumerate(THRESHOLD_SPEC):
        with t_cols[i % 2]:
            threshold_values[attr] = st.number_input(
                label, min_value=min_val, value=_default_for(attr), step=step,
                help=help_text, key=f"threshold_{attr}",
            )

process_clicked = st.button("Process data", type="primary", disabled=not (all_wells_zip or single_wells_zip))

if process_clicked:
    # Apply the (possibly edited) thresholds to dashboard_lib before running
    # any detection - every detector reads these as plain module globals,
    # so setting them here affects this run's process_folder()/build_summary().
    for attr, label, min_val, step, is_minutes, help_text in THRESHOLD_SPEC:
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

    # ---- Shutdown events (editable: add/edit/delete rows) ----
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
    # Always recompute Downtime from Start/End, overriding any stale or
    # manually-typed value - this runs on every rerun (i.e. every edit),
    # so the disabled column above reflects the latest Start/End on the
    # very next interaction.
    _start_ts = pd.to_datetime(edited_shutdown["Shutdown Start"], errors="coerce")
    _end_ts = pd.to_datetime(edited_shutdown["Shutdown End"], errors="coerce")
    edited_shutdown["Downtime (hrs)"] = (
        (_end_ts - _start_ts).dt.total_seconds() / 3600.0
    ).round(2)

    # ---- High vibration alerts (editable: delete rows only, matches notebook) ----
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

    # ---- Rising PIP trends (editable: delete rows only) ----
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

    # ---- Sustained motor temp increase (editable: delete rows only) ----
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
        # Recompute derived tables to match whatever edits were just made,
        # same logic as the notebook's review_shutdown_table().
        df = edited_shutdown.copy()
        if not df.empty:
            # Downtime is already auto-calculated from Start/End above; a
            # missing End (still ongoing / unknown) leaves it as NaN rather
            # than a manually-typed guess - treat that as 0.0 for sorting
            # and the chart/export, same fallback the notebook used.
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
