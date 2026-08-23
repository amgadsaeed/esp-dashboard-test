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
# Sidebar: GitHub status
# ------------------------------------------------------------------
with st.sidebar:
    st.header("GitHub history")
    if github_configured():
        st.success("Connected \u2014 reports are saved to GitHub automatically.")
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

# Apply thresholds immediately to DL namespace so they persist across Streamlit re-runs
for attr, label, min_val, step, is_minutes, help_text in ESSENTIAL_THRESHOLDS + ADVANCED_THRESHOLDS:
    val = threshold_values[attr]
    setattr(dl, attr, pd.Timedelta(minutes=val) if is_minutes else val)

process_clicked = st.button("Process data", type="primary", disabled=not (all_wells_zip or single_wells_zip))

if process_clicked:
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
            dates = [r["date_range"][0] for r in results if r.get("date_range")]
            if dates:
                report_start = dates[0].normalize() + pd.Timedelta(hours=7)
                if dates[0] < report_start:
                    report_start -= pd.Timedelta(days=1)
            else:
                report_start = pd.Timestamp(datetime.now().date()) + pd.Timedelta(hours=7)
            report_end = report_start + pd.Timedelta(days=1)

            if master_wells:
                summary = dl.build_summary(results, miscommunication_wells=master_wells, master_snapshot=master_df)
                st.info(f"Detected a field master list ({len(master_wells)} wells listed) \u2014 "
                        f"Miscommunication wells are reported by name.")
            else:
                total_expected = int(fallback_total) if fallback_total else len(results)
                summary = dl.build_summary(results, total_wells_expected=total_expected)

            # Store absolute bounds inside summary for plotting
            summary['report_start'] = report_start
            summary['report_end'] = report_end
            summary['vx_threshold'] = dl.VX_THRESHOLD_G
            summary['pip_threshold'] = dl.PIP_RISE_THRESHOLD_PSI
            summary['temp_threshold'] = dl.TEMP_RISE_THRESHOLD_F

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
                f"Lost Communication: {summary['miscommunication']} | "
                f"Shutdown events: {len(summary['shutdown_df'])} | "
                f"Vx > {dl.VX_THRESHOLD_G}G wells: {len(summary['vx_df'])} | "
                f"Rising PIP wells: {len(summary['pip_df'])} | "
                f"Motor temp rise wells: {len(summary['temp_df'])}"
            )
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

# ==================================================================
# STEP 3 - REVIEW / EDIT TABLES & LAYOUT
# ==================================================================
if st.session_state.stage in ("review", "done") and st.session_state.summary is not None:
    st.header("3. Review tables & design layout")
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
    st.subheader(f"High Vibration Alerts (Vx > {summary.get('vx_threshold', 2.0)}G)")
    vx_df = summary["vx_df"].copy().reset_index(drop=True)
    
    vx_ts_col = None
    if not vx_df.empty and 'timeseries' in vx_df.columns:
        vx_ts_col = vx_df[['Well', 'timeseries']]
        vx_df = vx_df.drop(columns=['timeseries'])
        
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
    st.subheader(f"Rising PIP Trends (> {summary.get('pip_threshold', 25.0)} psi)")
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
    st.subheader(f"Sustained Motor Temp Increase (> {summary.get('temp_threshold', 3.0)}\u00b0F, still up at 7AM)")
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

    # ---- Layout settings ----
    st.divider()
    st.subheader("Layout & Design Settings")
    with st.expander("Adjust margins, offsets, and spacing", expanded=False):
        lc1, lc2, lc3 = st.columns(3)
        with lc1:
            st.markdown("**Desktop Bars & Elements**")
            desktop_hspace = st.slider("Section Vertical Spacing (Desktop)", 0.2, 0.8, 0.45, 0.05)
            desktop_wspace = st.slider("Section Horizontal Spacing (Desktop)", 0.1, 0.5, 0.25, 0.05)
            desktop_sd_margin = st.slider("Shutdown Bar Left Margin (Desktop)", 0.05, 0.5, 0.19, 0.01)
            desktop_temp_margin = st.slider("Motor Temp Bar Left Margin (Desktop)", 0.05, 0.5, 0.19, 0.01)
            desktop_vx_margin = st.slider("Vibration Chart Left Margin (Desktop)", 0.05, 0.5, 0.19, 0.01)
            desktop_pie_offset = st.slider("Pie Chart X Offset (Desktop)", -0.5, 0.5, 0.0, 0.01)
            
        with lc2:
            st.markdown("**Mobile Bars & Elements**")
            mobile_hspace = st.slider("Section Vertical Spacing (Mobile)", 0.2, 1.0, 0.50, 0.05)
            mobile_sd_margin = st.slider("Shutdown Bar Left Margin (Mobile)", 0.05, 0.6, 0.17, 0.01)
            mobile_temp_margin = st.slider("Motor Temp Bar Left Margin (Mobile)", 0.05, 0.6, 0.17, 0.01)
            mobile_vx_margin = st.slider("Vibration Chart Left Margin (Mobile)", 0.05, 0.6, 0.17, 0.01)
            mobile_pie_offset = st.slider("Pie Chart X Offset (Mobile)", -0.5, 0.5, 0.0, 0.01)
            
        with lc3:
            st.markdown("**Spacing & Logos**")
            tam_scale = st.slider("TAM Logo Size (Scale)", 0.5, 2.0, 1.0, 0.05)
            khalda_scale = st.slider("Khalda Logo Size (Scale)", 0.5, 2.0, 1.0, 0.05)
            kpi_top_pad = st.slider("Space ABOVE Cards", -0.1, 0.3, 0.0, 0.01)
            kpi_space = st.slider("Space BEFORE Lost Comm Note", 0.0, 0.3, 0.02, 0.01)
            kpi_bottom_space = st.slider("Space AFTER Lost Comm Note", 0.0, 0.3, 0.02, 0.01)
            div_pad = st.slider("Green Divider Padding", 0.0, 0.2, 0.1, 0.01)
            footer_y = st.slider("Footer Line Y Position", 0.0, 0.1, 0.035, 0.005)

    st.divider()
    build_clicked = st.button("Build outputs", type="primary")

    if build_clicked:
        new_mc_wells = edited_mc["Well"].dropna().astype(str).tolist()
        summary["miscommunication_wells"] = new_mc_wells
        summary["miscommunication"] = len(new_mc_wells)
        summary['total'] = summary['files_found'] + summary['miscommunication']

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
        
        if not edited_vx.empty and vx_ts_col is not None:
            edited_vx = edited_vx.merge(vx_ts_col, on='Well', how='left')
            
        summary["vx_df"] = edited_vx.sort_values("Max Vx (G)", ascending=False).reset_index(drop=True) if not edited_vx.empty else edited_vx
        summary["pip_df"] = edited_pip.sort_values("Net Rise (psi)", ascending=False).reset_index(drop=True) if not edited_pip.empty else edited_pip
        summary["temp_df"] = edited_temp.sort_values("Rise (F)", ascending=False).reset_index(drop=True) if not edited_temp.empty else edited_temp
        
        # Inject design settings directly into the summary payload
        summary['design_settings'] = {
            'desktop_hspace': desktop_hspace,
            'desktop_wspace': desktop_wspace,
            'desktop_sd_margin': desktop_sd_margin,
            'desktop_temp_margin': desktop_temp_margin,
            'desktop_vx_margin': desktop_vx_margin,
            'desktop_pie_offset': desktop_pie_offset,
            'mobile_hspace': mobile_hspace,
            'mobile_sd_margin': mobile_sd_margin,
            'mobile_temp_margin': mobile_temp_margin,
            'mobile_vx_margin': mobile_vx_margin,
            'mobile_pie_offset': mobile_pie_offset,
            'tam_scale': tam_scale,
            'khalda_scale': khalda_scale,
            'kpi_top_pad': kpi_top_pad,
            'kpi_space': kpi_space,
            'kpi_bottom_space': kpi_bottom_space,
            'div_pad': div_pad,
            'footer_y': footer_y
        }
        
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
