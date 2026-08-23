"""
ESP Field Dashboard - core data processing + figure/report building.

This module is a faithful extraction of the processing/plotting logic from
KPC_Field_Dashboard_V3.ipynb, with the notebook-only bits (hardcoded local
paths, %matplotlib inline, interactive input() review flow) removed so it
can be imported by the Streamlit app (app.py) and unit tested independently.

Logo paths default to files shipped alongside the app, but can be
overridden at runtime (the Streamlit app lets a user upload replacements).
"""
import os, re, sys, glob, zipfile, tempfile, shutil, warnings
from datetime import datetime

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # headless rendering - required on a server (no display)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore', message='Workbook contains no default style')

# Logo files shown top-left (TAM) / top-right (Khalda) on the dashboard.
# If a file isn't found, that logo is simply skipped (dashboard still builds).
TAM_LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "tam_logo.png")
KHALDA_LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "khalda_logo.png")


# ============================================================
# (from notebook cell: ESP DATA PROCESSING FUNCTIONS)
# ============================================================
# ============================================================
# ESP DATA PROCESSING FUNCTIONS
# ============================================================
VX_THRESHOLD_G = 2.0          # vibration alert threshold
VX_GLITCH_CHECK_G = 5.5       # spikes at/above this need a nearby shutdown to be trusted
VX_GLITCH_WINDOW_BACK = pd.Timedelta(minutes=15)   # tolerance before the spike
VX_GLITCH_WINDOW_FWD = pd.Timedelta(hours=2)       # how long after the spike a shutdown still "counts"
PIP_RISE_THRESHOLD_PSI = 15   # sustained PIP increasing-trend threshold
IMPLAUSIBLE_TEMP_F = 50.0     # downhole temp reading below this = comm/scan glitch, not real
PIP_BASELINE_WINDOW = pd.Timedelta(minutes=60)  # window used to establish start/end PIP levels
TEMP_RISE_THRESHOLD_F = 5.0      # sustained motor-temp increasing-trend threshold
TEMP_BASELINE_WINDOW = pd.Timedelta(minutes=60)  # window used to establish start/end motor-temp levels
TEMP_DECLINE_TOLERANCE_F = 0.5   # if temp has cooled by at least this much from its recent level, treat it as recovering (not sustained)
VX_ACTIVE_THRESHOLD_G = 0.1   # fallback: Vx must be CHANGING and exceed this to count as active
VX_DOUBLE_RATIO = 2.0           # flag if current Vx is >= 2x its baseline
VX_DOUBLE_MIN_BASELINE_G = 0.05 # baseline must be at least this to avoid dividing by near-zero noise
VX_DOUBLE_BASELINE_WINDOW = pd.Timedelta(minutes=60)  # window used for start/7AM Vx levels, same idea as PIP_BASELINE_WINDOW

COL_TIMESTAMP = 'Timestamp'
COL_STATUS_MOTOR = 'Status-Motor[-]'
COL_STATUS_HZ = 'Status-Hz[Hz]'
COL_SETPOINT_FREQ = 'Setpoint-Operating Freq (Hz)[Hz]'
COL_PIP = 'Press-Pump Intake (Pi)[psi]'
COL_TEMP_INTAKE = 'Temp-Pump Intake[degF]'
COL_TEMP_MOTOR = 'Temp-Motor[degF]'
COL_VX = 'Vib-Pump X axis[G]'
COL_VY = 'Vib-Pump Y axis[G]'
COL_DISCHARGE = 'Press-Pump Discharge[psi]'
COL_MOTOR_AMPS = 'Pwr-Motor Amps Ph B[A]'
COL_OUTPUT_AMPS = 'Pwr-Output Amps Ph B[A]'
COL_SD_CURRENT = 'Shutdown-Current SD Reason[-]'
COL_SD_LAST = 'Shutdown-Last Shutdown Reason[-]'

NORMAL_REASONS = {'normal', 'no error', 'no fault', 'none', '', 'no'}

# Different wells/exports sometimes label the same run-state differently
# (e.g. 'Stop' vs 'Stopped'). Normalize every known variant to one of the
# two canonical values so shutdown detection, PIP filtering, etc. always
# match correctly regardless of which label a given well happens to use.
STATUS_MOTOR_ALIASES = {
    'Stop': 'Stopped', 'Stopped': 'Stopped', 'Off': 'Stopped',
    'Down': 'Stopped', 'Shutdown': 'Stopped', 'Shut Down': 'Stopped',
    'Idle': 'Stopped', 'Inactive': 'Stopped',
    'Run': 'Running', 'Running': 'Running', 'On': 'Running',
    'Active': 'Running', 'Started': 'Running',
}

# ------------------------------------------------------------------
# Shutdown reason normalization
# ------------------------------------------------------------------
# Several free-text reason strings from different VSD/SCADA setups all mean
# the same underlying event - collapse them to one canonical label.
REASON_ALIASES = {
    'Motor Current UL Alarm': 'Underload',
    'Underload SD Lockout': 'Underload',
    'VSD Underload': 'Underload',
    'Motor Current OL Alarm': 'Overload',
    'VSD overload': 'Overload',
    'MANUAL_OFF mode': 'Manual Stop',
    'Manual Keypad Stop Active': 'Manual Stop',
    'Stop operator': 'Manual Stop',
    'Manual Off': 'Manual Stop',
}
REASON_ALIASES_LOWER = {k.strip().lower(): v for k, v in REASON_ALIASES.items()}

# Some exports log the shutdown reason as a raw numeric VSD fault code
# instead of text. Map every known code to its human-readable name.
FAULT_CODE_TO_NAME = {}

# --- Drive faults (256-308) ---
FAULT_CODE_TO_NAME.update({
    256: 'No', 257: 'Power switch ph. U', 258: 'Power switch ph. V', 259: 'Power switch ph. W',
    260: 'Braking switch', 261: 'Overcurrent ph. U', 262: 'Overcurrent ph. V', 263: 'Overcurrent ph. W',
    264: 'DC undervoltage', 265: 'DC overvoltage', 266: 'DC not charging', 267: 'DC short circuit',
    268: 'SC to ground', 269: 'Overtorque', 270: 'PF overheated', 271: "BV6 isn't ready",
    272: 'IGBT U overheated', 273: 'IGBT V overheated', 274: 'IGBT W overheated',
    275: 'Rectifier overheated', 276: 'Phase loss mains 1', 277: 'Phase loss mains 3',
    278: 'Emergency stop Drive', 279: 'Inverter phasing', 280: 'Charge timeout',
    281: 'Connection failure', 282: 'Half-power contactor', 283: 'PF contactor',
    284: '50/60 Hz contactor', 285: 'Auto setting failure', 286: 'Phase loss mains 2',
    287: 'Work voltage', 288: 'SWF CT phasing', 289: 'Precharge contactor', 290: 'SUT saturation',
    291: 'VSD overload', 292: 'BVN connection', 293: 'BVN1 short circuit', 294: 'BVN1 not charging',
    295: 'Master BVN1 Umin', 296: 'Conn. Master-Slave in BVN1', 297: 'Slave BVN1 Umin',
    298: 'Conn. Slave-Master in BVN1', 299: 'BVN2 short circuit', 300: 'BVN2 not charging',
    301: 'Master BVN2 Umin', 302: 'Conn. Master-Slave in BVN2', 303: 'Slave BVN2 Umin',
    304: 'Conn. Slave-Master in BVN2', 305: "BVN isn't ready", 306: 'Overshoot',
    307: 'Phase loss mains 4', 308: 'Connected current sensor',
})

# --- Connection faults (512-530, 546) ---
FAULT_CODE_TO_NAME.update({
    512: 'No', 513: 'STM connection', 514: 'DME connection', 515: 'SCADA connection',
    516: 'SCADA2 connection', 517: 'RisS connection', 518: 'USP connection', 519: 'MUSP connection',
    520: 'AUSP_1 connection', 521: 'AUSP_2 connection', 522: 'DIN8DOUT4_1 connection',
    523: 'DIN8DOUT4_2 connection', 524: 'ADAM4017 connection', 525: 'ADAM4024_1 connection',
    526: 'ADAM4024_2 connection', 527: 'ADAM4055 connection', 528: 'ADAM4069 connection',
    529: 'ADC8_1 connection', 530: 'ADC8_2 connection', 546: 'Electricity meter connection',
})

# --- UMKA faults (768-790) ---
FAULT_CODE_TO_NAME.update({
    768: 'No', 769: 'Drive software incompatibility', 770: 'VSD configuration', 771: 'Type mains',
    772: 'UMKA temperature', 773: 'Insulation resistance', 774: 'Door',
    775: 'Passive filter overheated', 776: 'Cabinet temperature', 777: 'High line voltage',
    778: 'Low line voltage', 779: 'Voltage unbalance', 780: 'Overload', 781: 'Underload',
    782: 'Low frequency', 783: 'Phasing CBA', 784: 'Output currents unbalance',
    785: 'Operation mode mismatch', 786: 'Main frequency MAX', 787: 'Main frequency MIN',
    788: 'Backspin', 789: 'Coasting', 790: 'Power ON',
})

# --- Gauge faults (1024-1040) ---
FAULT_CODE_TO_NAME.update({
    1024: 'No', 1025: 'Motor oil temperature', 1026: 'Motor winding temperature',
    1027: 'Pump discharge temperature', 1028: 'Ambient temperature', 1029: 'Pump intake pressure',
    1030: 'Pump discharge pressure MAX', 1031: 'Pump discharge pressure MIN', 1032: 'Leakage current',
    1033: 'Flow MAX', 1034: 'Flow MIN', 1035: 'Vibration XY', 1036: 'Vibration Y',
    1037: 'Vibration Y', 1038: 'Vibration Z', 1039: 'Motor radial vibration speed',
    1040: 'Motor axial vibration speed',
})

# --- Discrete input faults (1280-1302) ---
FAULT_CODE_TO_NAME[1280] = 'No'
for _i in range(22):
    FAULT_CODE_TO_NAME[1281 + _i] = f'Digital input {_i}'

# --- Analog input faults (1536-1600): 16 inputs x 4 fault types ---
FAULT_CODE_TO_NAME[1536] = 'No'
for _i in range(16):
    _base = 1537 + _i * 4
    FAULT_CODE_TO_NAME[_base + 0] = f'Analog input {_i} MAX'
    FAULT_CODE_TO_NAME[_base + 1] = f'Analog input {_i} MIN'
    FAULT_CODE_TO_NAME[_base + 2] = f'Analog input {_i} sensor loss'
    FAULT_CODE_TO_NAME[_base + 3] = f'Analog input {_i} short circuit'

# --- Discrete outputs (1792-1808) ---
FAULT_CODE_TO_NAME[1792] = 'No'
for _i in range(16):
    FAULT_CODE_TO_NAME[1793 + _i] = f'Digital output {_i}'

# --- Starts (2560-2566) ---
FAULT_CODE_TO_NAME.update({
    2560: 'No', 2561: 'Start operator', 2562: 'Start SCADA', 2563: 'Start timer',
    2564: 'Start digital input', 2565: 'Start automatic restart', 2566: 'Start auto setting PMSM',
})

# --- Stops (2816-2821) ---
FAULT_CODE_TO_NAME.update({
    2816: 'No', 2817: 'Stop operator', 2818: 'Stop SCADA', 2819: 'Stop timer',
    2820: 'Stop digital input', 2821: 'Stop auto setting PMSM',
})


def normalize_shutdown_reason(raw):
    """
    Turns a raw shutdown-reason value into one canonical label:
      - A raw numeric VSD fault code (e.g. 291, or '291.0' from Excel) is
        replaced with its human-readable name (e.g. 'VSD overload').
      - Known free-text variants (e.g. 'Manual Keypad Stop Active',
        'MANUAL_OFF mode') are collapsed to one canonical phrase
        (e.g. 'Manual Stop').
      - Anything else is returned as-is (just whitespace-stripped).
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    s = str(raw).strip()
    if not s or s.lower() == 'nan':
        return None
    try:
        f = float(s)
        if f.is_integer() and int(f) in FAULT_CODE_TO_NAME:
            name = FAULT_CODE_TO_NAME[int(f)]
            # a numeric code can resolve to a name that's itself an alias
            # target (e.g. 2817 -> 'Stop operator' -> 'Manual Stop')
            return REASON_ALIASES_LOWER.get(name.lower(), name)
    except ValueError:
        pass
    return REASON_ALIASES_LOWER.get(s.lower(), s)


# ------------------------------------------------------------------
# Field "master list" detection (all-wells snapshot sheet)
# ------------------------------------------------------------------
# A zip can optionally also contain a single field-wide snapshot export
# (one row per well, e.g. "KHALDA_PETROLEUM_COMPANY_...xlsx" - the live
# PULSE/ProductionLink field list) alongside the normal per-well
# "ESP_<WellName>_..." time-series exports. When present, that sheet's
# "Well Name" column is the authoritative list of every well that should
# have sent an export - so any well on that list with no matching
# ESP_<WellName>_... file is a genuine "Miscommunication" well (no export
# received at all), and we can report it BY NAME instead of only as a count.
WELL_FILE_PATTERN = re.compile(r'^ESP_.+_\d{2}_\d{2}_\d{4}_')


def is_well_export_file(filepath):
    """True for a per-well time-series export (ESP_<WellName>_MM_DD_YYYY_...)."""
    return bool(WELL_FILE_PATTERN.match(os.path.basename(filepath)))


def is_master_list_file(filepath):
    """True for a field-wide, one-row-per-well snapshot sheet (has a
    'Well Name' column and isn't itself a per-well export)."""
    if is_well_export_file(filepath):
        return False
    try:
        head = pd.read_excel(filepath, nrows=0)
    except Exception:
        return False
    return 'Well Name' in head.columns


def find_master_list_file(folder):
    """Look through every .xlsx in `folder` for a master well-list sheet.
    Returns (filepath, sorted_well_names, full_dataframe) or
    (None, None, None) if no such sheet is present."""
    candidates = [f for f in glob.glob(os.path.join(folder, '*.xlsx'))
                  if not os.path.basename(f).startswith('~$')]
    for f in candidates:
        if not is_master_list_file(f):
            continue
        try:
            df = pd.read_excel(f)
        except Exception:
            continue
        if 'Well Name' not in df.columns:
            continue
        names = df['Well Name'].astype(str).str.strip()
        names = sorted({n for n in names if n and n.lower() != 'nan'})
        if names:
            return f, names, df
    return None, None, None


def extract_well_name(filepath):
    """ESP_<WellName>_MM_DD_YYYY_HH_MM_SS_ms_Egypt.xlsx -> WellName"""
    fname = os.path.basename(filepath)
    m = re.match(r'^ESP_(.+?)_\d{2}_\d{2}_\d{4}_', fname)
    if m:
        return m.group(1)
    # fallback: second underscore-separated token
    parts = fname.split('_')
    return parts[1] if len(parts) > 1 else os.path.splitext(fname)[0]


def get_col(df, name):
    if name in df.columns:
        return df[name]
    return pd.Series([np.nan] * len(df), index=df.index)


def clean_sensor_glitches(df):
    """
    Nulls out whole rows that are clearly a telemetry/comm glitch rather than
    a real reading: a downhole intake or motor temperature below
    IMPLAUSIBLE_TEMP_F (e.g. reading ~10 degF) is not physically possible for
    these wells. When that happens, PIP/discharge/vibration on the same row
    are unreliable too (they typically crash or spike at the same instant),
    so the entire row's sensor values are dropped rather than just the
    temperature - this is what lets a single-scan glitch (PIP diving to near
    0 alongside a bogus temperature) get excluded from every downstream
    calculation (PIP trend, vibration alerts, etc.), not just the PIP one.
    A genuine transient (e.g. a real PIP dip from a downhole restriction)
    never crashes the temperature reading at the same time, so this rule
    doesn't touch those.
    """
    sensor_cols = [c for c in
                   [COL_PIP, COL_DISCHARGE, COL_TEMP_INTAKE, COL_TEMP_MOTOR, COL_VX, COL_VY]
                   if c in df.columns]
    if not sensor_cols:
        return df

    glitch_mask = pd.Series(False, index=df.index)
    if COL_TEMP_INTAKE in df.columns:
        glitch_mask |= df[COL_TEMP_INTAKE] < IMPLAUSIBLE_TEMP_F
    if COL_TEMP_MOTOR in df.columns:
        glitch_mask |= df[COL_TEMP_MOTOR] < IMPLAUSIBLE_TEMP_F

    if glitch_mask.any():
        df.loc[glitch_mask, sensor_cols] = np.nan
    return df


def load_well_file(filepath):
    xl = pd.ExcelFile(filepath)
    sheet = 'Raw' if 'Raw' in xl.sheet_names else xl.sheet_names[0]
    df = pd.read_excel(filepath, sheet_name=sheet)
    df[COL_TIMESTAMP] = pd.to_datetime(df[COL_TIMESTAMP], errors='coerce')
    df = df.dropna(subset=[COL_TIMESTAMP]).sort_values(COL_TIMESTAMP).reset_index(drop=True)
    if COL_STATUS_MOTOR in df.columns:
        df[COL_STATUS_MOTOR] = df[COL_STATUS_MOTOR].astype(str).str.strip().str.title()
        df.loc[df[COL_STATUS_MOTOR].isin(['Nan', 'None']), COL_STATUS_MOTOR] = np.nan
        df[COL_STATUS_MOTOR] = df[COL_STATUS_MOTOR].apply(
            lambda v: STATUS_MOTOR_ALIASES.get(v, v) if pd.notna(v) else v
        )
    df = clean_sensor_glitches(df)
    return df


def determine_status(df):
    """
    Rule (as specified):
      - Take the last logged Freq value (Status-Hz, fallback Setpoint-Operating Freq).
      - If Freq is available: Freq > 0 => Running, Freq == 0 => Stopped.
      - If Freq is N/A (never logged - this is the case for wells with an N/A
        motor status): fall back to the Vib-Pump X series. The well is
        considered active (Running) only if Vx is actually CHANGING (not a
        stuck/flat reading) AND its max exceeds VX_ACTIVE_THRESHOLD_G (0.1G).
        Otherwise Stopped.
      - If neither Freq nor Vx ever logged => No Data.
    """
    freq_series = get_col(df, COL_STATUS_HZ).dropna()
    if freq_series.empty:
        freq_series = get_col(df, COL_SETPOINT_FREQ).dropna()

    if not freq_series.empty:
        last_freq = freq_series.iloc[-1]
        return ('Running' if last_freq > 0 else 'Stopped'), last_freq, None

    vx_series = get_col(df, COL_VX).dropna()
    if not vx_series.empty:
        last_vx = vx_series.iloc[-1]
        vx_is_changing = vx_series.nunique() > 1
        vx_active = vx_is_changing and (vx_series.max() > VX_ACTIVE_THRESHOLD_G)
        return ('Running' if vx_active else 'Stopped'), None, last_vx

    return 'No Data', None, None


def _change_events(df, col):
    """Return list of (timestamp, value) only where the value genuinely CHANGES
    from the previous logged value - filters out fields that persist/repeat a
    static 'last known' value on every row."""
    if col not in df.columns:
        return []
    sub = df[[COL_TIMESTAMP, col]].dropna(subset=[col]).sort_values(COL_TIMESTAMP)
    out = []
    prev = None
    for _, r in sub.iterrows():
        val = r[col]
        if val != prev:
            out.append((r[COL_TIMESTAMP], val))
        prev = val
    return out


def find_shutdown_events(df, well_name, file_end_time):
    """Group consecutive Status-Motor rows into blocks; for each Stopped block
    compute downtime + first associated shutdown reason."""
    events = []
    if COL_STATUS_MOTOR not in df.columns:
        return events

    status_rows = df[[COL_TIMESTAMP, COL_STATUS_MOTOR]].dropna(subset=[COL_STATUS_MOTOR]).reset_index(drop=True)
    if status_rows.empty:
        return events

    # Genuine reason "events": current SD reason entries (already discrete),
    # plus real changes in the Last Shutdown Reason field (which otherwise
    # persists/repeats a stale value on every row).
    current_events = [(t, normalize_shutdown_reason(v)) for t, v in _change_events(df, COL_SD_CURRENT)]
    current_events = [(t, v) for t, v in current_events if v and v.strip().lower() not in NORMAL_REASONS]
    last_events = [(t, normalize_shutdown_reason(v)) for t, v in _change_events(df, COL_SD_LAST)]
    last_events = [(t, v) for t, v in last_events if v and v.strip().lower() not in NORMAL_REASONS]

    # Raw (non-change-filtered) Last Shutdown Reason values, used as a
    # fallback below. COL_SD_LAST persists once set: if two consecutive
    # stops happen to share the same reason, the field never "changes" for
    # the second stop, so it produces no entry in last_events even though
    # the column is correctly showing that reason for the whole window.
    if COL_SD_LAST in df.columns:
        last_raw = df[[COL_TIMESTAMP, COL_SD_LAST]].dropna(subset=[COL_SD_LAST]).sort_values(COL_TIMESTAMP)
    else:
        last_raw = df.iloc[0:0]

    def find_reason(start, end):
        # Blocks are contiguous (end == next block's start), so only look
        # strictly within this block's own window - looking further forward
        # would leak the *next* event's reason into this one.
        for ts, v in current_events:
            if start <= ts <= end:
                return v
        for ts, v in last_events:
            if start <= ts <= end:
                return v
        # Fallback: read the raw column value directly within the window
        # (not requiring a "change") - catches back-to-back stops that
        # share the same reason, which the change-detection above misses.
        window = last_raw[(last_raw[COL_TIMESTAMP] >= start) & (last_raw[COL_TIMESTAMP] <= end)]
        for v in window[COL_SD_LAST]:
            norm = normalize_shutdown_reason(v)
            if norm and norm.strip().lower() not in NORMAL_REASONS:
                return norm
        return None

    # build contiguous status blocks
    blocks = []
    cur_status = status_rows.loc[0, COL_STATUS_MOTOR]
    cur_start = status_rows.loc[0, COL_TIMESTAMP]
    for i in range(1, len(status_rows)):
        s = status_rows.loc[i, COL_STATUS_MOTOR]
        t = status_rows.loc[i, COL_TIMESTAMP]
        if s != cur_status:
            blocks.append((cur_status, cur_start, t))
            cur_status = s
            cur_start = t
    # final block runs to end of file (or last timestamp if status never changes again)
    last_ts = max(status_rows[COL_TIMESTAMP].iloc[-1], file_end_time)
    blocks.append((cur_status, cur_start, last_ts))

    for status, start, end in blocks:
        if status != 'Stopped':
            continue
        downtime_hr = (end - start).total_seconds() / 3600.0
        reason = find_reason(start, end)
        is_ongoing = (end == file_end_time) and (end == status_rows[COL_TIMESTAMP].iloc[-1])
        events.append({
            'Well': well_name,
            'Shutdown Start': start,
            'Shutdown End': end,
            'Downtime (hrs)': round(downtime_hr, 2),
            'Reason': reason if reason else 'Not logged / Unknown',
            'Ongoing': is_ongoing,
        })
    return events


def find_vx_alerts(df, well_name, shutdown_starts):
    """
    Flags a well two ways (either one is enough to include it):

      1. EXISTING CONCEPT: any Vib-Pump X reading above VX_THRESHOLD_G.
         Sensor spikes at/above VX_GLITCH_CHECK_G (e.g. 5.5G) are treated as
         glitches - and excluded - unless a real shutdown started within a
         window around that spike. A high reading that never actually
         tripped the well is noise, not a genuine vibration event.

      2. NEW CONCEPT: vibration that has sustained-doubled (or more) from
         its baseline level, AND is still at/above double that baseline at
         the end of the reporting period (i.e. still doubled by 7 AM).
         Baseline = median Vx over the first VX_DOUBLE_BASELINE_WINDOW of
         the day; current level = median Vx over the last
         VX_DOUBLE_BASELINE_WINDOW (mirrors find_pip_trend's approach, so a
         one-off spike doesn't get mistaken for a sustained doubling). This
         catches a well trending toward failure even if it never crosses
         the absolute VX_THRESHOLD_G ceiling.
    """
    vx_raw = get_col(df, COL_VX)

    # ---- concept 1: existing absolute-threshold alert ----
    mask = vx_raw > VX_THRESHOLD_G
    sub_valid = pd.DataFrame(columns=[COL_TIMESTAMP, 'Vx'])
    if mask.any():
        sub = df.loc[mask, [COL_TIMESTAMP]].copy()
        sub['Vx'] = vx_raw[mask]

        def is_glitch(ts, val):
            if val < VX_GLITCH_CHECK_G:
                return False
            window_start = ts - VX_GLITCH_WINDOW_BACK
            window_end = ts + VX_GLITCH_WINDOW_FWD
            return not any(window_start <= s <= window_end for s in shutdown_starts)

        sub['is_glitch'] = [is_glitch(t, v) for t, v in zip(sub[COL_TIMESTAMP], sub['Vx'])]
        sub_valid = sub[~sub['is_glitch']]
    threshold_hit = not sub_valid.empty

    # ---- concept 2: NEW - sustained doubling, still doubled at 7 AM ----
    vx_valid = vx_raw[vx_raw.notna() & (vx_raw > 0)]
    ts = df.loc[vx_valid.index, COL_TIMESTAMP]
    baseline = current_level = None
    doubling_hit = False
    if len(vx_valid) >= 4:
        baseline_mask = ts <= (ts.iloc[0] + VX_DOUBLE_BASELINE_WINDOW)
        final_mask = ts >= (ts.iloc[-1] - VX_DOUBLE_BASELINE_WINDOW)
        baseline = vx_valid[baseline_mask].median()
        current_level = vx_valid[final_mask].median()
        if pd.notna(baseline) and baseline >= VX_DOUBLE_MIN_BASELINE_G and pd.notna(current_level):
            doubling_hit = current_level >= VX_DOUBLE_RATIO * baseline

    if not threshold_hit and not doubling_hit:
        return None

    if threshold_hit:
        max_row = sub_valid.loc[sub_valid['Vx'].idxmax()]
        max_vx, max_time, n_over = round(float(max_row['Vx']), 3), max_row[COL_TIMESTAMP], int(len(sub_valid))
    else:
        max_idx = vx_valid.idxmax()
        max_vx, max_time, n_over = round(float(vx_valid.max()), 3), df.loc[max_idx, COL_TIMESTAMP], 0

    return {
        'Well': well_name,
        'Max Vx (G)': max_vx,
        'Time of Max': max_time,
        'Readings > 2G': n_over,
        'Doubled & Still Doubled at 7AM': 'Yes' if doubling_hit else 'No',
        'Baseline Vx (G)': round(float(baseline), 3) if pd.notna(baseline) else None,
        'Current Vx (G)': round(float(current_level), 3) if pd.notna(current_level) else None,
    }


def find_pip_trend(df, well_name):
    """
    Rising-PIP detector, glitch- and noise-aware:
      - Any PIP reading of exactly 0 is treated as a sensor glitch and dropped
        (whole-row multi-sensor glitches are already dropped upstream by
        clean_sensor_glitches).
      - Rows logged while the pump is OFF (Status-Motor == 'Stopped') are
        dropped too, so a PIP climb that happens purely during a shutdown
        doesn't get counted as a genuine rising-PIP trend.
      - The rise is measured as a SUSTAINED shift: the median PIP over the
        first PIP_BASELINE_WINDOW of the day (the well's starting baseline)
        vs. the median PIP over the last PIP_BASELINE_WINDOW (its current,
        settled level). Using medians over a window - rather than a single
        instantaneous minimum/maximum - means a short-lived dip-then-recover
        fluctuation (common on some wells) does not get misread as a rising
        trend, since it washes out of the median once it's over.
    """
    if COL_STATUS_MOTOR in df.columns:
        # forward-fill so every row inherits the last-known run state
        running_mask = df[COL_STATUS_MOTOR].ffill() != 'Stopped'
    else:
        running_mask = pd.Series(True, index=df.index)

    pip_raw = get_col(df, COL_PIP)
    valid_mask = pip_raw.notna() & (pip_raw != 0) & running_mask
    pip = pip_raw[valid_mask]
    ts = df.loc[pip.index, COL_TIMESTAMP]

    if len(pip) < 4:
        return None

    baseline_mask = ts <= (ts.iloc[0] + PIP_BASELINE_WINDOW)
    final_mask = ts >= (ts.iloc[-1] - PIP_BASELINE_WINDOW)

    baseline = pip[baseline_mask].median()
    current_level = pip[final_mask].median()
    sustained_rise = current_level - baseline

    if pd.isna(sustained_rise) or sustained_rise <= PIP_RISE_THRESHOLD_PSI:
        return None

    peak_val = pip.max()
    peak_idx = pip.idxmax()

    return {
        'Well': well_name,
        'PIP-Yesterday 7AM (psi)': round(float(baseline), 1),
        'PIP-Today 7AM (psi)': round(float(current_level), 1),
        'Net Rise (psi)': round(float(sustained_rise), 1),
        'Peak PIP (psi)': round(float(peak_val), 1),
        'Peak Time': df.loc[peak_idx, COL_TIMESTAMP],
    }


def find_motor_temp_rise(df, well_name, shutdown_events=None):
    """
    Rising motor-temperature detector, sustained-trend approach:

      - Implausible readings (< IMPLAUSIBLE_TEMP_F, a comm/scan glitch) and
        rows logged while the pump is OFF are dropped, same as before.

      - RESTART-AWARE: if the well had a shutdown during the period, the
        comparison is anchored to the restart (the end of the LAST
        shutdown) instead of the start of the file. A restart naturally
        runs hot for a while as it comes back up to temperature, so
        comparing against the well's state from BEFORE the trip would
        misread a normal post-restart heat-up as a rising-temp alert.
        "Baseline" becomes the first TEMP_BASELINE_WINDOW after restart,
        and everything before the restart is ignored.

      - STILL RISING CHECK: temp is expected to climb for a while after any
        startup (field start OR restart) - that's normal, not a fault. So a
        rise is only flagged if it's STILL elevated/climbing at the end of
        the period. If the most recent window is already cooling back down
        compared to the window just before it (by more than
        TEMP_DECLINE_TOLERANCE_F), the well is treated as recovering, not
        as having a sustained problem, even if it's still above baseline.
    """
    if COL_STATUS_MOTOR in df.columns:
        running_mask = df[COL_STATUS_MOTOR].ffill() != 'Stopped'
    else:
        running_mask = pd.Series(True, index=df.index)

    temp_raw = get_col(df, COL_TEMP_MOTOR)
    valid_mask = temp_raw.notna() & (temp_raw >= IMPLAUSIBLE_TEMP_F) & running_mask

    # Anchor the whole comparison to the restart, if there was one.
    restart_time = None
    if shutdown_events:
        ends = [e['Shutdown End'] for e in shutdown_events if pd.notna(e.get('Shutdown End'))]
        restart_time = max(ends) if ends else None
    if restart_time is not None:
        valid_mask &= df[COL_TIMESTAMP] > restart_time

    temp = temp_raw[valid_mask]
    ts = df.loc[temp.index, COL_TIMESTAMP]

    if len(temp) < 4:
        return None

    baseline_mask = ts <= (ts.iloc[0] + TEMP_BASELINE_WINDOW)
    final_mask = ts >= (ts.iloc[-1] - TEMP_BASELINE_WINDOW)
    prior_mask = (ts >= (ts.iloc[-1] - 2 * TEMP_BASELINE_WINDOW)) & (ts < (ts.iloc[-1] - TEMP_BASELINE_WINDOW))

    baseline = temp[baseline_mask].median()
    current_level = temp[final_mask].median()
    prior_level = temp[prior_mask].median() if prior_mask.any() else None
    sustained_rise = current_level - baseline

    if pd.isna(sustained_rise) or sustained_rise <= TEMP_RISE_THRESHOLD_F:
        return None

    # Already cooling back down toward baseline (normal post-startup
    # settle) rather than still climbing/holding - don't flag it.
    if prior_level is not None and pd.notna(prior_level):
        if current_level <= prior_level - TEMP_DECLINE_TOLERANCE_F:
            return None

    peak_val = temp.max()
    peak_idx = temp.idxmax()

    return {
        'Well': well_name,
        'Baseline Temp (F)': round(float(baseline), 1),
        'Current Temp (F)': round(float(current_level), 1),
        'Rise (F)': round(float(sustained_rise), 1),
        'Peak Temp (F)': round(float(peak_val), 1),
        'Peak Time': df.loc[peak_idx, COL_TIMESTAMP],
        'Since Restart': 'Yes' if restart_time is not None else 'No',
    }


def process_well_file(filepath):
    well_name = extract_well_name(filepath)
    try:
        df = load_well_file(filepath)
    except Exception as e:
        return {
            'well_name': well_name, 'filepath': filepath, 'error': str(e),
            'status': 'No Data', 'shutdown_events': [], 'vx_alert': None, 'pip_trend': None, 'temp_rise': None,
        }

    if df.empty:
        return {
            'well_name': well_name, 'filepath': filepath, 'error': 'empty file',
            'status': 'No Data', 'shutdown_events': [], 'vx_alert': None, 'pip_trend': None, 'temp_rise': None,
        }

    file_end_time = df[COL_TIMESTAMP].max()
    status, last_freq, last_vx = determine_status(df)
    shutdown_events = find_shutdown_events(df, well_name, file_end_time)
    shutdown_starts = [e['Shutdown Start'] for e in shutdown_events]
    vx_alert = find_vx_alerts(df, well_name, shutdown_starts)
    pip_trend = find_pip_trend(df, well_name)
    temp_rise = find_motor_temp_rise(df, well_name, shutdown_events)

    return {
        'well_name': well_name,
        'filepath': filepath,
        'error': None,
        'status': status,
        'last_freq': last_freq,
        'last_vx': last_vx,
        'shutdown_events': shutdown_events,
        'vx_alert': vx_alert,
        'pip_trend': pip_trend,
        'temp_rise': temp_rise,
        'n_rows': len(df),
        'date_range': (df[COL_TIMESTAMP].min(), df[COL_TIMESTAMP].max()),
    }


def process_folder(folder):
    files = sorted(glob.glob(os.path.join(folder, '*.xlsx')))
    files = [f for f in files if not os.path.basename(f).startswith('~$')]

    # Pull out the field master-list sheet (if any) BEFORE processing, so
    # it never gets mistaken for a per-well time-series export (it has no
    # Timestamp column and would otherwise just show up as a read error).
    master_path, master_wells, master_df = find_master_list_file(folder)
    if master_path:
        print(f"Detected field master list: {os.path.basename(master_path)} "
              f"({len(master_wells)} wells listed)")
        files = [f for f in files if f != master_path]

    total = len(files)

    results = []
    for i, f in enumerate(files, 1):
        well_name = extract_well_name(f)
        sys.stdout.write(f"\rAnalyzing well files: {i}/{total}  ({well_name})" + " " * 20)
        sys.stdout.flush()
        results.append(process_well_file(f))
    if total:
        sys.stdout.write(f"\rAnalyzing well files: {total}/{total} complete." + " " * 30 + "\n")
        sys.stdout.flush()
    return results, master_wells, master_df

# ============================================================
# DASHBOARD BUILDING FUNCTIONS
# ============================================================
# ---------- PALETTE ----------
DARK_GREEN = '#05322B'
GREEN = '#018374'
LIGHT_GREEN = '#02BC94'
WHITE = '#FFFFFF'
NAVY = DARK_GREEN
SLATE = '#475569'
BG_PANEL = '#f8fafc'
GRID = '#e2e8f0'
RED = '#dc2626'
AMBER = '#d97706'
PURPLE = '#7c3aed'

plt.rcParams['font.family'] = 'DejaVu Sans'


def extract_zips_to_temp(all_wells_path, single_wells_path, tmp_root):
    """Extracts both the master 'all wells' zip and the 'single wells' zip 
    into a unified temporary folder so process_folder can read them all."""
    extract_dir = os.path.join(tmp_root, 'extracted')
    os.makedirs(extract_dir, exist_ok=True)
    
    def extract_and_flatten(path):
        if not path or not os.path.exists(path):
            return
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as z:
                for item in z.namelist():
                    # Filter for Excel files, ignoring hidden/system files
                    if item.lower().endswith('.xlsx') and not item.startswith('~') and '__MACOSX' not in item:
                        filename = os.path.basename(item)
                        if filename:  # Ignore directories
                            source = z.open(item)
                            target = open(os.path.join(extract_dir, filename), "wb")
                            with source, target:
                                shutil.copyfileobj(source, target)
                                
    # Extract both ZIPs to the same temporary location
    extract_and_flatten(all_wells_path)
    extract_and_flatten(single_wells_path)
    
    return extract_dir


def build_summary(results, total_wells_expected=None, miscommunication_wells=None, master_snapshot=None):
    files_found = len(results)
    found_names = {r['well_name'] for r in results}

    if miscommunication_wells is not None:
        # Preferred path: an actual field master-list sheet was found, so we
        # know exactly WHICH wells never sent an export (not just how many).
        miscommunication_wells = sorted(w for w in miscommunication_wells if w not in found_names)
        miscommunication = len(miscommunication_wells)
        total = files_found + miscommunication
    else:
        # Fallback (no master list available): the user tells us the true
        # well count for the field, and we only know the size of the gap,
        # not which wells it is - so no names can be listed.
        miscommunication_wells = []
        total = total_wells_expected if total_wells_expected is not None else files_found
        miscommunication = max(0, total - files_found)

    running = sum(1 for r in results if r['status'] == 'Running')
    stopped = sum(1 for r in results if r['status'] == 'Stopped')
    stopped_wells = [r['well_name'] for r in results if r['status'] == 'Stopped']
    no_data = sum(1 for r in results if r['status'] == 'No Data')
    no_data_wells = [r['well_name'] for r in results if r['status'] == 'No Data']

    shutdown_rows = []
    for r in results:
        shutdown_rows.extend(r['shutdown_events'])
    shutdown_df = pd.DataFrame(shutdown_rows)
    if not shutdown_df.empty:
        shutdown_df = shutdown_df.sort_values('Downtime (hrs)', ascending=False).reset_index(drop=True)

    vx_rows = [r['vx_alert'] for r in results if r['vx_alert']]
    vx_df = pd.DataFrame(vx_rows)
    if not vx_df.empty:
        vx_df = vx_df.sort_values('Max Vx (G)', ascending=False).reset_index(drop=True)

    # Only surface a rising-PIP trend for wells that had NO shutdown events
    # at all during the period - a well that tripped off is better explained
    # by the shutdown, so we don't also flag it as a rising-PIP well.
    pip_rows = [r['pip_trend'] for r in results if r['pip_trend'] and not r['shutdown_events']]
    pip_df = pd.DataFrame(pip_rows)
    if not pip_df.empty:
        pip_df = pip_df.sort_values('Net Rise (psi)', ascending=False).reset_index(drop=True)

    # Wells WITH shutdown events are still included here - find_motor_temp_rise
    # already re-anchors its baseline/current comparison to the restart (end
    # of the last shutdown) for those wells, rather than being excluded
    # outright like the PIP trend above.
    temp_rows = [r['temp_rise'] for r in results if r['temp_rise']]
    temp_df = pd.DataFrame(temp_rows)
    if not temp_df.empty:
        temp_df = temp_df.sort_values('Rise (F)', ascending=False).reset_index(drop=True)

    # shutdown count per well (for the bar chart)
    if not shutdown_df.empty:
        shutdown_count_df = (
            shutdown_df.groupby('Well').size().reset_index(name='Shutdown Count')
            .sort_values('Shutdown Count', ascending=False).reset_index(drop=True)
        )
        reason_counts = shutdown_df['Reason'].value_counts()
    else:
        shutdown_count_df = pd.DataFrame(columns=['Well', 'Shutdown Count'])
        reason_counts = pd.Series(dtype=int)

    well_status_rows = [{
        'Well': r['well_name'], 'Status': r['status'],
        'Last Freq (Hz)': r.get('last_freq'), 'Last Vx (G)': r.get('last_vx'),
        'Rows Logged': r.get('n_rows'), 'Error': r.get('error'),
    } for r in results]

    # Miscommunication wells never sent a file at all, so there's no
    # timeseries to summarize - but if the master list carries a live
    # snapshot for that well, surface it in the Error column so the note
    # isn't just a bare "no data" entry.
    for w in miscommunication_wells:
        note = 'No export file received for this well.'
        if master_snapshot is not None and 'Well Name' in master_snapshot.columns:
            match = master_snapshot[master_snapshot['Well Name'].astype(str).str.strip() == w]
            if not match.empty:
                mrow = match.iloc[0]
                note = (
                    f"No export file received - last known snapshot: "
                    f"Status={mrow.get('Status-Motor[-]', 'N/A')}, "
                    f"Freq={mrow.get('Status-Hz[Hz]', 'N/A')}Hz, "
                    f"PIP={mrow.get('Press-Pump Intake (Pi)[psi]', 'N/A')}psi"
                )
        well_status_rows.append({
            'Well': w, 'Status': 'Miscommunication',
            'Last Freq (Hz)': None, 'Last Vx (G)': None,
            'Rows Logged': 0, 'Error': note,
        })

    well_status_df = pd.DataFrame(well_status_rows)

    return {
        'total': total, 'files_found': files_found, 'miscommunication': miscommunication,
        'miscommunication_wells': miscommunication_wells,
        'running': running, 'stopped': stopped, 'stopped_wells': stopped_wells,
        'no_data': no_data, 'no_data_wells': no_data_wells,
        'well_status_df': well_status_df,
        'shutdown_df': shutdown_df, 'vx_df': vx_df, 'pip_df': pip_df, 'temp_df': temp_df,
        'shutdown_count_df': shutdown_count_df, 'reason_counts': reason_counts,
    }


def style_ax(ax, title):
    ax.set_facecolor(BG_PANEL)
    ax.set_title(title, fontsize=16, fontweight='bold', color=DARK_GREEN, pad=10, loc='left')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=SLATE, labelsize=11)
    ax.grid(axis='x', color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def draw_kpi_cards(fig, gs, summary, report_date):
    ax = fig.add_subplot(gs[0, :])
    ax.axis('off')

    kpi_values = [
        (str(summary['total']), 'TOTAL WELLS', DARK_GREEN),
        (str(summary['running']), 'ACTIVE / RUNNING', LIGHT_GREEN),
        (str(summary['stopped']), 'STOPPED', RED),
        (str(summary['miscommunication']), 'MISCOMMUNICATION', PURPLE),
    ]
    n = len(kpi_values)

    # Highlight callouts below the KPI cards: stopped wells (red) get top
    # billing since they need immediate attention, followed by any wells
    # with no power/status data (purple). Each note reserves its own row
    # of y < 0 space via ylim below, so it always renders fully inside
    # this axes' own gridspec cell instead of bleeding into (and getting
    # painted over by) the table axes underneath it.
    notes = []
    if summary['stopped'] > 0:
        stopped_wells = summary.get('stopped_wells', [])
        preview = ', '.join(map(str, stopped_wells[:10]))
        if len(stopped_wells) > 10:
            preview += f" (+{len(stopped_wells) - 10} more)"
        notes.append((f"Stopped Well(s) ({summary['stopped']}): {preview}", RED, '#fee2e2'))

    if summary['miscommunication'] > 0 and summary.get('miscommunication_wells'):
        mc_wells = summary['miscommunication_wells']
        preview = ', '.join(map(str, mc_wells[:10]))
        if len(mc_wells) > 10:
            preview += f" (+{len(mc_wells) - 10} more)"
        notes.append((f"Miscommunication - no data since last day ({summary['miscommunication']}): {preview}",
                      PURPLE, '#f3e8ff'))

    if summary['no_data'] > 0:
        preview = ', '.join(map(str, summary['no_data_wells'][:10]))
        if len(summary['no_data_wells']) > 10:
            preview += f" (+{len(summary['no_data_wells']) - 10} more)"
        notes.append((f"No Power/Status Data ({summary['no_data']}): {preview}", PURPLE, '#f3e8ff'))

    NOTE_ROW_H = 0.40
    ax.set_ylim(-NOTE_ROW_H * len(notes), 1)
    ax.set_xlim(0, n)
    card_w, gap = 0.88, 0.12

    for i, (value, label, accent) in enumerate(kpi_values):
        x0 = i * (card_w + gap) + 0.02
        card_color = DARK_GREEN if i == 0 else BG_PANEL
        number_color = WHITE if i == 0 else DARK_GREEN
        label_color = WHITE if i == 0 else SLATE

        ax.add_patch(FancyBboxPatch((x0, 0.08), card_w, 0.80,
                                     boxstyle="round,pad=0,rounding_size=0.06",
                                     linewidth=0, facecolor=card_color, zorder=1))
        ax.add_patch(FancyBboxPatch((x0, 0.08), 0.05, 0.80,
                                     boxstyle="round,pad=0,rounding_size=0.03",
                                     linewidth=0, facecolor=accent, zorder=2))
        ax.text(x0 + card_w / 2 + 0.02, 0.58, value, fontsize=38, fontweight='bold',
                color=number_color, ha='center', va='center', zorder=3)
        ax.text(x0 + card_w / 2 + 0.02, 0.24, label, fontsize=13.5, fontweight='bold',
                color=label_color, ha='center', va='center', zorder=3)

    for i, (text, color, bg) in enumerate(notes):
        note_y = -0.11 - i * NOTE_ROW_H
        ax.text(0.01, note_y, '\u26a0 ' + text,
                fontsize=12, fontweight='bold', color=color, ha='left', va='top',
                bbox=dict(facecolor=bg, edgecolor=color, boxstyle='round,pad=0.5'))


def draw_table(fig, gs_cell, df, columns, title, accent, empty_msg, max_rows=15, col_widths=None):
    ax = fig.add_subplot(gs_cell)
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold', color=DARK_GREEN, pad=10, loc='left')

    if df is None or df.empty:
        ax.text(0.02, 0.55, empty_msg, fontsize=13, color=SLATE, transform=ax.transAxes)
        return

    shown = df.head(max_rows).copy()
    cell_text = shown[columns].astype(str).values.tolist()
    n_rows = len(shown)

    row_h = 0.115
    table_h = min(0.90, row_h * (n_rows + 1))
    top, bottom = 0.90, 0.90 - table_h

    # Card background behind the table (soft border + shadow-like edge)
    # instead of a full grid, for a cleaner, modern look.
    ax.add_patch(FancyBboxPatch((0, bottom), 1, table_h,
                                 boxstyle="round,pad=0,rounding_size=0.012",
                                 transform=ax.transAxes, linewidth=1.1,
                                 edgecolor=GRID, facecolor=WHITE, zorder=0))

    tbl = ax.table(cellText=cell_text, colLabels=columns, cellLoc='center',
                    colWidths=col_widths, loc='upper left',
                    bbox=[0, bottom, 1, table_h])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)

    for (row, col), cell in tbl.get_celld().items():
        cell.PAD = 0.025
        if row == 0:
            # Header: solid accent bar, no border lines, bold white text
            cell.set_linewidth(0)
            cell.set_facecolor(accent)
            cell.set_text_props(color=WHITE, fontweight='bold')
        else:
            cell.set_facecolor(BG_PANEL if row % 2 == 0 else WHITE)
            cell.set_text_props(color=NAVY)
            # Only a thin bottom separator between rows - no vertical
            # grid lines - reads as a clean list rather than a spreadsheet.
            cell.visible_edges = 'B' if row < n_rows else ''
            cell.set_edgecolor(GRID)
            cell.set_linewidth(0.7)

    if len(df) > max_rows:
        ax.text(0.02, bottom - 0.06, f'+ {len(df) - max_rows} more \u2014 see full detail workbook',
                fontsize=10.5, color=SLATE, style='italic', transform=ax.transAxes)


def draw_shutdown_count_bar(fig, gs_cell, shutdown_count_df, max_wells=15):
    outer_ax = fig.add_subplot(gs_cell)
    outer_ax.axis('off')
    outer_ax.set_title('Total Shutdowns per Well', fontsize=16, fontweight='bold', color=DARK_GREEN, pad=10, loc='left')

    if shutdown_count_df is None or shutdown_count_df.empty:
        outer_ax.text(0.02, 0.5, 'No shutdown events logged for this period.',
                       fontsize=13, color=SLATE, transform=outer_ax.transAxes)
        return

    shown = shutdown_count_df.head(max_wells).iloc[::-1]

    # The chart lives in the right column of a nested grid, with a
    # reserved blank column on the left - this gives well-name tick labels
    # guaranteed room to render inside the panel instead of spilling past
    # its edge. (inset_axes doesn't reliably support categorical y-axes in
    # this matplotlib version, so a subgridspec column is used instead.)
    gs_inner = gs_cell.subgridspec(1, 2, width_ratios=[0.22, 0.78])
    ax = fig.add_subplot(gs_inner[0, 1])
    ax.set_facecolor(BG_PANEL)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=SLATE, labelsize=11)
    ax.set_axisbelow(True)
    ax.grid(axis='x', color=GRID, linewidth=0.8, zorder=0)
    ax.grid(axis='y', visible=False)

    green_shades = sns.light_palette(LIGHT_GREEN, n_colors=len(shown) + 2)[2:]
    ax.barh(shown['Well'].astype(str), shown['Shutdown Count'], color=green_shades,
            zorder=3, edgecolor='white', linewidth=1)
    for i, v in enumerate(shown['Shutdown Count'].values):
        ax.text(v + max(shown['Shutdown Count']) * 0.02, i, str(int(v)), va='center',
                fontweight='bold', fontsize=11.5, color=NAVY)
    ax.set_xlim(0, shown['Shutdown Count'].max() * 1.18)
    ax.set_xlabel('Number of Shutdowns', fontsize=12, color=SLATE)

    if len(shutdown_count_df) > max_wells:
        outer_ax.text(0.02, -0.06, f'+ {len(shutdown_count_df) - max_wells} more wells \u2014 see detail workbook',
                       fontsize=10.5, color=SLATE, style='italic', transform=outer_ax.transAxes)


def draw_motor_temp_bar(fig, gs_cell, temp_df, max_wells=15):
    """
    Stacked horizontal bar per well: the baseline (usual/normal) motor
    temperature in a neutral shade, with the sustained increase on top of
    it in a distinct warm color, so the "extra" portion driving the alert
    is visually separated from the well's normal operating temperature.
    """
    outer_ax = fig.add_subplot(gs_cell)
    outer_ax.axis('off')
    outer_ax.set_title(
        f'Sustained Motor Temp Increase (> {TEMP_RISE_THRESHOLD_F}\u00b0F, still up at 7AM)',
        fontsize=16, fontweight='bold', color=DARK_GREEN, pad=10, loc='left',
    )

    if temp_df is None or temp_df.empty:
        outer_ax.text(0.02, 0.5, f'No wells showed a sustained motor-temp rise greater than {TEMP_RISE_THRESHOLD_F}\u00b0F.',
                       fontsize=13, color=SLATE, transform=outer_ax.transAxes)
        return

    shown = temp_df.head(max_wells).iloc[::-1]

    gs_inner = gs_cell.subgridspec(1, 2, width_ratios=[0.22, 0.78])
    ax = fig.add_subplot(gs_inner[0, 1])
    ax.set_facecolor(BG_PANEL)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=SLATE, labelsize=11)
    ax.set_axisbelow(True)
    ax.grid(axis='x', color=GRID, linewidth=0.8, zorder=0)
    ax.grid(axis='y', visible=False)

    wells = shown['Well'].astype(str)
    baseline = shown['Baseline Temp (F)']
    rise = shown['Rise (F)']

    # Normal/usual portion (baseline temp) vs. the increase, in a clearly
    # different color from the "normal" shade used everywhere else on the
    # dashboard.
    ax.barh(wells, baseline, color=SLATE, alpha=0.35, zorder=3, edgecolor='white', linewidth=1, label='Baseline (normal) temp')
    ax.barh(wells, rise, left=baseline, color=RED, zorder=3, edgecolor='white', linewidth=1, label='Sustained increase')

    for i, (b, r) in enumerate(zip(baseline.values, rise.values)):
        ax.text(b + r + max(baseline + rise) * 0.015, i, f'+{r:.1f}\u00b0F', va='center',
                fontweight='bold', fontsize=11.5, color=RED)

    ax.set_xlim(0, (baseline + rise).max() * 1.18)
    ax.set_xlabel('Motor Temperature (\u00b0F)', fontsize=12, color=SLATE)
    ax.legend(loc='lower right', fontsize=10, frameon=False, labelcolor=SLATE)

    if len(temp_df) > max_wells:
        outer_ax.text(0.02, -0.06, f'+ {len(temp_df) - max_wells} more wells \u2014 see detail workbook',
                       fontsize=10.5, color=SLATE, style='italic', transform=outer_ax.transAxes)


def draw_reason_pie(fig, gs_cell, reason_counts, max_slices=8):
    ax = fig.add_subplot(gs_cell)
    ax.axis('off')
    ax.set_title('Shutdown Reasons \u2014 All Wells', fontsize=16, fontweight='bold',
                  color=DARK_GREEN, pad=10, loc='left')

    if reason_counts is None or len(reason_counts) == 0:
        ax.text(0.02, 0.5, 'No shutdown events logged for this period.',
                fontsize=13, color=SLATE, transform=ax.transAxes)
        return

    counts = reason_counts.copy()
    if len(counts) > max_slices:
        top = counts.iloc[:max_slices - 1]
        other = pd.Series({'Other reasons': counts.iloc[max_slices - 1:].sum()})
        counts = pd.concat([top, other])

    palette = [DARK_GREEN, GREEN, LIGHT_GREEN, AMBER, RED, PURPLE, SLATE, '#0891b2', '#be185d']
    colors = [palette[i % len(palette)] for i in range(len(counts))]

    wedges, _, autotexts = ax.pie(
        counts.values, colors=colors, autopct=lambda p: f'{p:.0f}%' if p >= 4 else '',
        startangle=90, pctdistance=0.78, radius=0.95,
        wedgeprops=dict(edgecolor='white', linewidth=1.5),
        textprops=dict(color='white', fontweight='bold', fontsize=10.5),
    )
    ax.legend(
        wedges, [f'{name}  ({int(val)})' for name, val in zip(counts.index, counts.values)],
        loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=11, frameon=False,
        labelcolor=SLATE,
    )


def draw_logos(fig):
    """Places the TAM logo top-left and the Khalda logo top-right.
    Silently skips a logo if its file isn't found on disk."""
    if TAM_LOGO_PATH and os.path.isfile(TAM_LOGO_PATH):
        try:
            tam_logo = mpimg.imread(TAM_LOGO_PATH)
            ax_logo_left = fig.add_axes([0.06, 0.865, 0.165, 0.095])
            ax_logo_left.imshow(tam_logo)
            ax_logo_left.axis('off')
        except Exception as e:
            print(f'Could not load TAM logo ({TAM_LOGO_PATH}): {e}')

    if KHALDA_LOGO_PATH and os.path.isfile(KHALDA_LOGO_PATH):
        try:
            khalda_logo = mpimg.imread(KHALDA_LOGO_PATH)
            ax_logo_right = fig.add_axes([0.75, 0.865, 0.165, 0.095])
            ax_logo_right.imshow(khalda_logo)
            ax_logo_right.axis('off')
        except Exception as e:
            print(f'Could not load Khalda logo ({KHALDA_LOGO_PATH}): {e}')


def compute_row_height_ratios(summary):
    """
    Sizes each content row (KPI cards / shutdown table / bar+pie / vx+PIP
    tables) according to how many rows it actually has to show, instead of
    always reserving space for the maximum possible (20/15/12 rows). A
    report with few events ends up as a shorter, tighter page instead of
    mostly blank space; a report with lots of events still grows up to the
    same maximum size as before.
    """
    n_sd = min(len(summary['shutdown_df']), 20) if not summary['shutdown_df'].empty else 0
    n_bar = min(len(summary['shutdown_count_df']), 15) if not summary['shutdown_count_df'].empty else 0
    n_vx = min(len(summary['vx_df']), 12) if not summary['vx_df'].empty else 0
    n_pip = min(len(summary['pip_df']), 12) if not summary['pip_df'].empty else 0
    n_temp = min(len(summary['temp_df']), 15) if not summary['temp_df'].empty else 0
    n_detail = max(n_vx, n_pip)

    # Extra room under the KPI row for each stacked highlight note (stopped
    # wells / no-data wells) - keeps draw_kpi_cards' NOTE_ROW_H reservation
    # from being squeezed.
    note_lines = (
        (1 if summary.get('stopped', 0) > 0 else 0)
        + (1 if summary.get('miscommunication', 0) > 0 and summary.get('miscommunication_wells') else 0)
        + (1 if summary.get('no_data', 0) > 0 else 0)
    )
    ratio_kpi = 0.42 + note_lines * 0.17
    ratio_shutdown = min(1.35, max(0.35, 0.30 + 0.0525 * n_sd))
    ratio_bar_pie = min(1.00, max(0.55, 0.50 + 0.0333 * n_bar))
    ratio_detail = min(1.05, max(0.40, 0.30 + 0.0625 * n_detail))
    ratio_temp = min(1.00, max(0.35, 0.30 + 0.0333 * n_temp))
    return [ratio_kpi, ratio_shutdown, ratio_bar_pie, ratio_detail, ratio_temp]


# Constants tuned so a "full" report (20/15/12 rows) reproduces the original,
# comfortably-spaced layout: at max ratios (sum = 3.82) this yields the same
# ~19x30in page as before.
INCHES_PER_RATIO_UNIT = 6.126
GRIDSPEC_TOP = 0.835
GRIDSPEC_BOTTOM = 0.055
GRIDSPEC_FRACTION = GRIDSPEC_TOP - GRIDSPEC_BOTTOM
MIN_FIGURE_HEIGHT_IN = 16.0


def draw_section_dividers(fig, gs):
    """
    Thin light-green dotted divider lines between each dashboard section,
    with added vertical spacing before and after the line.
    """
    bottoms, tops, lefts, rights = gs.get_grid_positions(fig)
    x0, x1 = lefts[0], rights[-1]

    for row in range(len(tops) - 1):
        # Calculate the exact midpoint between sections
        y_mid = (bottoms[row] + tops[row + 1]) / 2.0
        
        # Add spacing offset (e.g., 0.008 figure units of padding above and below)
        padding = 0.1

        # Draw a top separator marker/line for spacing
        fig.add_artist(Line2D([x0, x1], [y_mid + padding, y_mid + padding], transform=fig.transFigure,
                               color='none', linewidth=0, zorder=6))

        # Main dotted divider line
        fig.add_artist(Line2D([x0, x1], [y_mid], transform=fig.transFigure,
                               color=LIGHT_GREEN, linewidth=1.5, alpha=0.8,
                               linestyle=':', zorder=6))
                               
        # Draw a bottom separator marker/line for spacing
        fig.add_artist(Line2D([x0, x1], [y_mid - padding, y_mid - padding], transform=fig.transFigure,
                               color='none', linewidth=0, zorder=6))


def build_dashboard_figure(summary, report_date, output_png):
    height_ratios = compute_row_height_ratios(summary)
    gridspec_in = sum(height_ratios) * INCHES_PER_RATIO_UNIT
    fig_height = max(MIN_FIGURE_HEIGHT_IN, gridspec_in / GRIDSPEC_FRACTION)

    fig = plt.figure(figsize=(19, fig_height), facecolor='white')

    gs = fig.add_gridspec(5, 2, height_ratios=height_ratios, hspace=0.45, wspace=0.18,
                           left=0.14, right=0.86, top=GRIDSPEC_TOP, bottom=GRIDSPEC_BOTTOM)

    draw_section_dividers(fig, gs)

    fig.text(0.5, 0.945, 'Daily ESP Surveillance Summary', fontsize=32, fontweight='bold',
              color=DARK_GREEN, ha='center')
    fig.text(0.5, 0.895, f'Field Summary  \u2022  {summary["total"]} Wells  \u2022  {report_date}',
              fontsize=16, color=GREEN, ha='center')

    # ---------- LOGOS ----------
    draw_logos(fig)

    draw_kpi_cards(fig, gs, summary, report_date)

    draw_table(
        fig, gs[1, :], summary['shutdown_df'],
        columns=['Well', 'Reason', 'Shutdown Start', 'Shutdown End', 'Downtime (hrs)'],
        title=f'Shutdown Events & Downtime (sorted by downtime)',
        accent=LIGHT_GREEN, empty_msg='No shutdown events logged for this period.',
        max_rows=20, col_widths=[0.14, 0.34, 0.18, 0.18, 0.16],
    )

    draw_shutdown_count_bar(fig, gs[2, 0], summary['shutdown_count_df'])
    draw_reason_pie(fig, gs[2, 1], summary['reason_counts'])

    draw_table(
        fig, gs[3, 0], summary['vx_df'],
        columns=['Well', 'Max Vx (G)', 'Readings > 2G', 'Time of Max', 'Doubled & Still Doubled at 7AM'],
        title=f'High Vibration Alerts (Vx > {VX_THRESHOLD_G}G or doubled & still doubled at 7AM)',
        accent=LIGHT_GREEN, empty_msg=f'No wells exceeded {VX_THRESHOLD_G}G or showed a sustained doubling.',
        max_rows=12, col_widths=[0.22, 0.16, 0.16, 0.22, 0.24],
    )

    pip_df = summary['pip_df']
    pip_cols = ['Well', 'PIP-Yesterday 7AM (psi)', 'PIP-Today 7AM (psi)', 'Net Rise (psi)'] if not pip_df.empty else []
    draw_table(
        fig, gs[3, 1], pip_df,
        columns=pip_cols,
        title=f'Rising PIP Trends (> {PIP_RISE_THRESHOLD_PSI} psi sustained, no shutdowns)',
        accent=LIGHT_GREEN, empty_msg=f'No wells showed a sustained PIP rise greater than {PIP_RISE_THRESHOLD_PSI} psi.',
        max_rows=12, col_widths=[0.18, 0.34, 0.29, 0.19],
    )

    fig.text(0.08, 0.03, f'Generated {datetime.now().strftime("%d-%b-%Y %H:%M")}  \u2022  Generated by ProductionLink Team at TAM OIL',
              fontsize=11, color=SLATE, ha='left', style='italic')

    draw_motor_temp_bar(fig, gs[4, :], summary['temp_df'])

    plt.savefig(output_png, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def export_excel(summary, results, output_xlsx):
    with pd.ExcelWriter(output_xlsx, engine='openpyxl') as writer:
        summary['well_status_df'].to_excel(writer, sheet_name='Well Status', index=False)
        (summary['shutdown_df'] if not summary['shutdown_df'].empty else pd.DataFrame(
            columns=['Well', 'Reason', 'Shutdown Start', 'Shutdown End', 'Downtime (hrs)', 'Ongoing']
        )).to_excel(writer, sheet_name='Shutdown Events', index=False)
        (summary['vx_df'] if not summary['vx_df'].empty else pd.DataFrame(
            columns=['Well', 'Max Vx (G)', 'Time of Max', 'Readings > 2G',
                     'Doubled & Still Doubled at 7AM', 'Baseline Vx (G)', 'Current Vx (G)']
        )).to_excel(writer, sheet_name='High Vibration (Vx gt 2G)', index=False)
        (summary['pip_df'] if not summary['pip_df'].empty else pd.DataFrame(
            columns=['Well', 'PIP-Yesterday 7AM (psi)', 'PIP-Today 7AM (psi)', 'Net Rise (psi)',
                     'Peak PIP (psi)', 'Peak Time']
        )).to_excel(writer, sheet_name='PIP Rising Trend', index=False)
        (summary['temp_df'] if not summary['temp_df'].empty else pd.DataFrame(
            columns=['Well', 'Baseline Temp (F)', 'Current Temp (F)', 'Rise (F)', 'Peak Temp (F)',
                     'Peak Time', 'Since Restart']
        )).to_excel(writer, sheet_name='Motor Temp Rising Trend', index=False)
        (summary['shutdown_count_df'] if not summary['shutdown_count_df'].empty else pd.DataFrame(
            columns=['Well', 'Shutdown Count']
        )).to_excel(writer, sheet_name='Shutdown Count per Well', index=False)
        if len(summary['reason_counts']) > 0:
            summary['reason_counts'].rename_axis('Reason').reset_index(name='Count').to_excel(
                writer, sheet_name='Shutdown Reasons (All Wells)', index=False)

        errors = [(r['well_name'], r['error']) for r in results if r.get('error')]
        if errors:
            pd.DataFrame(errors, columns=['Well', 'Error']).to_excel(writer, sheet_name='Read Errors', index=False)

        mc_wells = summary.get('miscommunication_wells') or []
        if mc_wells:
            mc_df = summary['well_status_df'][summary['well_status_df']['Status'] == 'Miscommunication'][
                ['Well', 'Error']
            ].rename(columns={'Error': 'Last Known Snapshot / Note'})
            mc_df.to_excel(writer, sheet_name='Miscommunication Wells', index=False)


def build_whatsapp_message(summary, report_date):
    """
    Builds a clean, consistently formatted WhatsApp status and summary message 
    without emojis, including specific stopped and miscommunicated wells.
    """
    total = summary['total']
    files_found = summary['files_found']
    running = summary['running']
    stopped = summary['stopped']
    no_data = summary['no_data']
    miscomm = summary['miscommunication']
    
    shutdown_df = summary['shutdown_df']
    vx_df = summary['vx_df']
    pip_df = summary['pip_df']
    temp_df = summary['temp_df']
    
    stopped_wells = summary.get('stopped_wells', [])
    miscomm_wells = summary.get('miscommunication_wells', [])
    
    # Header Section
    msg = [
        "*Daily ESP Surveillance Summary*",
        f"{report_date}",
        "----------------------------------------",
        f"Total Wells Tracked: {total}",
        f"Running: {running}",
        f"Stopped: {stopped}",
        f"Miscommunication: {miscomm}",
        "----------------------------------------",
        "*KEY OPERATIONAL ALERTS*"
    ]
    
    # Stopped Wells Detailed Section
    if stopped > 0:
        wells_str = ", ".join(map(str, stopped_wells))
        msg.append(f"\n*Stopped Wells* ({stopped}): {wells_str}")
    else:
        msg.append("\nStopped Wells: None recorded")

    # Miscommunication Wells Detailed Section
    if miscomm > 0:
        mc_str = ", ".join(map(str, miscomm_wells))
        msg.append(f"\n*Miscommunication Wells* ({miscomm}): {mc_str}")
    else:
        msg.append("\n*Miscommunication Wells:* None recorded")

    # Shutdown Events Section
    if not shutdown_df.empty:
        msg.append(f"\n*Shutdown Events* ({len(shutdown_df)} total):")
        for _, row in shutdown_df.head(5).iterrows():
            msg.append(f"   • {row['Well']}: {row['Downtime (hrs)']} hrs ({row['Reason']})")
        if len(shutdown_df) > 5:
            msg.append(f"   _...and {len(shutdown_df) - 5} more in the report._")
    else:
        msg.append("\n*Shutdown Events:* None recorded")

    # Vibration Alerts Section
    if not vx_df.empty:
        msg.append(f"\n*Vibration Alerts* > {VX_THRESHOLD_G}G or doubled & still doubled at 7AM ({len(vx_df)} wells):")
        for _, row in vx_df.head(5).iterrows():
            flag = " \u26A0\uFE0F doubled & still doubled at 7AM" if row.get('Doubled & Still Doubled at 7AM') == 'Yes' else ""
            msg.append(f"   • {row['Well']}: Max {row['Max Vx (G)']} G{flag}")
    else:
        msg.append(f"\n*Vibration Alerts:* None > {VX_THRESHOLD_G}G, none doubled & still doubled at 7AM")

    # Rising PIP Section
    if not pip_df.empty:
        msg.append(f"\n*Rising PIP Trends* ({len(pip_df)} wells):")
        for _, row in pip_df.head(5).iterrows():
            msg.append(f"   • {row['Well']}: +{row['Net Rise (psi)']} psi")
    else:
        msg.append("\nRising PIP Trends: None flagged")

    # Rising Motor Temp Section
    if not temp_df.empty:
        msg.append(f"\n*Sustained Motor Temp Increase* > {TEMP_RISE_THRESHOLD_F}\u00b0F, still up at 7AM ({len(temp_df)} wells):")
        for _, row in temp_df.head(5).iterrows():
            msg.append(f"   • {row['Well']}: +{row['Rise (F)']}\u00b0F (now {row['Current Temp (F)']}\u00b0F)")
    else:
        msg.append(f"\nSustained Motor Temp Increase: None > {TEMP_RISE_THRESHOLD_F}\u00b0F")

    # Footer
    msg.extend([
        "----------------------------------------",
        "_Dashboard PNG, Detail Excel Workbook, and text log attached below._"
    ])
    
    return "\n".join(msg)


# ============================================================
# MOBILE (WHATSAPP) DASHBOARD BUILDING FUNCTIONS
# ============================================================
# Single-column, portrait-oriented version of the dashboard, sized so
# that when WhatsApp shrinks it to chat-bubble width, text is still
# legible without tapping to zoom. Font sizes here are deliberately
# bumped above the desktop build's, and the components are mobile-only
# copies (not shared with draw_table/draw_shutdown_count_bar/draw_reason_pie)
# so tuning one doesn't affect the other.

MOBILE_FIG_WIDTH_IN = 10.5
MOBILE_LEFT, MOBILE_RIGHT = 0.09, 0.91
MOBILE_INCHES_PER_RATIO_UNIT = 6.126
MOBILE_TOP = 0.93
MOBILE_BOTTOM = 0.035
MOBILE_MIN_HEIGHT_IN = 14.0


def _shorten_dt(val):
    """'2026-08-16 10:30:00' -> '16-Aug 10:30' so the mobile shutdown
    table doesn't burn its whole width on full timestamps."""
    try:
        return pd.to_datetime(val).strftime('%d-%b %H:%M')
    except Exception:
        return str(val)


def draw_kpi_cards_mobile(fig, gs_cell, summary):
    """Same 4 KPI numbers as the desktop version, in one compact row
    (not stacked 2x2) so the row stays short and doesn't eat vertical
    space that could go to the report content below it."""
    ax = fig.add_subplot(gs_cell)
    ax.axis('off')

    kpi_values = [
        (str(summary['total']), 'TOTAL WELLS', DARK_GREEN),
        (str(summary['running']), 'ACTIVE / RUNNING', LIGHT_GREEN),
        (str(summary['stopped']), 'STOPPED', RED),
        (str(summary['miscommunication']), 'MISCOMMUNICATION', PURPLE),
    ]
    n = len(kpi_values)

    notes = []
    if summary['stopped'] > 0:
        stopped_wells = summary.get('stopped_wells', [])
        preview = ', '.join(map(str, stopped_wells[:8]))
        if len(stopped_wells) > 8:
            preview += f" (+{len(stopped_wells) - 8} more)"
        notes.append((f"Stopped Well(s) ({summary['stopped']}): {preview}", RED, '#fee2e2'))

    if summary['miscommunication'] > 0 and summary.get('miscommunication_wells'):
        mc_wells = summary['miscommunication_wells']
        preview = ', '.join(map(str, mc_wells[:8]))
        if len(mc_wells) > 8:
            preview += f" (+{len(mc_wells) - 8} more)"
        notes.append((f"Miscommunication - no data since last day ({summary['miscommunication']}): {preview}",
                      PURPLE, '#f3e8ff'))

    if summary['no_data'] > 0:
        preview = ', '.join(map(str, summary['no_data_wells'][:8]))
        if len(summary['no_data_wells']) > 8:
            preview += f" (+{len(summary['no_data_wells']) - 8} more)"
        notes.append((f"No Power/Status Data ({summary['no_data']}): {preview}", PURPLE, '#f3e8ff'))

    NOTE_ROW_H = 0.40
    ax.set_ylim(-NOTE_ROW_H * len(notes), 1)
    ax.set_xlim(0, n)
    card_w, gap = 0.88, 0.12

    for i, (value, label, accent) in enumerate(kpi_values):
        x0 = i * (card_w + gap) + 0.02
        card_color = DARK_GREEN if i == 0 else BG_PANEL
        number_color = WHITE if i == 0 else DARK_GREEN
        label_color = WHITE if i == 0 else SLATE

        ax.add_patch(FancyBboxPatch((x0, 0.08), card_w, 0.80,
                                     boxstyle="round,pad=0,rounding_size=0.06",
                                     linewidth=0, facecolor=card_color, zorder=1))
        ax.add_patch(FancyBboxPatch((x0, 0.08), 0.05, 0.80,
                                     boxstyle="round,pad=0,rounding_size=0.03",
                                     linewidth=0, facecolor=accent, zorder=2))
        ax.text(x0 + card_w / 2 + 0.02, 0.56, value, fontsize=27, fontweight='bold',
                color=number_color, ha='center', va='center', zorder=3)
        ax.text(x0 + card_w / 2 + 0.02, 0.23, label, fontsize=9.7, fontweight='bold',
                color=label_color, ha='center', va='center', zorder=3)

    for i, (text, color, bg) in enumerate(notes):
        note_y = -0.11 - i * NOTE_ROW_H
        ax.text(0.01, note_y, '\u26a0 ' + text,
                fontsize=14, fontweight='bold', color=color, ha='left', va='top',
                bbox=dict(facecolor=bg, edgecolor=color, boxstyle='round,pad=0.5'),
                wrap=True)


def style_ax_mobile(ax, title):
    ax.set_facecolor(BG_PANEL)
    ax.set_title(title, fontsize=19, fontweight='bold', color=DARK_GREEN, pad=14, loc='left')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=SLATE, labelsize=13)
    ax.grid(axis='x', color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def draw_table_mobile(fig, gs_cell, df, columns, title, accent, empty_msg, max_rows=15, col_widths=None):
    ax = fig.add_subplot(gs_cell)
    ax.axis('off')
    ax.set_title(title, fontsize=19, fontweight='bold', color=DARK_GREEN, pad=14, loc='left')

    if df is None or df.empty:
        ax.text(0.02, 0.55, empty_msg, fontsize=15, color=SLATE, transform=ax.transAxes)
        return

    shown = df.head(max_rows).copy()
    cell_text = shown[columns].astype(str).values.tolist()
    n_rows = len(shown)

    row_h = 0.16
    table_h = min(0.90, row_h * (n_rows + 1))
    top, bottom = 0.90, 0.90 - table_h

    # Card background behind the table (soft border) instead of a full
    # grid, for a cleaner, modern look that matches the desktop version.
    ax.add_patch(FancyBboxPatch((0, bottom), 1, table_h,
                                 boxstyle="round,pad=0,rounding_size=0.012",
                                 transform=ax.transAxes, linewidth=1.1,
                                 edgecolor=GRID, facecolor=WHITE, zorder=0))

    tbl = ax.table(cellText=cell_text, colLabels=columns, cellLoc='center',
                    colWidths=col_widths, loc='upper left',
                    bbox=[0, bottom, 1, table_h])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(13.5)

    for (row, col), cell in tbl.get_celld().items():
        cell.PAD = 0.03
        if row == 0:
            # Header: solid accent bar, no border lines, bold white text
            cell.set_linewidth(0)
            cell.set_facecolor(accent)
            cell.set_text_props(color=WHITE, fontweight='bold')
        else:
            cell.set_facecolor(BG_PANEL if row % 2 == 0 else WHITE)
            cell.set_text_props(color=NAVY)
            # Only a thin bottom separator between rows - no vertical
            # grid lines - reads as a clean list rather than a spreadsheet.
            cell.visible_edges = 'B' if row < n_rows else ''
            cell.set_edgecolor(GRID)
            cell.set_linewidth(0.7)

    if len(df) > max_rows:
        ax.text(0.02, bottom - 0.05, f'+ {len(df) - max_rows} more \u2014 see full detail workbook',
                fontsize=12, color=SLATE, style='italic', transform=ax.transAxes)


def draw_shutdown_count_bar_mobile(fig, gs_cell, shutdown_count_df, max_wells=15):
    outer_ax = fig.add_subplot(gs_cell)
    outer_ax.axis('off')
    outer_ax.set_title('Total Shutdowns per Well', fontsize=19, fontweight='bold', color=DARK_GREEN, pad=14, loc='left')

    if shutdown_count_df is None or shutdown_count_df.empty:
        outer_ax.text(0.02, 0.5, 'No shutdown events logged for this period.',
                       fontsize=15, color=SLATE, transform=outer_ax.transAxes)
        return

    shown = shutdown_count_df.head(max_wells).iloc[::-1]

    # The chart lives in the right column of a nested grid, with a
    # reserved blank column on the left - this gives well-name tick labels
    # guaranteed room to render inside the panel instead of spilling past
    # its edge. (inset_axes doesn't reliably support categorical y-axes in
    # this matplotlib version, so a subgridspec column is used instead.)
    gs_inner = gs_cell.subgridspec(1, 2, width_ratios=[0.28, 0.72])
    ax = fig.add_subplot(gs_inner[0, 1])
    ax.set_facecolor(BG_PANEL)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=SLATE, labelsize=13)
    ax.set_axisbelow(True)
    ax.grid(axis='x', color=GRID, linewidth=0.8, zorder=0)
    ax.grid(axis='y', visible=False)

    green_shades = sns.light_palette(LIGHT_GREEN, n_colors=len(shown) + 2)[2:]
    ax.barh(shown['Well'].astype(str), shown['Shutdown Count'], color=green_shades,
            zorder=3, edgecolor='white', linewidth=1)
    for i, v in enumerate(shown['Shutdown Count'].values):
        ax.text(v + max(shown['Shutdown Count']) * 0.02, i, str(int(v)), va='center',
                fontweight='bold', fontsize=14, color=NAVY)
    ax.set_xlim(0, shown['Shutdown Count'].max() * 1.18)
    ax.set_xlabel('Number of Shutdowns', fontsize=14, color=SLATE)

    if len(shutdown_count_df) > max_wells:
        outer_ax.text(0.02, -0.06, f'+ {len(shutdown_count_df) - max_wells} more wells \u2014 see detail workbook',
                       fontsize=12, color=SLATE, style='italic', transform=outer_ax.transAxes)


def draw_motor_temp_bar_mobile(fig, gs_cell, temp_df, max_wells=15):
    outer_ax = fig.add_subplot(gs_cell)
    outer_ax.axis('off')
    outer_ax.set_title(
        f'Motor Temp Increase (> {TEMP_RISE_THRESHOLD_F}\u00b0F, still up at 7AM)',
        fontsize=19, fontweight='bold', color=DARK_GREEN, pad=14, loc='left',
    )

    if temp_df is None or temp_df.empty:
        outer_ax.text(0.02, 0.5, f'No wells showed a sustained motor-temp rise greater than {TEMP_RISE_THRESHOLD_F}\u00b0F.',
                       fontsize=15, color=SLATE, transform=outer_ax.transAxes)
        return

    shown = temp_df.head(max_wells).iloc[::-1]

    gs_inner = gs_cell.subgridspec(1, 2, width_ratios=[0.28, 0.72])
    ax = fig.add_subplot(gs_inner[0, 1])
    ax.set_facecolor(BG_PANEL)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=SLATE, labelsize=13)
    ax.set_axisbelow(True)
    ax.grid(axis='x', color=GRID, linewidth=0.8, zorder=0)
    ax.grid(axis='y', visible=False)

    wells = shown['Well'].astype(str)
    baseline = shown['Baseline Temp (F)']
    rise = shown['Rise (F)']

    ax.barh(wells, baseline, color=SLATE, alpha=0.35, zorder=3, edgecolor='white', linewidth=1, label='Baseline (normal)')
    ax.barh(wells, rise, left=baseline, color=RED, zorder=3, edgecolor='white', linewidth=1, label='Increase')

    for i, (b, r) in enumerate(zip(baseline.values, rise.values)):
        ax.text(b + r + max(baseline + rise) * 0.015, i, f'+{r:.1f}\u00b0F', va='center',
                fontweight='bold', fontsize=14, color=RED)

    ax.set_xlim(0, (baseline + rise).max() * 1.18)
    ax.set_xlabel('Motor Temperature (\u00b0F)', fontsize=14, color=SLATE)
    ax.legend(loc='lower right', fontsize=12, frameon=False, labelcolor=SLATE)

    if len(temp_df) > max_wells:
        outer_ax.text(0.02, -0.06, f'+ {len(temp_df) - max_wells} more wells \u2014 see detail workbook',
                       fontsize=12, color=SLATE, style='italic', transform=outer_ax.transAxes)


def draw_reason_pie_mobile(fig, gs_cell, reason_counts, max_slices=8):
    # Title gets its own thin row (rather than sharing the pie's axes), with
    # a real gap (hspace) before the pie row below it - this both keeps the
    # title anchored to the true left margin of the page and guarantees
    # breathing room between the title text and the top of the pie.
    gs_rows = gs_cell.subgridspec(2, 1, height_ratios=[0.16, 0.84], hspace=0.35)
    title_ax = fig.add_subplot(gs_rows[0])
    title_ax.axis('off')
    title_ax.set_title('Shutdown Reasons \u2014 All Wells', fontsize=19, fontweight='bold',
                        color=DARK_GREEN, pad=0, loc='left')

    if reason_counts is None or len(reason_counts) == 0:
        title_ax.text(0.02, -0.6, 'No shutdown events logged for this period.',
                       fontsize=15, color=SLATE, transform=title_ax.transAxes)
        return

    counts = reason_counts.copy()
    if len(counts) > max_slices:
        top = counts.iloc[:max_slices - 1]
        other = pd.Series({'Other reasons': counts.iloc[max_slices - 1:].sum()})
        counts = pd.concat([top, other])

    palette = [DARK_GREEN, GREEN, LIGHT_GREEN, AMBER, RED, PURPLE, SLATE, '#0891b2', '#be185d']
    colors = [palette[i % len(palette)] for i in range(len(counts))]

    # Pie + legend sit on their own nested axes in the row below the title.
    # A slim padding column on each side keeps the pie itself centered on
    # the row while the legend hugs the right side.
    gs_inner = gs_rows[1].subgridspec(1, 3, width_ratios=[0.04, 0.62, 0.34])
    ax = fig.add_subplot(gs_inner[0, 1])
    ax.axis('off')
    ax_legend = fig.add_subplot(gs_inner[0, 2])
    ax_legend.axis('off')

    wedges, _, autotexts = ax.pie(
        counts.values, colors=colors, autopct=lambda p: f'{p:.0f}%' if p >= 4 else '',
        startangle=90, pctdistance=0.78, radius=1.35,
        wedgeprops=dict(edgecolor='white', linewidth=1.5),
        textprops=dict(color='white', fontweight='bold', fontsize=13),
    )
    ax_legend.legend(
        wedges, [f'{name}  ({int(val)})' for name, val in zip(counts.index, counts.values)],
        loc='center left', fontsize=13, frameon=False, labelcolor=SLATE,
    )


def compute_row_height_ratios_mobile(summary):
    """Mirrors compute_row_height_ratios, but for 6 stacked single-column
    rows: KPI, shutdown table, bar chart, pie chart, vx table, pip table.
    Floors are kept tight so a quiet report (few/no rows in a section)
    doesn't reserve a tall, mostly-empty block for it."""
    n_sd = min(len(summary['shutdown_df']), 20) if not summary['shutdown_df'].empty else 0
    n_bar = min(len(summary['shutdown_count_df']), 15) if not summary['shutdown_count_df'].empty else 0
    n_vx = min(len(summary['vx_df']), 12) if not summary['vx_df'].empty else 0
    n_pip = min(len(summary['pip_df']), 12) if not summary['pip_df'].empty else 0
    n_pie = min(len(summary['reason_counts']), 8) if len(summary['reason_counts']) else 0
    n_temp = min(len(summary['temp_df']), 15) if not summary['temp_df'].empty else 0

    note_lines = (
        (1 if summary.get('stopped', 0) > 0 else 0)
        + (1 if summary.get('miscommunication', 0) > 0 and summary.get('miscommunication_wells') else 0)
        + (1 if summary.get('no_data', 0) > 0 else 0)
    )
    ratio_kpi = 0.50 + note_lines * 0.24
    ratio_shutdown = min(1.55, max(0.32, 0.20 + 0.065 * n_sd))
    ratio_bar = min(1.05, max(0.38, 0.22 + 0.052 * n_bar))
    ratio_pie = min(1.05, max(0.52, 0.30 + 0.068 * n_pie))
    ratio_vx = min(1.05, max(0.32, 0.19 + 0.068 * n_vx))
    ratio_pip = min(1.05, max(0.32, 0.19 + 0.068 * n_pip))
    ratio_temp = min(1.05, max(0.32, 0.19 + 0.052 * n_temp))
    return [ratio_kpi, ratio_shutdown, ratio_bar, ratio_pie, ratio_vx, ratio_pip, ratio_temp]


def build_mobile_dashboard_figure(summary, report_date, output_png):
    """Portrait, single-column dashboard sized for WhatsApp: narrower
    page width means everything renders bigger relative to the chat
    bubble WhatsApp scales it into, so no pinch-zoom needed to read it."""
    height_ratios = compute_row_height_ratios_mobile(summary)
    gridspec_in = sum(height_ratios) * MOBILE_INCHES_PER_RATIO_UNIT
    fig_height = max(MOBILE_MIN_HEIGHT_IN, gridspec_in / (MOBILE_TOP - MOBILE_BOTTOM))

    fig = plt.figure(figsize=(MOBILE_FIG_WIDTH_IN, fig_height), facecolor='white')
    gs = fig.add_gridspec(7, 1, height_ratios=height_ratios, hspace=0.60,
                           left=MOBILE_LEFT, right=MOBILE_RIGHT, top=MOBILE_TOP, bottom=MOBILE_BOTTOM)

    draw_section_dividers(fig, gs)

    fig.text(0.5, 0.975, 'Daily ESP Surveillance Summary', fontsize=25, fontweight='bold',
              color=DARK_GREEN, ha='center')
    fig.text(0.5, 0.957, f'{summary["total"]} Wells  \u2022  {report_date}',
              fontsize=13.5, color=GREEN, ha='center')

    draw_logos_mobile(fig, fig_height)

    draw_kpi_cards_mobile(fig, gs[0], summary)

    shutdown_df_mobile = summary['shutdown_df'].copy()
    if not shutdown_df_mobile.empty:
        shutdown_df_mobile['Shutdown Start'] = shutdown_df_mobile['Shutdown Start'].apply(_shorten_dt)
        shutdown_df_mobile['Shutdown End'] = shutdown_df_mobile['Shutdown End'].apply(_shorten_dt)
        shutdown_df_mobile = shutdown_df_mobile.rename(columns={
            'Shutdown Start': 'Start', 'Shutdown End': 'End', 'Downtime (hrs)': 'Hrs',
        })

    draw_table_mobile(
        fig, gs[1], shutdown_df_mobile,
        columns=['Well', 'Reason', 'Start', 'End', 'Hrs'],
        title='Shutdown Events & Downtime',
        accent=LIGHT_GREEN, empty_msg='No shutdown events logged for this period.',
        max_rows=20, col_widths=[0.13, 0.39, 0.17, 0.17, 0.14],
    )

    draw_shutdown_count_bar_mobile(fig, gs[2], summary['shutdown_count_df'])
    draw_reason_pie_mobile(fig, gs[3], summary['reason_counts'])

    draw_table_mobile(
        fig, gs[4], summary['vx_df'],
        columns=['Well', 'Max Vx (G)', 'Time of Max', 'Doubled & Still Doubled at 7AM'],
        title=f'High Vibration Alerts (Vx > {VX_THRESHOLD_G}G or doubled at 7AM)',
        accent=LIGHT_GREEN, empty_msg=f'No wells exceeded {VX_THRESHOLD_G}G or showed a sustained doubling.',
        max_rows=12, col_widths=[0.24, 0.20, 0.24, 0.32],
    )

    pip_df = summary['pip_df']
    pip_cols = ['Well', 'PIP-Yesterday 7AM (psi)', 'PIP-Today 7AM (psi)', 'Net Rise (psi)'] if not pip_df.empty else []
    draw_table_mobile(
        fig, gs[5], pip_df,
        columns=pip_cols,
        title=f'Rising PIP Trends (> {PIP_RISE_THRESHOLD_PSI} psi, no shutdowns)',
        accent=LIGHT_GREEN, empty_msg=f'No wells showed a sustained PIP rise greater than {PIP_RISE_THRESHOLD_PSI} psi.',
        max_rows=12, col_widths=[0.20, 0.30, 0.28, 0.22],
    )

    fig.text(0.5, 0.015, f'Generated {datetime.now().strftime("%d-%b-%Y %H:%M")}  \u2022  ProductionLink Team, TAM OIL',
              fontsize=10.5, color=SLATE, ha='center', style='italic')

    draw_motor_temp_bar_mobile(fig, gs[6], summary['temp_df'])

    plt.savefig(output_png, dpi=200, facecolor='white', bbox_inches='tight')
    plt.close(fig)


MOBILE_LOGO_HEIGHT_IN = 0.58
MOBILE_LOGO_MAX_WIDTH_IN = 2.0


def _place_logo_mobile(fig, path, fig_height, side):
    """Places a logo at a fixed physical size (inches) rather than a fixed
    figure-fraction size, since the mobile page height varies a lot
    report-to-report - a fraction-based size would balloon on tall pages."""
    img = mpimg.imread(path)
    px_h, px_w = img.shape[0], img.shape[1]
    aspect = px_w / px_h
    h_in = MOBILE_LOGO_HEIGHT_IN
    w_in = min(MOBILE_LOGO_MAX_WIDTH_IN, aspect * h_in)

    w_frac = w_in / MOBILE_FIG_WIDTH_IN
    h_frac = h_in / fig_height
    y0 = 0.975 - h_frac / 2  # roughly vertically centered on the title row
    x0 = 0.05 if side == 'left' else (0.95 - w_frac)

    ax_logo = fig.add_axes([x0, y0, w_frac, h_frac])
    ax_logo.imshow(img)
    ax_logo.axis('off')


def draw_logos_mobile(fig, fig_height):
    """Same idea as draw_logos, sized/positioned for the narrower,
    height-variable mobile page."""
    if TAM_LOGO_PATH and os.path.isfile(TAM_LOGO_PATH):
        try:
            _place_logo_mobile(fig, TAM_LOGO_PATH, fig_height, 'left')
        except Exception as e:
            print(f'Could not load TAM logo ({TAM_LOGO_PATH}): {e}')

    if KHALDA_LOGO_PATH and os.path.isfile(KHALDA_LOGO_PATH):
        try:
            _place_logo_mobile(fig, KHALDA_LOGO_PATH, fig_height, 'right')
        except Exception as e:
            print(f'Could not load Khalda logo ({KHALDA_LOGO_PATH}): {e}')
