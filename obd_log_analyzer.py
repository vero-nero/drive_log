#!/usr/bin/env python3
"""OBD driving-log analyzer and desktop viewer.

Usage:
    python obd_log_analyzer.py
    python obd_log_analyzer.py path/to/log.csv
    python obd_log_analyzer.py path/to/log.csv --report report.html

Dependencies:
    pandas, numpy, matplotlib
Tkinter is included with most standard Python installations.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

import matplotlib

# Select a non-interactive backend only for command-line report generation.
if "--report" in sys.argv:
    matplotlib.use("Agg")

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import tkinter as tk
from tkinter import filedialog, messagebox, ttk


APP_TITLE = "OBD Drive Log Analyzer"

# Restrained palette: blue, amber, slate, green and red only where meaningful.
COLORS = {
    "blue": "#315A7D",
    "amber": "#B27A2A",
    "slate": "#5B6573",
    "green": "#50745D",
    "red": "#A44949",
    "light": "#F3F5F7",
    "border": "#D8DDE3",
    "text": "#20262D",
    "muted": "#66717D",
}


@dataclass
class Thresholds:
    atmospheric_kpa: float = 101.3
    warning_boost_bar: float = 1.15
    critical_boost_bar: float = 1.30
    warning_coolant_c: float = 105.0
    critical_coolant_c: float = 115.0
    warning_iat_c: float = 50.0
    critical_iat_c: float = 65.0
    warning_ltft_abs: float = 10.0
    critical_ltft_abs: float = 15.0
    warning_combined_trim_abs: float = 20.0
    critical_combined_trim_abs: float = 30.0
    warning_rpm: float = 6500.0
    lean_lambda: float = 1.05
    critical_lean_lambda: float = 1.10


ALIASES: dict[str, tuple[str, ...]] = {
    "timestamp": ("timestamp", "time", "datetime", "date time"),
    "monitor": ("monitor status since dtcs cleared", "monitor status", "mil status"),
    "fuel_status": ("fuel system status", "fuel status"),
    "load": ("calculated engine load", "engine load", "load"),
    "coolant": ("engine coolant temperature", "coolant temperature", "ect"),
    "stft": ("short-term fuel trim bank 1", "short term fuel trim bank 1", "stft bank 1", "stft"),
    "ltft": ("long-term fuel trim bank 1", "long term fuel trim bank 1", "ltft bank 1", "ltft"),
    "map": ("intake manifold absolute pressure", "manifold absolute pressure", "map"),
    "rpm": ("engine rpm", "rpm"),
    "speed": ("vehicle speed", "speed"),
    "timing": ("ignition timing advance", "timing advance", "ignition timing", "timing"),
    "iat": ("intake air temperature", "iat"),
    "maf": ("maf airflow", "mass air flow", "maf"),
    "throttle": ("throttle position", "absolute throttle position", "throttle"),
    "o2_b1s2": ("o2 sensor b1s2", "oxygen sensor b1s2"),
    "wideband": (
        "wideband o2 b1s1 equivalence ratio / voltage",
        "wideband o2 b1s1 equivalence ratio",
        "equivalence ratio bank 1 sensor 1",
        "lambda b1s1",
        "lambda",
    ),
    "boost_kpa": ("derived boost relative to atmosphere (kpa)", "boost kpa", "boost pressure kpa"),
    "boost_bar": ("derived boost relative to atmosphere (bar)", "boost bar", "boost pressure bar"),
    "combined_trim": ("combined fuel trim bank 1 (%)", "combined fuel trim bank 1", "combined fuel trim"),
}

DISPLAY_NAMES = {
    "rpm": "Engine RPM",
    "speed": "Vehicle speed",
    "coolant": "Coolant temperature",
    "iat": "Intake air temperature",
    "load": "Calculated engine load",
    "stft": "Short-term fuel trim B1",
    "ltft": "Long-term fuel trim B1",
    "combined_trim": "Combined fuel trim B1",
    "map": "Manifold pressure",
    "boost_bar": "Boost relative to atmosphere",
    "maf": "MAF airflow",
    "throttle": "Throttle position",
    "timing": "Ignition timing advance",
    "lambda": "Wideband lambda B1S1",
    "o2_b1s2_v": "O2 voltage B1S2",
}

UNITS = {
    "rpm": "rpm",
    "speed": "km/h",
    "coolant": "°C",
    "iat": "°C",
    "load": "%",
    "stft": "%",
    "ltft": "%",
    "combined_trim": "%",
    "map": "kPa abs",
    "boost_bar": "bar rel.",
    "maf": "g/s",
    "throttle": "%",
    "timing": "°",
    "lambda": "λ",
    "o2_b1s2_v": "V",
}


@dataclass
class AnalysisResult:
    source: Path
    data: pd.DataFrame
    mapping: dict[str, str]
    summary: dict[str, object]
    events: pd.DataFrame
    quality: list[str]


def normalize_name(value: str) -> str:
    value = value.strip().lower().replace("λ", "lambda")
    value = re.sub(r"[^a-z0-9%]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def detect_separator(path: Path, encoding: str) -> str:
    with path.open("r", encoding=encoding, errors="replace", newline="") as handle:
        sample = handle.read(8192)
    try:
        return csv.Sniffer().sniff(sample, delimiters=";,\t|").delimiter
    except csv.Error:
        counts = {sep: sample.count(sep) for sep in (";", ",", "\t", "|")}
        return max(counts, key=counts.get)


def read_csv_robust(path: Path) -> pd.DataFrame:
    last_error: Optional[Exception] = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            sep = detect_separator(path, encoding)
            frame = pd.read_csv(path, sep=sep, encoding=encoding, engine="python")
            if frame.shape[1] <= 1:
                continue
            frame.columns = [str(column).strip() for column in frame.columns]
            return frame
        except Exception as exc:  # Try the next common encoding.
            last_error = exc
    raise ValueError(f"Could not read CSV file: {last_error}")


def resolve_columns(columns: Iterable[str]) -> dict[str, str]:
    normalized = {normalize_name(column): column for column in columns}
    mapping: dict[str, str] = {}
    for key, aliases in ALIASES.items():
        for alias in aliases:
            candidate = normalize_name(alias)
            if candidate in normalized:
                mapping[key] = normalized[candidate]
                break
        if key in mapping:
            continue
        # Conservative partial matching for headers containing units or bank labels.
        for norm, original in normalized.items():
            if any(normalize_name(alias) in norm for alias in aliases):
                mapping[key] = original
                break
    return mapping


def to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.replace("\u2212", "-", regex=False)
        .str.replace(",", ".", regex=False)
        .str.extract(r"([-+]?\d+(?:\.\d+)?)", expand=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def extract_number(series: pd.Series, pattern: str) -> pd.Series:
    extracted = series.astype(str).str.extract(pattern, flags=re.IGNORECASE, expand=False)
    return pd.to_numeric(extracted.str.replace(",", ".", regex=False), errors="coerce")


def prepare_data(raw: pd.DataFrame, thresholds: Thresholds) -> tuple[pd.DataFrame, dict[str, str]]:
    mapping = resolve_columns(raw.columns)
    if "timestamp" not in mapping:
        raise ValueError("No timestamp column was found.")

    data = raw.copy()
    data["_timestamp"] = pd.to_datetime(data[mapping["timestamp"]], errors="coerce")
    data = data.loc[data["_timestamp"].notna()].copy()
    data.sort_values("_timestamp", inplace=True)
    data.reset_index(drop=True, inplace=True)
    if data.empty:
        raise ValueError("The CSV contains no valid timestamped rows.")

    start = data["_timestamp"].iloc[0]
    data["_elapsed_s"] = (data["_timestamp"] - start).dt.total_seconds()
    data["_elapsed_min"] = data["_elapsed_s"] / 60.0

    numeric_keys = (
        "load", "coolant", "stft", "ltft", "map", "rpm", "speed", "timing",
        "iat", "maf", "throttle", "boost_kpa", "boost_bar", "combined_trim",
    )
    for key in numeric_keys:
        if key in mapping:
            data[f"_{key}"] = to_numeric(data[mapping[key]])

    if "wideband" in mapping:
        source = data[mapping["wideband"]]
        data["_lambda"] = extract_number(source, r"(?:lambda|λ)\s*([-+]?\d+(?:[\.,]\d+)?)")
        data["_wideband_v"] = extract_number(source, r"([-+]?\d+(?:[\.,]\d+)?)\s*V")

    if "o2_b1s2" in mapping:
        source = data[mapping["o2_b1s2"]]
        data["_o2_b1s2_v"] = extract_number(source, r"([-+]?\d+(?:[\.,]\d+)?)\s*V")
        data["_o2_b1s2_trim"] = extract_number(source, r"trim\s*([-+]?\d+(?:[\.,]\d+)?)")

    # Prefer a logged boost value. Derive it from absolute MAP when absent.
    if "_boost_bar" not in data or data["_boost_bar"].notna().sum() == 0:
        if "_map" in data:
            data["_boost_bar"] = (data["_map"] - thresholds.atmospheric_kpa) / 100.0
            data["_boost_source"] = "Derived from MAP"
    else:
        if "_map" in data:
            derived = (data["_map"] - thresholds.atmospheric_kpa) / 100.0
            data["_boost_bar"] = data["_boost_bar"].fillna(derived)
        data["_boost_source"] = "Logged/derived"

    # Approximate distance using trapezoidal integration of vehicle speed.
    if "_speed" in data and len(data) > 1:
        seconds = data["_elapsed_s"].to_numpy(dtype=float)
        speed_ms = data["_speed"].fillna(0).to_numpy(dtype=float) / 3.6
        dt = np.diff(seconds, prepend=seconds[0])
        previous_speed = np.r_[speed_ms[0], speed_ms[:-1]]
        data["_distance_step_m"] = ((speed_ms + previous_speed) / 2.0) * dt
        data["_distance_km"] = data["_distance_step_m"].clip(lower=0).cumsum() / 1000.0

    return data, mapping


def add_event_rows(
    event_rows: list[dict[str, object]],
    data: pd.DataFrame,
    mask: pd.Series,
    severity: str,
    sensor: str,
    value_column: Optional[str],
    reason: str,
) -> None:
    """Collapse consecutive matching samples into readable event ranges."""
    valid_mask = mask.fillna(False).astype(bool)
    if not valid_mask.any():
        return

    groups = valid_mask.ne(valid_mask.shift(fill_value=False)).cumsum()
    for _, block in data.loc[valid_mask].groupby(groups[valid_mask]):
        first = block.iloc[0]
        last = block.iloc[-1]
        values = block[value_column].dropna() if value_column and value_column in block else pd.Series(dtype=float)
        peak = np.nan
        if not values.empty:
            # Use maximum absolute magnitude while retaining its sign.
            peak = values.iloc[np.abs(values.to_numpy(dtype=float)).argmax()]
        event_rows.append({
            "Severity": severity,
            "Start": first["_timestamp"],
            "End": last["_timestamp"],
            "Elapsed": float(first["_elapsed_s"]),
            "Sensor": sensor,
            "Peak value": float(peak) if pd.notna(peak) else np.nan,
            "Reason": reason,
            "Samples": int(len(block)),
        })


def evaluate_events(data: pd.DataFrame, mapping: dict[str, str], t: Thresholds) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def col(name: str) -> pd.Series:
        return data[name] if name in data else pd.Series(np.nan, index=data.index)

    if "monitor" in mapping:
        monitor = data[mapping["monitor"]].astype(str)
        dtc_count = extract_number(monitor, r"(\d+)\s*DTC")
        add_event_rows(rows, data, monitor.str.contains("MIL ON", case=False, na=False), "Critical", "MIL", None,
                       "Malfunction indicator lamp reported as ON.")
        add_event_rows(rows, data, dtc_count.fillna(0) > 0, "Critical", "DTC count", None,
                       "The log reports one or more diagnostic trouble codes.")

    coolant = col("_coolant")
    add_event_rows(rows, data, coolant >= t.critical_coolant_c, "Critical", "Coolant", "_coolant",
                   f"Coolant temperature reached at least {t.critical_coolant_c:.0f} °C.")
    add_event_rows(rows, data, (coolant >= t.warning_coolant_c) & (coolant < t.critical_coolant_c),
                   "Warning", "Coolant", "_coolant",
                   f"Coolant temperature reached at least {t.warning_coolant_c:.0f} °C.")

    iat = col("_iat")
    add_event_rows(rows, data, iat >= t.critical_iat_c, "Critical", "Intake air temperature", "_iat",
                   f"Intake air temperature reached at least {t.critical_iat_c:.0f} °C.")
    add_event_rows(rows, data, (iat >= t.warning_iat_c) & (iat < t.critical_iat_c),
                   "Warning", "Intake air temperature", "_iat",
                   f"Intake air temperature reached at least {t.warning_iat_c:.0f} °C.")

    ltft = col("_ltft").abs()
    add_event_rows(rows, data, ltft >= t.critical_ltft_abs, "Critical", "LTFT B1", "_ltft",
                   f"Absolute long-term fuel trim reached at least {t.critical_ltft_abs:.0f} %.")
    add_event_rows(rows, data, (ltft >= t.warning_ltft_abs) & (ltft < t.critical_ltft_abs),
                   "Warning", "LTFT B1", "_ltft",
                   f"Absolute long-term fuel trim reached at least {t.warning_ltft_abs:.0f} %.")

    closed_loop = pd.Series(True, index=data.index)
    if "fuel_status" in mapping:
        closed_loop = data[mapping["fuel_status"]].astype(str).str.contains("closed loop", case=False, na=False)
    combined = col("_combined_trim").abs()
    add_event_rows(rows, data, closed_loop & (combined >= t.critical_combined_trim_abs),
                   "Critical", "Combined fuel trim B1", "_combined_trim",
                   f"Closed-loop combined fuel trim reached at least ±{t.critical_combined_trim_abs:.0f} %.")
    add_event_rows(rows, data,
                   closed_loop & (combined >= t.warning_combined_trim_abs) & (combined < t.critical_combined_trim_abs),
                   "Warning", "Combined fuel trim B1", "_combined_trim",
                   f"Closed-loop combined fuel trim reached at least ±{t.warning_combined_trim_abs:.0f} %.")

    boost = col("_boost_bar")
    add_event_rows(rows, data, boost >= t.critical_boost_bar, "Critical", "Boost", "_boost_bar",
                   f"Relative boost reached at least {t.critical_boost_bar:.2f} bar.")
    add_event_rows(rows, data, (boost >= t.warning_boost_bar) & (boost < t.critical_boost_bar),
                   "Warning", "Boost", "_boost_bar",
                   f"Relative boost reached at least {t.warning_boost_bar:.2f} bar.")

    rpm = col("_rpm")
    add_event_rows(rows, data, rpm >= t.warning_rpm, "Warning", "Engine RPM", "_rpm",
                   f"Engine speed reached at least {t.warning_rpm:.0f} rpm.")

    # Lambda under meaningful positive load. This is an indicator, not a substitute for a calibrated wideband log.
    lambda_value = col("_lambda")
    throttle = col("_throttle")
    manifold = col("_map")
    loaded = (rpm >= 2500) & (throttle >= 35) & (manifold >= t.atmospheric_kpa)
    add_event_rows(rows, data, loaded & (lambda_value >= t.critical_lean_lambda),
                   "Critical", "Lambda under load", "_lambda",
                   f"Lambda reached at least {t.critical_lean_lambda:.2f} under positive-load conditions.")
    add_event_rows(rows, data,
                   loaded & (lambda_value >= t.lean_lambda) & (lambda_value < t.critical_lean_lambda),
                   "Warning", "Lambda under load", "_lambda",
                   f"Lambda reached at least {t.lean_lambda:.2f} under positive-load conditions.")

    timing = col("_timing")
    load = col("_load")
    add_event_rows(rows, data, (timing < 0) & (load >= 50), "Warning", "Ignition timing", "_timing",
                   "Negative ignition timing was logged at calculated load of at least 50 %.")

    events = pd.DataFrame(rows)
    if events.empty:
        return pd.DataFrame(columns=["Severity", "Start", "End", "Elapsed", "Sensor", "Peak value", "Reason", "Samples"])
    order = {"Critical": 0, "Warning": 1, "Info": 2}
    events["_order"] = events["Severity"].map(order).fillna(9)
    events.sort_values(["_order", "Start"], inplace=True)
    events.drop(columns="_order", inplace=True)
    events.reset_index(drop=True, inplace=True)
    return events


def calculate_quality(data: pd.DataFrame, mapping: dict[str, str]) -> list[str]:
    messages: list[str] = []
    intervals = data["_elapsed_s"].diff().dropna()
    if not intervals.empty:
        median = float(intervals.median())
        maximum = float(intervals.max())
        if median > 2.0:
            messages.append(f"Low sampling rate: median interval is {median:.2f} s. Fast transients may be missed.")
        if median > 0 and maximum > median * 3:
            messages.append(f"Irregular sampling: largest interval is {maximum:.2f} s versus {median:.2f} s median.")

    expected = ["rpm", "speed", "coolant", "stft", "ltft", "map", "maf", "throttle", "timing"]
    missing = [DISPLAY_NAMES.get(key, key) for key in expected if f"_{key}" not in data]
    if missing:
        messages.append("Missing common sensors: " + ", ".join(missing) + ".")

    if "_boost_source" in data and data["_boost_source"].iloc[0] == "Derived from MAP":
        messages.append("Boost was calculated from MAP using the configured atmospheric pressure.")

    for key in ("_rpm", "_speed", "_coolant", "_iat", "_map", "_maf", "_throttle", "_lambda"):
        if key in data:
            valid = data[key].dropna()
            if len(valid) >= 5 and valid.nunique() <= 1:
                messages.append(f"{DISPLAY_NAMES.get(key[1:], key)} is constant throughout the log; verify sensor support or logging.")

    if "_lambda" in data:
        bad = ((data["_lambda"] < 0.65) | (data["_lambda"] > 1.35)).mean()
        if bad > 0.20:
            messages.append("Wideband lambda contains many extreme values. Generic OBD equivalence-ratio data may be slow or unsupported.")

    return messages


def calculate_summary(data: pd.DataFrame, mapping: dict[str, str], events: pd.DataFrame) -> dict[str, object]:
    intervals = data["_elapsed_s"].diff().dropna()
    summary: dict[str, object] = {
        "rows": len(data),
        "start": data["_timestamp"].iloc[0],
        "end": data["_timestamp"].iloc[-1],
        "duration_s": float(data["_elapsed_s"].iloc[-1]),
        "sample_interval_s": float(intervals.median()) if not intervals.empty else 0.0,
        "critical_events": int((events["Severity"] == "Critical").sum()) if not events.empty else 0,
        "warning_events": int((events["Severity"] == "Warning").sum()) if not events.empty else 0,
    }

    for key in ("rpm", "speed", "coolant", "iat", "load", "stft", "ltft", "combined_trim", "map", "boost_bar", "maf", "throttle", "timing", "lambda"):
        column = f"_{key}"
        if column in data and data[column].notna().any():
            summary[f"{key}_min"] = float(data[column].min())
            summary[f"{key}_max"] = float(data[column].max())
            summary[f"{key}_mean"] = float(data[column].mean())

    if "_distance_km" in data:
        summary["distance_km"] = float(data["_distance_km"].iloc[-1])

    if "fuel_status" in mapping:
        status = data[mapping["fuel_status"]].astype(str)
        summary["closed_loop_pct"] = float(status.str.contains("closed loop", case=False, na=False).mean() * 100)

    if "monitor" in mapping:
        status = data[mapping["monitor"]].astype(str)
        summary["mil_on"] = bool(status.str.contains("MIL ON", case=False, na=False).any())
        counts = extract_number(status, r"(\d+)\s*DTC")
        summary["max_dtc_count"] = int(counts.max()) if counts.notna().any() else 0

    return summary


def analyze_file(path: Path, thresholds: Optional[Thresholds] = None) -> AnalysisResult:
    thresholds = thresholds or Thresholds()
    raw = read_csv_robust(path)
    data, mapping = prepare_data(raw, thresholds)
    events = evaluate_events(data, mapping, thresholds)
    summary = calculate_summary(data, mapping, events)
    quality = calculate_quality(data, mapping)
    return AnalysisResult(path, data, mapping, summary, events, quality)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:d}:{secs:02d}"


def fmt(value: object, decimals: int = 1, suffix: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    return f"{number:.{decimals}f}{suffix}"


def available_series(data: pd.DataFrame) -> list[str]:
    order = ["rpm", "speed", "coolant", "iat", "load", "throttle", "map", "boost_bar", "maf", "timing", "stft", "ltft", "combined_trim", "lambda", "o2_b1s2_v"]
    return [key for key in order if f"_{key}" in data and data[f"_{key}"].notna().any()]


CHART_PRESETS: dict[str, tuple[list[str], list[str]]] = {
    "Driving overview": (["rpm"], ["speed", "throttle"]),
    "Temperatures": (["coolant", "iat"], []),
    "Fuel trims": (["stft", "ltft", "combined_trim"], []),
    "Air and boost": (["maf"], ["map", "boost_bar"]),
    "Lambda and timing": (["lambda"], ["timing"]),
    "Load and throttle": (["load", "throttle"], []),
}

PLOT_COLORS = [COLORS["blue"], COLORS["amber"], COLORS["slate"], COLORS["green"]]


def plot_preset(figure: Figure, result: AnalysisResult, preset: str, smoothing: int = 1) -> None:
    figure.clear()
    axis = figure.add_subplot(111)
    axis2 = None
    data = result.data
    x = data["_elapsed_min"]
    left_keys, right_keys = CHART_PRESETS.get(preset, ([], []))
    left_keys = [key for key in left_keys if f"_{key}" in data and data[f"_{key}"].notna().any()]
    right_keys = [key for key in right_keys if f"_{key}" in data and data[f"_{key}"].notna().any()]

    if not left_keys and not right_keys:
        axis.text(0.5, 0.5, "No compatible sensors in this file", ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
        figure.tight_layout()
        return

    color_index = 0
    line_handles = []
    for key in left_keys:
        series = data[f"_{key}"].rolling(max(1, smoothing), center=True, min_periods=1).mean()
        line, = axis.plot(x, series, linewidth=1.6, color=PLOT_COLORS[color_index % len(PLOT_COLORS)], label=DISPLAY_NAMES[key])
        line_handles.append(line)
        color_index += 1

    if right_keys:
        axis2 = axis.twinx()
        for key in right_keys:
            series = data[f"_{key}"].rolling(max(1, smoothing), center=True, min_periods=1).mean()
            line, = axis2.plot(x, series, linewidth=1.45, linestyle="--", color=PLOT_COLORS[color_index % len(PLOT_COLORS)], label=DISPLAY_NAMES[key])
            line_handles.append(line)
            color_index += 1

    axis.set_xlabel("Elapsed time (minutes)")
    if left_keys:
        left_units = sorted({UNITS[key] for key in left_keys})
        axis.set_ylabel(" / ".join(left_units))
    if axis2 is not None and right_keys:
        right_units = sorted({UNITS[key] for key in right_keys})
        axis2.set_ylabel(" / ".join(right_units))

    axis.grid(True, linewidth=0.6, alpha=0.28)
    axis.set_facecolor("#FAFBFC")
    axis.spines["top"].set_visible(False)
    if axis2 is not None:
        axis2.spines["top"].set_visible(False)
    if line_handles:
        axis.legend(line_handles, [line.get_label() for line in line_handles], loc="upper left", frameon=False, ncol=2)
    axis.set_title(preset, loc="left", fontsize=12, fontweight="bold")
    figure.tight_layout()


def figure_to_base64(figure: Figure) -> str:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=145, bbox_inches="tight")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def create_html_report(result: AnalysisResult, output: Path) -> None:
    summary = result.summary
    chart_images: list[tuple[str, str]] = []
    for preset in CHART_PRESETS:
        fig = Figure(figsize=(10.5, 4.2), dpi=100)
        plot_preset(fig, result, preset, smoothing=1)
        chart_images.append((preset, figure_to_base64(fig)))

    event_rows = []
    for _, row in result.events.iterrows():
        peak = fmt(row["Peak value"], 2)
        event_rows.append(
            "<tr>"
            f"<td><span class='severity {html.escape(str(row['Severity']).lower())}'>{html.escape(str(row['Severity']))}</span></td>"
            f"<td>{html.escape(str(row['Start']))}</td>"
            f"<td>{html.escape(str(row['Sensor']))}</td>"
            f"<td>{peak}</td>"
            f"<td>{html.escape(str(row['Reason']))}</td>"
            "</tr>"
        )
    if not event_rows:
        event_rows.append("<tr><td colspan='5' class='muted'>No configured warning conditions were detected.</td></tr>")

    quality_items = "".join(f"<li>{html.escape(item)}</li>" for item in result.quality)
    if not quality_items:
        quality_items = "<li>No obvious data-quality issue was detected.</li>"

    cards = [
        ("Duration", format_duration(float(summary["duration_s"]))),
        ("Samples", f"{int(summary['rows']):,}"),
        ("Distance", fmt(summary.get("distance_km"), 2, " km")),
        ("Maximum RPM", fmt(summary.get("rpm_max"), 0, " rpm")),
        ("Maximum speed", fmt(summary.get("speed_max"), 0, " km/h")),
        ("Maximum coolant", fmt(summary.get("coolant_max"), 0, " °C")),
        ("Maximum boost", fmt(summary.get("boost_bar_max"), 2, " bar")),
        ("Critical / warning", f"{summary['critical_events']} / {summary['warning_events']}"),
    ]
    cards_html = "".join(
        f"<div class='card'><div class='label'>{html.escape(label)}</div><div class='value'>{html.escape(value)}</div></div>"
        for label, value in cards
    )
    charts_html = "".join(
        f"<section><h2>{html.escape(title)}</h2><img class='chart' src='data:image/png;base64,{image}' alt='{html.escape(title)}'></section>"
        for title, image in chart_images
    )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OBD report - {html.escape(result.source.name)}</title>
<style>
:root {{ --blue:#315A7D; --amber:#B27A2A; --red:#A44949; --green:#50745D; --text:#20262D; --muted:#66717D; --light:#F3F5F7; --border:#D8DDE3; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:Inter,Segoe UI,Arial,sans-serif; color:var(--text); background:#EEF1F4; }}
main {{ max-width:1180px; margin:28px auto; padding:0 20px 40px; }}
header, section {{ background:#fff; border:1px solid var(--border); border-radius:10px; padding:22px; margin-bottom:18px; box-shadow:0 3px 14px rgba(24,36,48,.05); }}
h1 {{ font-size:24px; margin:0 0 7px; }}
h2 {{ font-size:17px; margin:0 0 14px; }}
.meta, .muted {{ color:var(--muted); }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin-top:20px; }}
.card {{ background:var(--light); border:1px solid var(--border); border-radius:8px; padding:14px; }}
.label {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
.value {{ font-size:20px; font-weight:650; margin-top:5px; }}
.chart {{ display:block; width:100%; height:auto; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th, td {{ text-align:left; padding:10px 8px; border-bottom:1px solid var(--border); vertical-align:top; }}
th {{ color:var(--muted); font-weight:600; background:#FAFBFC; }}
.severity {{ display:inline-block; padding:3px 8px; border-radius:999px; font-size:11px; font-weight:700; }}
.severity.critical {{ color:#fff; background:var(--red); }}
.severity.warning {{ color:#fff; background:var(--amber); }}
ul {{ margin-bottom:0; }}
</style>
</head>
<body><main>
<header>
<h1>OBD Drive Log Report</h1>
<div class="meta">{html.escape(result.source.name)} · {html.escape(str(summary['start']))} to {html.escape(str(summary['end']))}</div>
<div class="cards">{cards_html}</div>
</header>
<section><h2>Detected conditions</h2><div style="overflow:auto"><table>
<thead><tr><th>Severity</th><th>Start</th><th>Sensor</th><th>Peak</th><th>Evaluation</th></tr></thead>
<tbody>{''.join(event_rows)}</tbody></table></div></section>
<section><h2>Data quality</h2><ul>{quality_items}</ul></section>
{charts_html}
</main></body></html>"""
    output.write_text(document, encoding="utf-8")


class OBDAnalyzerApp(tk.Tk):
    def __init__(self, initial_file: Optional[Path] = None) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1280x800")
        self.minsize(1000, 650)
        self.configure(bg=COLORS["light"])

        self.thresholds = Thresholds()
        self.result: Optional[AnalysisResult] = None
        self.current_path: Optional[Path] = None
        self._configure_style()
        self._build_ui()

        if initial_file:
            self.after(100, lambda: self.load_file(initial_file))

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=COLORS["light"])
        style.configure("Panel.TFrame", background="white", relief="flat")
        style.configure("TLabel", background=COLORS["light"], foreground=COLORS["text"], font=("Segoe UI", 10))
        style.configure("Title.TLabel", background=COLORS["light"], foreground=COLORS["text"], font=("Segoe UI Semibold", 18))
        style.configure("Muted.TLabel", background=COLORS["light"], foreground=COLORS["muted"], font=("Segoe UI", 9))
        style.configure("CardTitle.TLabel", background="white", foreground=COLORS["muted"], font=("Segoe UI", 9))
        style.configure("CardValue.TLabel", background="white", foreground=COLORS["text"], font=("Segoe UI Semibold", 17))
        style.configure("TButton", font=("Segoe UI", 9), padding=(10, 6))
        style.configure("Accent.TButton", foreground="white", background=COLORS["blue"])
        style.map("Accent.TButton", background=[("active", "#284A67")])
        style.configure("TNotebook", background=COLORS["light"], borderwidth=0)
        style.configure("TNotebook.Tab", font=("Segoe UI", 10), padding=(14, 8))
        style.map("TNotebook.Tab", background=[("selected", "white")])
        style.configure("Treeview", font=("Segoe UI", 9), rowheight=25, background="white", fieldbackground="white")
        style.configure("Treeview.Heading", font=("Segoe UI Semibold", 9), background="#E9EDF1")

    def _build_ui(self) -> None:
        header = ttk.Frame(self, padding=(18, 14))
        header.pack(fill="x")
        title_group = ttk.Frame(header)
        title_group.pack(side="left", fill="x", expand=True)
        ttk.Label(title_group, text=APP_TITLE, style="Title.TLabel").pack(anchor="w")
        self.file_label = ttk.Label(title_group, text="Open a CSV driving log to begin.", style="Muted.TLabel")
        self.file_label.pack(anchor="w", pady=(2, 0))

        ttk.Button(header, text="Open CSV", style="Accent.TButton", command=self.open_file).pack(side="right", padx=(8, 0))
        ttk.Button(header, text="Export HTML", command=self.export_report).pack(side="right", padx=(8, 0))
        ttk.Button(header, text="Settings", command=self.open_settings).pack(side="right", padx=(8, 0))

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=18, pady=(0, 18))

        self.overview_tab = ttk.Frame(self.notebook, padding=14)
        self.charts_tab = ttk.Frame(self.notebook, padding=12)
        self.events_tab = ttk.Frame(self.notebook, padding=12)
        self.data_tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(self.overview_tab, text="Overview")
        self.notebook.add(self.charts_tab, text="Charts")
        self.notebook.add(self.events_tab, text="Findings")
        self.notebook.add(self.data_tab, text="Data")

        self._build_overview()
        self._build_charts()
        self._build_events()
        self._build_data_table()

    def _build_overview(self) -> None:
        cards_frame = ttk.Frame(self.overview_tab)
        cards_frame.pack(fill="x")
        for column in range(4):
            cards_frame.columnconfigure(column, weight=1)
        self.card_widgets: dict[str, ttk.Label] = {}
        card_definitions = [
            ("duration", "Duration"), ("samples", "Samples"), ("distance", "Distance"), ("interval", "Median sample interval"),
            ("rpm", "Maximum RPM"), ("speed", "Maximum speed"), ("coolant", "Maximum coolant"), ("boost", "Maximum boost"),
        ]
        for index, (key, title) in enumerate(card_definitions):
            frame = ttk.Frame(cards_frame, style="Panel.TFrame", padding=14)
            frame.grid(row=index // 4, column=index % 4, sticky="nsew", padx=5, pady=5)
            ttk.Label(frame, text=title, style="CardTitle.TLabel").pack(anchor="w")
            value = ttk.Label(frame, text="—", style="CardValue.TLabel")
            value.pack(anchor="w", pady=(5, 0))
            self.card_widgets[key] = value

        lower = ttk.Panedwindow(self.overview_tab, orient="horizontal")
        lower.pack(fill="both", expand=True, pady=(12, 0))

        quality_panel = ttk.Frame(lower, style="Panel.TFrame", padding=15)
        event_panel = ttk.Frame(lower, style="Panel.TFrame", padding=15)
        lower.add(quality_panel, weight=1)
        lower.add(event_panel, weight=1)

        ttk.Label(quality_panel, text="Data quality", style="CardValue.TLabel").pack(anchor="w")
        self.quality_text = tk.Text(quality_panel, wrap="word", height=10, borderwidth=0, background="white", foreground=COLORS["text"], font=("Segoe UI", 10))
        self.quality_text.pack(fill="both", expand=True, pady=(10, 0))
        self.quality_text.configure(state="disabled")

        ttk.Label(event_panel, text="Evaluation summary", style="CardValue.TLabel").pack(anchor="w")
        self.summary_text = tk.Text(event_panel, wrap="word", height=10, borderwidth=0, background="white", foreground=COLORS["text"], font=("Segoe UI", 10))
        self.summary_text.pack(fill="both", expand=True, pady=(10, 0))
        self.summary_text.configure(state="disabled")

    def _build_charts(self) -> None:
        controls = ttk.Frame(self.charts_tab)
        controls.pack(fill="x", pady=(0, 8))
        ttk.Label(controls, text="View:").pack(side="left")
        self.preset_var = tk.StringVar(value="Driving overview")
        preset = ttk.Combobox(controls, textvariable=self.preset_var, values=list(CHART_PRESETS), state="readonly", width=24)
        preset.pack(side="left", padx=(6, 16))
        preset.bind("<<ComboboxSelected>>", lambda _event: self.refresh_chart())

        ttk.Label(controls, text="Smoothing samples:").pack(side="left")
        self.smoothing_var = tk.IntVar(value=1)
        smoothing = ttk.Spinbox(controls, from_=1, to=50, textvariable=self.smoothing_var, width=6, command=self.refresh_chart)
        smoothing.pack(side="left", padx=(6, 0))
        smoothing.bind("<Return>", lambda _event: self.refresh_chart())

        self.figure = Figure(figsize=(9, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.charts_tab)
        toolbar_frame = ttk.Frame(self.charts_tab)
        toolbar_frame.pack(fill="x")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="left")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_events(self) -> None:
        columns = ("Severity", "Start", "End", "Sensor", "Peak value", "Samples", "Reason")
        self.events_tree = ttk.Treeview(self.events_tab, columns=columns, show="headings")
        widths = {"Severity": 80, "Start": 150, "End": 150, "Sensor": 170, "Peak value": 90, "Samples": 70, "Reason": 520}
        for column in columns:
            self.events_tree.heading(column, text=column)
            self.events_tree.column(column, width=widths[column], minwidth=60, stretch=column == "Reason")
        self.events_tree.tag_configure("critical", foreground=COLORS["red"])
        self.events_tree.tag_configure("warning", foreground="#8A5D20")
        y_scroll = ttk.Scrollbar(self.events_tab, orient="vertical", command=self.events_tree.yview)
        x_scroll = ttk.Scrollbar(self.events_tab, orient="horizontal", command=self.events_tree.xview)
        self.events_tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.events_tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        self.events_tab.rowconfigure(0, weight=1)
        self.events_tab.columnconfigure(0, weight=1)

    def _build_data_table(self) -> None:
        self.data_tree = ttk.Treeview(self.data_tab, show="headings")
        y_scroll = ttk.Scrollbar(self.data_tab, orient="vertical", command=self.data_tree.yview)
        x_scroll = ttk.Scrollbar(self.data_tab, orient="horizontal", command=self.data_tree.xview)
        self.data_tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.data_tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        self.data_tab.rowconfigure(0, weight=1)
        self.data_tab.columnconfigure(0, weight=1)

    def open_file(self) -> None:
        filename = filedialog.askopenfilename(title="Open OBD CSV log", filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")])
        if filename:
            self.load_file(Path(filename))

    def load_file(self, path: Path) -> None:
        try:
            self.result = analyze_file(path, self.thresholds)
            self.current_path = path
            self.file_label.configure(text=f"{path.name} · {len(self.result.data):,} samples · {format_duration(self.result.summary['duration_s'])}")
            self.refresh_all()
        except Exception as exc:
            messagebox.showerror("Could not analyze file", str(exc))

    def refresh_all(self) -> None:
        if not self.result:
            return
        self.refresh_overview()
        self.refresh_chart()
        self.refresh_events()
        self.refresh_data_table()

    def refresh_overview(self) -> None:
        assert self.result is not None
        s = self.result.summary
        values = {
            "duration": format_duration(float(s["duration_s"])),
            "samples": f"{int(s['rows']):,}",
            "distance": fmt(s.get("distance_km"), 2, " km"),
            "interval": fmt(s.get("sample_interval_s"), 2, " s"),
            "rpm": fmt(s.get("rpm_max"), 0, " rpm"),
            "speed": fmt(s.get("speed_max"), 0, " km/h"),
            "coolant": fmt(s.get("coolant_max"), 0, " °C"),
            "boost": fmt(s.get("boost_bar_max"), 2, " bar"),
        }
        for key, value in values.items():
            self.card_widgets[key].configure(text=value)

        quality = self.result.quality or ["No obvious data-quality issue was detected."]
        self._set_text(self.quality_text, "\n\n".join(f"• {item}" for item in quality))

        critical = int(s.get("critical_events", 0))
        warning = int(s.get("warning_events", 0))
        lines = [f"Critical findings: {critical}", f"Warnings: {warning}"]
        if "closed_loop_pct" in s:
            lines.append(f"Closed-loop samples: {s['closed_loop_pct']:.1f} %")
        if "ltft_max" in s and "ltft_min" in s:
            lines.append(f"LTFT range: {s['ltft_min']:.1f} to {s['ltft_max']:.1f} %")
        if "combined_trim_max" in s and "combined_trim_min" in s:
            lines.append(f"Combined trim range: {s['combined_trim_min']:.1f} to {s['combined_trim_max']:.1f} %")
        if "lambda_min" in s and "lambda_max" in s:
            lines.append(f"Logged lambda range: {s['lambda_min']:.3f} to {s['lambda_max']:.3f}")
        self._set_text(self.summary_text, "\n\n".join(lines))

    @staticmethod
    def _set_text(widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def refresh_chart(self) -> None:
        if not self.result:
            self.figure.clear()
            axis = self.figure.add_subplot(111)
            axis.text(0.5, 0.5, "Open a CSV log to display charts", ha="center", va="center", transform=axis.transAxes)
            axis.set_axis_off()
            self.canvas.draw_idle()
            return
        try:
            smoothing = max(1, int(self.smoothing_var.get()))
        except (ValueError, tk.TclError):
            smoothing = 1
        plot_preset(self.figure, self.result, self.preset_var.get(), smoothing)
        self.canvas.draw_idle()

    def refresh_events(self) -> None:
        assert self.result is not None
        self.events_tree.delete(*self.events_tree.get_children())
        for _, row in self.result.events.iterrows():
            values = (
                row["Severity"],
                pd.Timestamp(row["Start"]).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                pd.Timestamp(row["End"]).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                row["Sensor"],
                fmt(row["Peak value"], 2),
                row["Samples"],
                row["Reason"],
            )
            self.events_tree.insert("", "end", values=values, tags=(str(row["Severity"]).lower(),))

    def refresh_data_table(self) -> None:
        assert self.result is not None
        data = self.result.data.copy()
        derived_columns = [column for column in data.columns if column.startswith("_")]
        raw_columns = [column for column in data.columns if not column.startswith("_")]
        columns = raw_columns + [column for column in derived_columns if column in ("_elapsed_s", "_boost_bar", "_lambda", "_o2_b1s2_v")]
        self.data_tree.delete(*self.data_tree.get_children())
        self.data_tree["columns"] = columns
        for column in columns:
            heading = column.lstrip("_").replace("_", " ").title() if column.startswith("_") else column
            self.data_tree.heading(column, text=heading)
            self.data_tree.column(column, width=min(260, max(90, len(heading) * 8)), stretch=False)
        # Treeview is intended for inspection, not as a replacement for pandas on very large logs.
        for _, row in data.head(5000).iterrows():
            values = []
            for column in columns:
                value = row[column]
                if isinstance(value, pd.Timestamp):
                    value = value.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                elif pd.isna(value):
                    value = ""
                elif isinstance(value, float):
                    value = f"{value:.4g}"
                values.append(value)
            self.data_tree.insert("", "end", values=values)

    def export_report(self) -> None:
        if not self.result:
            messagebox.showinfo("No data", "Open a CSV log first.")
            return
        initial = self.result.source.with_suffix(".html").name
        filename = filedialog.asksaveasfilename(title="Export HTML report", defaultextension=".html", initialfile=initial, filetypes=[("HTML report", "*.html")])
        if not filename:
            return
        try:
            create_html_report(self.result, Path(filename))
            messagebox.showinfo("Report exported", f"Saved report to:\n{filename}")
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))

    def open_settings(self) -> None:
        dialog = tk.Toplevel(self)
        dialog.title("Evaluation settings")
        dialog.transient(self)
        dialog.grab_set()
        dialog.resizable(False, False)
        frame = ttk.Frame(dialog, padding=16)
        frame.pack(fill="both", expand=True)

        fields = [
            ("Atmospheric pressure (kPa)", "atmospheric_kpa"),
            ("Boost warning (bar)", "warning_boost_bar"),
            ("Boost critical (bar)", "critical_boost_bar"),
            ("Coolant warning (°C)", "warning_coolant_c"),
            ("Coolant critical (°C)", "critical_coolant_c"),
            ("IAT warning (°C)", "warning_iat_c"),
            ("IAT critical (°C)", "critical_iat_c"),
            ("LTFT warning absolute (%)", "warning_ltft_abs"),
            ("LTFT critical absolute (%)", "critical_ltft_abs"),
            ("RPM warning", "warning_rpm"),
        ]
        variables: dict[str, tk.StringVar] = {}
        for row, (label, attribute) in enumerate(fields):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 14), pady=4)
            variable = tk.StringVar(value=str(getattr(self.thresholds, attribute)))
            variables[attribute] = variable
            ttk.Entry(frame, textvariable=variable, width=12).grid(row=row, column=1, sticky="e", pady=4)

        def save() -> None:
            try:
                for attribute, variable in variables.items():
                    setattr(self.thresholds, attribute, float(variable.get()))
            except ValueError:
                messagebox.showerror("Invalid value", "All settings must be numeric.", parent=dialog)
                return
            dialog.destroy()
            if self.current_path:
                self.load_file(self.current_path)

        buttons = ttk.Frame(frame)
        buttons.grid(row=len(fields), column=0, columnspan=2, sticky="e", pady=(14, 0))
        ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="right")
        ttk.Button(buttons, text="Apply", style="Accent.TButton", command=save).pack(side="right", padx=(0, 8))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze and display OBD driving logs.")
    parser.add_argument("csv_file", nargs="?", type=Path, help="CSV log to open")
    parser.add_argument("--report", type=Path, help="Create an HTML report without opening the GUI")
    parser.add_argument("--atmospheric-kpa", type=float, default=101.3, help="Atmospheric pressure used for MAP-derived boost")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    thresholds = Thresholds(atmospheric_kpa=args.atmospheric_kpa)
    if args.report:
        if not args.csv_file:
            print("A CSV file is required when using --report.", file=sys.stderr)
            return 2
        try:
            result = analyze_file(args.csv_file, thresholds)
            create_html_report(result, args.report)
            print(f"Report written to {args.report}")
            return 0
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    app = OBDAnalyzerApp(args.csv_file)
    app.thresholds = thresholds
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
