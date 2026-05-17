"""
============================================================
analytics.py
AI-Based Real-Time Human Posture Detection and Correction System
============================================================
Analytics and Visualization Module

Generates:
1.  Good vs Bad posture pie chart
2.  Posture trend over time (line graph)
3.  Angle history graphs (neck, spine, shoulder)
4.  Daily posture statistics bar chart
5.  Session summary report
6.  PDF posture report (using reportlab/fpdf2)
7.  Real-time graph data for Streamlit
============================================================
"""

import os
import datetime
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
GRAPHS_DIR  = os.path.join(BASE_DIR, "graphs")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
HISTORY_CSV = os.path.join(BASE_DIR, "posture_history.csv")

for d in [GRAPHS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# SHARED DARK THEME SETTINGS
# ─────────────────────────────────────────────────────────────
DARK_BG    = "#0f0f1a"
CARD_BG    = "#1a1a2e"
ACCENT1    = "#00d4ff"  # cyan
ACCENT2    = "#7c3aed"  # purple
GOOD_COLOR = "#10b981"  # green
BAD_COLOR  = "#ef4444"  # red
WARN_COLOR = "#f59e0b"  # amber

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    CARD_BG,
    "text.color":        "white",
    "axes.labelcolor":   "white",
    "xtick.color":       "#aaa",
    "ytick.color":       "#aaa",
    "axes.edgecolor":    "#333",
    "grid.color":        "#222",
    "grid.alpha":        0.4,
    "font.family":       "sans-serif",
})


# ─────────────────────────────────────────────────────────────
# SECTION 1: LOAD HISTORY
# ─────────────────────────────────────────────────────────────

def load_history() -> pd.DataFrame:
    """
    Load posture_history.csv and return a clean DataFrame.

    Returns:
        pd.DataFrame: Posture history with datetime parsing
    """
    from utils import load_posture_history
    df = load_posture_history()
    if df.empty:
        logger.warning("[Analytics] No posture history data found.")
    return df


# ─────────────────────────────────────────────────────────────
# SECTION 2: PIE CHART — Good vs Bad
# ─────────────────────────────────────────────────────────────

def plot_posture_pie(df: pd.DataFrame = None, save=True) -> str:
    """
    Generate a donut chart showing Good vs Bad posture distribution.

    Parameters:
        df   (pd.DataFrame): Posture history (loads from CSV if None)
        save (bool): Save to graphs/ if True

    Returns:
        str: File path of saved chart
    """
    if df is None:
        df = load_history()

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    if df.empty:
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center",
                color="white", fontsize=14, transform=ax.transAxes)
        ax.axis("off")
    else:
        good_count = (df["posture_text"] == "GOOD POSTURE").sum()
        bad_count  = (df["posture_text"] == "BAD POSTURE").sum()
        total      = good_count + bad_count

        if total == 0:
            good_count, bad_count = 1, 1  # placeholder

        sizes  = [good_count, bad_count]
        labels = [f"Good\n{good_count}", f"Bad\n{bad_count}"]
        colors = [GOOD_COLOR, BAD_COLOR]
        explode = (0.05, 0.05)

        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors,
            autopct="%1.1f%%", startangle=90,
            explode=explode, pctdistance=0.75,
            wedgeprops={"edgecolor": DARK_BG, "linewidth": 3,
                        "width": 0.6}  # donut hole
        )

        for t in texts:
            t.set_color("white")
            t.set_fontsize(11)
        for at in autotexts:
            at.set_color("white")
            at.set_fontweight("bold")

    ax.set_title("Posture Distribution", color="white",
                 fontsize=14, fontweight="bold", pad=15)

    save_path = os.path.join(GRAPHS_DIR, "posture_pie.png")
    if save:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    return save_path


# ─────────────────────────────────────────────────────────────
# SECTION 3: TREND LINE — Posture Over Time
# ─────────────────────────────────────────────────────────────

def plot_posture_trend(df: pd.DataFrame = None, save=True, last_n=200) -> str:
    """
    Plot posture label over time as a colored step line chart.

    Parameters:
        df     (pd.DataFrame): Posture history
        save   (bool): Save to graphs/
        last_n (int): Use only the last N records

    Returns:
        str: File path of saved chart
    """
    if df is None:
        df = load_history()

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)

    if not df.empty and "posture_label" in df.columns:
        df_plot = df.tail(last_n).copy()
        x = range(len(df_plot))
        y = df_plot["posture_label"].values  # 0=bad, 1=good

        # Color each segment based on posture
        for i in range(len(x) - 1):
            color = GOOD_COLOR if y[i] == 1 else BAD_COLOR
            ax.fill_between([x[i], x[i+1]], [y[i], y[i+1]], alpha=0.3, color=color)
            ax.plot([x[i], x[i+1]], [y[i], y[i+1]], color=color, lw=1.5)

        ax.axhline(y=0.5, color="#444", linestyle="--", alpha=0.4)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Bad", "Good"], color="white", fontsize=10)
        ax.set_xlabel("Frame / Record #", color="white", fontsize=10)

    else:
        ax.text(0.5, 0.5, "No history data available", ha="center",
                va="center", color="#888", transform=ax.transAxes, fontsize=12)

    ax.set_title("Posture Trend Over Time", color="white",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax.spines[:].set_color("#333")

    save_path = os.path.join(GRAPHS_DIR, "posture_trend.png")
    if save:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    return save_path


# ─────────────────────────────────────────────────────────────
# SECTION 4: ANGLE HISTORY GRAPH
# ─────────────────────────────────────────────────────────────

def plot_angle_history(df: pd.DataFrame = None, save=True, last_n=150) -> str:
    """
    Plot neck angle, spine angle, and shoulder slope over time.

    Parameters:
        df     (pd.DataFrame): Posture history
        save   (bool): Save to graphs/
        last_n (int): Use only the last N records

    Returns:
        str: File path
    """
    if df is None:
        df = load_history()

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Posture Angle History", color="white",
                 fontsize=14, fontweight="bold", y=0.98)

    angle_info = [
        ("neck_angle",     "Neck Angle (°)",     ACCENT1,    25),
        ("spine_angle",    "Spine Angle (°)",    ACCENT2,    20),
        ("shoulder_slope", "Shoulder Slope (°)", WARN_COLOR, 12),
    ]

    df_plot = df.tail(last_n).copy() if not df.empty else pd.DataFrame()

    for ax, (col, ylabel, color, threshold) in zip(axes, angle_info):
        ax.set_facecolor(CARD_BG)
        ax.spines[:].set_color("#333")
        ax.grid(True, alpha=0.2)

        if not df_plot.empty and col in df_plot.columns:
            x = range(len(df_plot))
            y = df_plot[col].fillna(0).values
            ax.plot(x, y, color=color, lw=1.5, alpha=0.9)
            ax.fill_between(x, y, alpha=0.15, color=color)
            ax.axhline(y=threshold, color=BAD_COLOR, linestyle="--",
                       alpha=0.6, label=f"Threshold ({threshold}°)")
            ax.legend(fontsize=8, labelcolor="white",
                      facecolor=CARD_BG, edgecolor="#333")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    color="#666", transform=ax.transAxes)

        ax.set_ylabel(ylabel, color="white", fontsize=9)
        ax.tick_params(colors="white")

    axes[-1].set_xlabel("Record #", color="white", fontsize=10)
    plt.tight_layout()

    save_path = os.path.join(GRAPHS_DIR, "angle_history.png")
    if save:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    return save_path


# ─────────────────────────────────────────────────────────────
# SECTION 5: DAILY STATISTICS BAR CHART
# ─────────────────────────────────────────────────────────────

def plot_daily_stats(df: pd.DataFrame = None, save=True) -> str:
    """
    Plot daily good vs bad posture counts as a stacked bar chart.

    Parameters:
        df   (pd.DataFrame): Posture history
        save (bool): Save to graphs/

    Returns:
        str: File path
    """
    if df is None:
        df = load_history()

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)

    if not df.empty and "date" in df.columns and "posture_text" in df.columns:
        daily = df.groupby(["date", "posture_text"]).size().unstack(fill_value=0)

        good_col = "GOOD POSTURE" if "GOOD POSTURE" in daily.columns else None
        bad_col  = "BAD POSTURE"  if "BAD POSTURE"  in daily.columns else None

        x = range(len(daily))
        dates = list(daily.index)[-14:]  # last 14 days
        daily = daily.tail(14)

        if good_col:
            ax.bar(x, daily[good_col], color=GOOD_COLOR, label="Good Posture",
                   alpha=0.85, width=0.6)
        if bad_col:
            bottom = daily[good_col] if good_col else 0
            ax.bar(x, daily[bad_col], bottom=bottom, color=BAD_COLOR,
                   label="Bad Posture", alpha=0.85, width=0.6)

        ax.set_xticks(range(len(dates)))
        ax.set_xticklabels(dates, rotation=30, ha="right",
                           color="white", fontsize=8)
        ax.legend(facecolor=CARD_BG, labelcolor="white")
    else:
        ax.text(0.5, 0.5, "No daily data available", ha="center",
                va="center", color="#888", transform=ax.transAxes)

    ax.set_title("Daily Posture Statistics", color="white",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Detections", color="white")
    ax.grid(True, alpha=0.2, axis="y")
    ax.spines[:].set_color("#333")

    save_path = os.path.join(GRAPHS_DIR, "daily_stats.png")
    if save:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    return save_path


# ─────────────────────────────────────────────────────────────
# SECTION 6: SESSION SUMMARY
# ─────────────────────────────────────────────────────────────

def compute_session_summary(session_stats_dict: dict) -> dict:
    """
    Compute a human-readable session summary.

    Parameters:
        session_stats_dict (dict): From SessionStats.summary()

    Returns:
        dict: Enriched summary with grades and recommendations
    """
    good_pct = session_stats_dict.get("good_posture_pct", 0)
    duration = session_stats_dict.get("session_duration", 0)

    # Grade
    if good_pct >= 85:
        grade = "A+ Excellent"
        color = GOOD_COLOR
    elif good_pct >= 70:
        grade = "B Good"
        color = ACCENT1
    elif good_pct >= 50:
        grade = "C Needs Improvement"
        color = WARN_COLOR
    else:
        grade = "D Poor"
        color = BAD_COLOR

    hours, rem = divmod(int(duration), 3600)
    mins, secs = divmod(rem, 60)
    duration_str = f"{hours:02d}:{mins:02d}:{secs:02d}"

    return {
        **session_stats_dict,
        "grade":        grade,
        "grade_color":  color,
        "duration_str": duration_str,
        "good_pct_str": f"{good_pct:.1f}%",
        "recommendation": _get_recommendation(good_pct),
    }


def _get_recommendation(good_pct: float) -> str:
    if good_pct >= 85:
        return "Excellent posture! Keep up the great work and maintain regular breaks."
    elif good_pct >= 70:
        return "Good posture overall. Focus on keeping your neck straight."
    elif good_pct >= 50:
        return "Posture needs improvement. Set reminders to sit straight every 20 minutes."
    else:
        return "Poor posture detected frequently. Consider an ergonomic chair and monitor stand."


# ─────────────────────────────────────────────────────────────
# SECTION 7: PDF REPORT
# ─────────────────────────────────────────────────────────────

def generate_pdf_report(session_summary: dict, df: pd.DataFrame = None) -> str:
    """
    Generate a downloadable PDF posture report.

    Parameters:
        session_summary (dict): From compute_session_summary()
        df (pd.DataFrame): Posture history

    Returns:
        str: File path of the generated PDF
    """
    try:
        from fpdf import FPDF

        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # ── Header ───────────────────────────────────────────
        pdf.set_fill_color(15, 15, 26)
        pdf.rect(0, 0, 210, 297, "F")
        pdf.set_text_color(255, 255, 255)

        pdf.set_font("Helvetica", "B", 20)
        pdf.set_y(15)
        pdf.cell(0, 12, "PoseNova Posture Report", ln=True, align="C")
        pdf.set_font("Helvetica", "", 12)
        pdf.cell(0, 8, "Session Report", ln=True, align="C")
        pdf.cell(0, 6, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")

        pdf.ln(10)

        # ── Session Stats ────────────────────────────────────
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Session Summary", ln=True)
        pdf.set_font("Helvetica", "", 11)

        stats = [
            ("Session Duration",   session_summary.get("duration_str", "N/A")),
            ("Total Frames",       str(session_summary.get("total_frames", 0))),
            ("Good Posture",       session_summary.get("good_pct_str", "N/A")),
            ("Bad Posture",        f"{session_summary.get('bad_posture_pct', 0):.1f}%"),
            ("Alerts Triggered",   str(session_summary.get("alert_count", 0))),
            ("Posture Grade",      session_summary.get("grade", "N/A")),
        ]

        for label, value in stats:
            pdf.cell(80, 8, f"  {label}:", border=0)
            pdf.cell(0, 8, value, ln=True)

        pdf.ln(5)

        # ── Recommendation ───────────────────────────────────
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, "Recommendation", ln=True)
        pdf.set_font("Helvetica", "", 11)
        rec = session_summary.get("recommendation", "")
        pdf.multi_cell(0, 7, f"  {rec}")

        pdf.ln(5)

        # ── Tips ─────────────────────────────────────────────
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, "Posture Improvement Tips", ln=True)
        pdf.set_font("Helvetica", "", 10)

        tips = [
            "1. Keep your monitor at eye level to avoid neck strain.",
            "2. Sit with your back fully supported by the chair backrest.",
            "3. Keep both feet flat on the floor.",
            "4. Take a 5-minute break every hour to stand and stretch.",
            "5. Adjust armrests so elbows are at a 90-degree angle.",
            "6. Avoid slouching — maintain the natural curve of your spine.",
        ]
        for tip in tips:
            pdf.cell(0, 7, f"  {tip}", ln=True)

        # ── Save ─────────────────────────────────────────────
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = os.path.join(REPORTS_DIR, f"posture_report_{ts}.pdf")
        pdf.output(pdf_path)
        logger.info(f"[Analytics] PDF report saved: {pdf_path}")
        return pdf_path

    except ImportError:
        logger.warning("[Analytics] fpdf2 not installed. PDF generation skipped.")
        return ""
    except Exception as e:
        logger.error(f"[Analytics] PDF generation error: {e}")
        return ""


# ─────────────────────────────────────────────────────────────
# SECTION 8: GENERATE ALL GRAPHS
# ─────────────────────────────────────────────────────────────

def generate_all_graphs(df: pd.DataFrame = None) -> dict:
    """
    Generate all analytics graphs at once.

    Parameters:
        df (pd.DataFrame): Optional preloaded history DataFrame

    Returns:
        dict: {graph_name: file_path}
    """
    if df is None:
        df = load_history()

    paths = {}
    try:
        paths["pie"]         = plot_posture_pie(df)
        paths["trend"]       = plot_posture_trend(df)
        paths["angles"]      = plot_angle_history(df)
        paths["daily"]       = plot_daily_stats(df)
    except Exception as e:
        logger.error(f"[Analytics] Graph generation error: {e}")

    return paths
