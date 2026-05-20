"""
============================================================
app.py  —  Streamlit Dashboard
AI-Based Real-Time Human Posture Detection and Correction System
============================================================
Run with:  streamlit run app.py
============================================================
"""

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import os
import sys
import pickle
import threading
from PIL import Image

# ── page config (MUST be first Streamlit call) ───────────────
st.set_page_config(
    page_title="PoseNova — AI Posture System",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ── local imports ────────────────────────────────────────────
from posture_detection import PostureDetector
from analytics import (
    plot_posture_pie, plot_posture_trend,
    plot_angle_history, plot_daily_stats,
    generate_pdf_report, compute_session_summary,
)
from utils import (
    load_posture_history, save_screenshot,
    get_session_id, init_history_csv,
)

from posture_detection import (
    BaseOptions, PoseLandmarker, PoseLandmarkerOpts, RunningMode, MODEL_PATH
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400..900;1,400..900&family=Outfit:wght@300;400;600;700&display=swap');

  /* Base Typography & Color Reset */
  html, body, [class*="css"] { 
      font-family: 'Outfit', sans-serif; 
      color: #4b423f;
  }
  
  h1, h2, h3 {
      font-family: 'Playfair Display', serif !important;
      font-weight: 600 !important;
      color: #7d4c62 !important;
      letter-spacing: -0.5px;
  }

  /* 🌌 Blooming Plum, Sage & Cream Canvas */
  .stApp {
    background-color: #faf7f0;
    color: #4b423f;
  }

  /* 🧊 Sidebar - Warm Cream & Sage Frosted Panel */
  [data-testid="stSidebar"] {
    background: #faf7f0 !important;
    border-right: 1px solid rgba(91, 105, 75, 0.18);
    box-shadow: 2px 0 15px rgba(91, 105, 75, 0.05);
  }
  /* Make all labels, selectbox elements, sliders, and toggle texts inside the sidebar highly visible */
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] span,
  [data-testid="stSidebar"] small,
  [data-testid="stSidebar"] div {
    color: #7d4c62 !important;
  }
  /* Preserve white text inside buttons */
  [data-testid="stSidebar"] button * {
    color: #ffffff !important;
  }
  /* Style Selectbox to match cream theme and be perfectly readable (both closed and open) */
  div[data-baseweb="select"] > div,
  [data-testid="stSidebar"] div[data-baseweb="select"] > div,
  div[data-baseweb="select"],
  [data-testid="stSidebar"] div[data-baseweb="select"] {
    background-color: #ffffff !important;
    border-radius: 8px !important;
    border: 1px solid #5b694b !important;
  }
  div[data-baseweb="select"] *,
  [data-testid="stSidebar"] div[data-baseweb="select"] * {
    color: #7d4c62 !important;
  }

  /* 📊 Metric Cards - Crisp White with Sage-Green Outline */
  [data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #5b694b;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 15px rgba(91, 105, 75, 0.04);
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
  }
  [data-testid="metric-container"]:hover {
    transform: translateY(-3px);
    border-color: #7d4c62;
    box-shadow: 0 8px 25px rgba(125, 76, 98, 0.08);
  }
  
  /* Metric Text Colors - Editorial Slate */
  [data-testid="stMetricValue"] {
      font-family: 'Playfair Display', serif !important;
      font-weight: 700 !important;
      color: #5b694b !important;
  }
  [data-testid="stMetricLabel"] {
      color: #7d4c62 !important;
      font-weight: 600;
      text-transform: uppercase;
      font-size: 0.75rem !important;
      letter-spacing: 1.5px;
  }

  /* 🕹️ Sage-Green Capsule Buttons - Fills & Hovers in Dusty Plum */
  .stButton > button,
  .stDownloadButton > button {
    background: #5b694b;
    color: #ffffff;
    border: none;
    border-radius: 30px; /* Pill shape */
    padding: 10px 12px;
    font-weight: 600;
    font-family: 'Outfit', sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    width: 100%;
    font-size: 0.74rem;
    box-shadow: 0 4px 12px rgba(91, 105, 75, 0.15);
  }
  .stButton > button:hover,
  .stDownloadButton > button:hover {
    color: #ffffff !important;
    background: #7d4c62 !important;
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(125, 76, 98, 0.25);
  }

  /* 🔗 Streamlit Selectbox Listbox Popovers - Beautiful High-Contrast Cream & Plum */
  div[data-baseweb="popover"] *,
  div[role="listbox"] *,
  ul[role="listbox"] * {
    color: #7d4c62 !important;
    background-color: #ffffff !important;
  }
  div[data-baseweb="popover"] li,
  div[role="listbox"] li,
  ul[role="listbox"] li {
    color: #7d4c62 !important;
    background-color: #ffffff !important;
  }
  div[data-baseweb="popover"] li:hover,
  div[role="listbox"] li:hover {
    background-color: #faf7f0 !important;
    color: #5b694b !important;
  }

  /* 🟢 Posture Badges - Blooming Editorial Solid Blocks */
  .good-badge {
    background: #5b694b;
    border: none;
    border-radius: 12px;
    padding: 22px;
    text-align: center;
    color: #ffffff !important;
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    box-shadow: 0 10px 25px rgba(91, 105, 75, 0.15);
    animation: goodPulse 3s infinite alternate;
  }
  @keyframes goodPulse {
    from { transform: scale(1); }
    to { transform: scale(1.01); }
  }

  .bad-badge {
    background: #7d4c62;
    border: none;
    border-radius: 12px;
    padding: 22px;
    text-align: center;
    color: #ffffff !important;
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    box-shadow: 0 10px 25px rgba(125, 76, 98, 0.2);
    animation: dangerAlert 0.8s infinite;
  }
  @keyframes dangerAlert {
    0%   { transform: scale(1); box-shadow: 0 0 0 0 rgba(125, 76, 98, 0.4); }
    50%  { transform: scale(1.01); box-shadow: 0 0 0 12px rgba(125, 76, 98, 0); }
    100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(125, 76, 98, 0); }
  }

  /* 🛸 Blooming Birth Nest Banner - Dusty Rose-Plum hero block */
  .app-header {
    background: linear-gradient(135deg, #7d4c62, #683b50);
    border-radius: 20px;
    padding: 40px 25px;
    margin-bottom: 35px;
    text-align: center;
    position: relative;
    box-shadow: 0 8px 25px rgba(125, 76, 98, 0.15);
  }
  .app-header h1 {
    font-size: 3rem;
    font-weight: 600;
    margin: 0;
    letter-spacing: -1px;
    color: #ffffff !important;
    font-family: 'Playfair Display', serif !important;
  }
  .app-header p {
    color: #faf7f0;
    margin: 12px 0 0 0;
    font-size: 0.8rem;
    font-family: 'Outfit', sans-serif;
    letter-spacing: 3px;
    text-transform: uppercase;
    font-weight: 600;
  }

  /* 🪟 Interactive Luxury Info Cards */
  .info-card {
    background: #ffffff;
    border: 1px solid rgba(91, 105, 75, 0.15);
    border-left: 3px solid #5b694b;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 15px;
    color: #4b423f;
    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.02);
    transition: all 0.3s ease;
  }
  .info-card:hover {
    border-left-color: #7d4c62;
    transform: translateX(4px);
    box-shadow: 0 5px 15px rgba(125, 76, 98, 0.05);
  }

  /* 🌡️ Thin Muted Sage Angle Gauge Bars */
  .angle-bar-outer {
    background: rgba(91, 105, 75, 0.08);
    border-radius: 6px;
    height: 8px;
    width: 100%;
    margin-top: 8px;
    overflow: hidden;
  }

  /* 🏷️ Elegant Editorial Section Headers */
  .section-title {
    color: #7d4c62;
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    display: inline-block;
    position: relative;
    margin: 25px 0 15px 0;
  }
  .section-title::after {
      content: '';
      position: absolute;
      bottom: -6px; left: 0;
      width: 25px; height: 1px;
      background: #5b694b;
      transition: width 0.3s ease;
  }
  .section-title:hover::after {
      width: 100%;
      background: #7d4c62;
  }

  /* Toggles & Sliders Color Customization */
  .stToggle>div>div {
      background-color: #5b694b !important;
  }

  /* Tab Styling */
  .stTabs [data-baseweb="tab-list"] {
      gap: 12px;
      border-bottom: 2px solid rgba(91, 105, 75, 0.15);
  }
  .stTabs [data-baseweb="tab"] {
      background-color: transparent;
      padding: 10px 16px;
      color: #7d4c62;
      font-family: 'Playfair Display', serif;
      font-size: 1.1rem;
      opacity: 0.6;
  }
  .stTabs [aria-selected="true"] {
      opacity: 1;
      font-weight: bold;
      border-bottom: 2px solid #5b694b !important;
  }

  /* Hide Streamlit default noise */
  #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "camera_running":   False,
        "sound_enabled":    True,
        "voice_enabled":    True,
        "screenshot_bad":   False,
        "bad_threshold":    2.0,
        "alert_cooldown":   6.0,
        "detector":         None,
        "pose_engine":      None,
        "last_metrics":     {},
        "session_id":       get_session_id(),
        "frame_placeholder": None,
        "use_ml":           True,
        "camera_index":     0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()
init_history_csv()


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def bgr_to_rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def angle_bar(label, value, max_val, good_thresh, bad_thresh):
    """Render an angle label + colored progress bar."""
    pct    = min(100, int(100 * value / max_val))
    color  = "#10b981" if value < good_thresh else ("#f59e0b" if value < bad_thresh else "#ef4444")
    st.markdown(f"""
    <div style="margin-bottom:10px;">
      <div style="display:flex;justify-content:space-between;font-size:0.8rem;color:#aaa;">
        <span>{label}</span><span style="color:{color};font-weight:700;">{value:.1f}°</span>
      </div>
      <div class="angle-bar-outer">
        <div style="background:{color};width:{pct}%;height:8px;border-radius:6px;
                    transition:width 0.4s ease;"></div>
      </div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌀 PoseNova System")
    st.markdown("---")

    # ── Camera controls ───────────────────────────────────────
    st.markdown('<div class="section-title">📷 Camera</div>', unsafe_allow_html=True)
    st.session_state.camera_index = st.selectbox(
        "Camera Index", [0, 1, 2], index=0,
        help="0 = default webcam"
    )

    def start_camera_cb():
        if not st.session_state.camera_running:
            try:
                detector = PostureDetector(
                    camera_index=st.session_state.camera_index,
                    sound_enabled=st.session_state.sound_enabled,
                    voice_enabled=st.session_state.voice_enabled,
                    use_ml_model=st.session_state.use_ml,
                    bad_posture_threshold=st.session_state.bad_threshold,
                    alert_cooldown=st.session_state.alert_cooldown,
                    screenshot_on_bad=st.session_state.screenshot_bad,
                )
                options = PoseLandmarkerOpts(
                    base_options=BaseOptions(model_asset_path=MODEL_PATH),
                    running_mode=RunningMode.IMAGE,
                    num_poses=1,
                    min_pose_detection_confidence=0.3,
                    min_tracking_confidence=0.3,
                )
                pose_engine = PoseLandmarker.create_from_options(options)
                detector.start()
                st.session_state.detector = detector
                st.session_state.pose_engine = pose_engine
                st.session_state.camera_running = True
            except Exception as e:
                st.error(f"Failed to start camera: {e}")

    def stop_camera_cb():
        if st.session_state.camera_running:
            if st.session_state.detector:
                st.session_state.detector.stop()
            if st.session_state.pose_engine:
                st.session_state.pose_engine.close()
            st.session_state.camera_running = False
            st.session_state.detector = None
            st.session_state.pose_engine = None

    col1, col2 = st.columns(2)
    with col1:
        st.button("▶ START", key="btn_start", on_click=start_camera_cb, use_container_width=True)
    with col2:
        st.button("⏹ STOP",  key="btn_stop", on_click=stop_camera_cb, use_container_width=True)

    screenshot_btn = st.button("📸 Screenshot", key="btn_ss", use_container_width=True)

    st.markdown("---")

    # ── Settings ──────────────────────────────────────────────
    st.markdown('<div class="section-title">⚙️ Settings</div>', unsafe_allow_html=True)

    st.session_state.sound_enabled  = st.toggle("🔊 Sound Alert",  value=st.session_state.sound_enabled)
    st.session_state.voice_enabled  = st.toggle("🗣️ Voice Alert",  value=st.session_state.voice_enabled)
    st.session_state.use_ml         = st.toggle("🤖 Use ML Model", value=st.session_state.use_ml,
                                                 help="Uncheck to use rule-based detection")
    st.session_state.screenshot_bad = st.toggle("📸 Auto-screenshot on bad posture",
                                                  value=st.session_state.screenshot_bad)

    st.session_state.bad_threshold  = st.slider(
        "Bad Posture Alert Threshold (s)", 1.0, 10.0,
        value=st.session_state.bad_threshold, step=0.5,
        help="How many seconds of bad posture before alert fires"
    )
    st.session_state.alert_cooldown = st.slider(
        "Alert Cooldown (s)", 3.0, 30.0,
        value=st.session_state.alert_cooldown, step=1.0
    )

    st.markdown("---")

    # ── Analytics menu ────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Analytics</div>', unsafe_allow_html=True)
    gen_graphs_btn = st.button("📈 Generate Graphs",   key="btn_graphs", use_container_width=True)
    gen_pdf_btn    = st.button("📄 Download PDF Report", key="btn_pdf", use_container_width=True)

    st.markdown("---")
    st.markdown('<div style="color:#64748b;font-size:0.75rem;text-align:center;font-weight:600;">PoseNova v1.0<br>MediaPipe + Scikit-learn</div>',
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# BUTTON HANDLERS
# ─────────────────────────────────────────────────────────────
# Handlers are now attached via on_click callbacks in the sidebar


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <h1>🌀 PoseNova</h1>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MAIN LAYOUT — two-column (metrics | live feed)
# ─────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 2], gap="medium")

# ── LEFT COLUMN — status panel ────────────────────────────────
with left_col:
    # 1. Posture Status Badge
    badge_ph = st.empty()
    st.markdown("")

    # 2. Confidence + FPS metrics row
    m1, m2, m3 = st.columns(3)
    conf_ph = m1.empty()
    fps_ph  = m2.empty()
    timer_ph = m3.empty()
    st.markdown("")

    # 3. Angle gauges
    st.markdown('<div class="section-title">📐 Posture Angles</div>', unsafe_allow_html=True)
    angles_ph = st.empty()
    st.markdown("")

    # 4. Session stats
    st.markdown('<div class="section-title">📊 Session Stats</div>', unsafe_allow_html=True)
    sess_ph = st.empty()
    st.markdown("")

    # 5. Tips
    tips_ph = st.empty()

    # Helper function to render left column metrics
    def render_metrics(metrics):
        # Badge
        posture_text = metrics.get("posture_text", "—")
        is_bad       = metrics.get("is_bad", False)
        bad_posture_note = metrics.get("bad_posture_note", "")

        if posture_text == "GOOD POSTURE":
            badge_ph.markdown('<div class="good-badge">✅ GOOD POSTURE</div>', unsafe_allow_html=True)
        elif posture_text == "BAD POSTURE":
            if bad_posture_note:
                badge_ph.markdown(f'<div class="bad-badge">⚠️ BAD POSTURE ({bad_posture_note})</div>', unsafe_allow_html=True)
            else:
                badge_ph.markdown('<div class="bad-badge">⚠️ BAD POSTURE</div>', unsafe_allow_html=True)
        else:
            badge_ph.markdown(f'<div class="info-card" style="text-align:center;color:#666;">— {posture_text} —</div>', unsafe_allow_html=True)

        # Top Metrics
        conf_ph.metric("Confidence",  f"{metrics.get('confidence', 0)*100:.1f}%")
        fps_ph.metric("FPS",         f"{metrics.get('fps', 0):.0f}")
        timer_ph.metric("Bad Timer",   f"{metrics.get('bad_duration', 0):.1f}s")

        # Angles
        html_angles = ""
        def ab(label, val, max_val, good_t, bad_t):
            pct = min(100, int(100 * val / max_val))
            c = "#5b694b" if val < good_t else ("#f59e0b" if val < bad_t else "#7d4c62")
            return f'''<div style="margin-bottom:10px;">
                <div style="display:flex;justify-content:space-between;font-size:0.8rem;color:#7d4c62;">
                <span>{label}</span><span style="color:{c};font-weight:700;">{val:.1f}°</span></div>
                <div class="angle-bar-outer"><div style="background:{c};width:{pct}%;height:8px;border-radius:4px;transition:width 0.2s ease;"></div></div></div>'''
        
        html_angles += ab("Neck Angle", metrics.get("neck_avg", 0), 60, 25, 35)
        html_angles += ab("Spine Angle", metrics.get("spine_angle", 0), 60, 15, 22)
        html_angles += ab("Shoulder Slope", metrics.get("shoulder_slope", 0), 45, 8, 12)
        html_angles += ab("Head Tilt", metrics.get("head_tilt", 0), 30, 8, 12)
        angles_ph.markdown(html_angles, unsafe_allow_html=True)

        # Session Stats
        sess = metrics.get("session", {})
        if sess:
            # We can't use columns inside empty easily without rewriting, so we use markdown grid
            sess_html = f'''<div style="display:flex;gap:20px;margin-bottom:10px;">
                <div><div style="font-size:0.8rem;color:#7d4c62;">Good %</div><div style="font-size:1.5rem;font-weight:700;color:#5b694b;">{sess.get('good_posture_pct',0):.1f}%</div></div>
                <div><div style="font-size:0.8rem;color:#7d4c62;">Alerts</div><div style="font-size:1.5rem;font-weight:700;color:#5b694b;">{sess.get('alert_count',0)}</div></div>
                <div><div style="font-size:0.8rem;color:#7d4c62;">Duration</div><div style="font-size:1.5rem;font-weight:700;color:#5b694b;">{int(sess.get('session_duration',0))}s</div></div>
                </div>'''
            sess_ph.markdown(sess_html, unsafe_allow_html=True)
            
        # Tips
        tips = metrics.get("tips", [])
        if tips:
            tips_html = '<div class="section-title">💡 Posture Tips</div>'
            for tip in tips[:3]:
                tips_html += f'<div class="info-card" style="font-size:0.82rem;color:#4b423f;">{tip}</div>'
            tips_ph.markdown(tips_html, unsafe_allow_html=True)
        else:
            tips_ph.empty()

    # Initial render if camera is not running but we have old metrics
    if not st.session_state.camera_running and st.session_state.last_metrics:
        render_metrics(st.session_state.last_metrics)
    elif not st.session_state.camera_running:
        badge_ph.markdown('<div class="info-card" style="text-align:center;color:#7d4c62;">— Ready to Start —</div>', unsafe_allow_html=True)


    # Calibrate / Audio toggle inline buttons
    st.markdown("")
    cb1, cb2 = st.columns(2)
    with cb1:
        if st.button("🎯 Calibrate", key="btn_cal", use_container_width=True):
            if st.session_state.detector:
                st.session_state.detector.alert_manager.trigger_good_posture_clear()
                st.toast("Calibrated! Sit upright as reference.", icon="✅")
    with cb2:
        audio_label = "🔇 Mute" if st.session_state.sound_enabled else "🔊 Unmute"
        if st.button(audio_label, key="btn_audio", use_container_width=True):
            st.session_state.sound_enabled = not st.session_state.sound_enabled
            if st.session_state.detector:
                st.session_state.detector.alert_manager.toggle_sound()


# ── RIGHT COLUMN — live video feed ───────────────────────────
with right_col:
    st.markdown('<div class="section-title">📹 Live Feed</div>', unsafe_allow_html=True)
    frame_placeholder = st.empty()

    # Status bar below feed
    status_bar = st.empty()

    if not st.session_state.camera_running:
        # Show placeholder
        frame_placeholder.markdown("""
        <div style="background:#0d0d25;border:2px dashed #2a2a5e;border-radius:16px;
                    height:420px;display:flex;align-items:center;justify-content:center;
                    flex-direction:column;gap:12px;">
          <div style="font-size:3rem;">📷</div>
          <div style="color:#555;font-size:1rem;">Camera not started</div>
          <div style="color:#3a3a6a;font-size:0.82rem;">Press ▶ START in the sidebar</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LIVE DETECTION LOOP
# ─────────────────────────────────────────────────────────────
if st.session_state.camera_running:
    detector    = st.session_state.detector
    pose_engine = st.session_state.pose_engine

    if detector and pose_engine:
        # High-performance while loop to prevent UI blinking
        last_metric_update = 0
        while st.session_state.camera_running:
            frame, metrics = detector.process_frame(pose_engine)

            if frame is not None:
                # Display frame (OpenCV handle overlays cleanly for both posture & no_person state)
                rgb_frame = bgr_to_rgb(frame)
                frame_placeholder.image(
                    rgb_frame, channels="RGB",
                    use_column_width=True,
                    caption="Real-time posture analysis with MediaPipe skeleton overlay"
                )

                st.session_state.last_metrics = metrics
                
                # Update sidebar metrics instantly without full page reload
                # Throttle UI text updates to prevent WebSocket flooding and flickering
                current_time = time.time()
                if current_time - last_metric_update > 0.5:
                    render_metrics(metrics)
                    last_metric_update = current_time

            # Check for stop click or sleep to yield
            time.sleep(0.01)

# ─────────────────────────────────────────────────────────────
# TABS — History / Analytics / Model Info
# ─────────────────────────────────────────────────────────────
st.markdown("---")
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Posture History",
    "📊 Analytics",
    "🤖 Model Info",
    "ℹ️ About"
])

# ── TAB 1: History ────────────────────────────────────────────
with tab1:
    st.markdown("### 📋 Posture Detection History")
    df = load_posture_history()
    if df.empty:
        st.info("No posture history yet. Start the camera to begin logging.")
    else:
        # Summary metrics
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Total Records",   len(df))
        good_n = (df["posture_text"] == "GOOD POSTURE").sum()
        bad_n  = (df["posture_text"] == "BAD POSTURE").sum()
        h2.metric("Good Posture",    good_n)
        h3.metric("Bad Posture",     bad_n)
        h4.metric("Good %",          f"{100*good_n/max(len(df),1):.1f}%")

        st.markdown("#### Recent Records")
        display_df = df[["timestamp","date","time","posture_text",
                          "confidence","neck_angle","spine_angle","shoulder_slope"]].tail(50)
        st.dataframe(display_df[::-1], use_container_width=True, height=300)

        # Download CSV
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Full History CSV",
            data=csv_data,
            file_name="posture_history.csv",
            mime="text/csv",
        )

# ── TAB 2: Analytics ─────────────────────────────────────────
with tab2:
    st.markdown("### 📊 Analytics Dashboard")

    df_hist = load_posture_history()

    if gen_graphs_btn or not df_hist.empty:
        g1, g2 = st.columns(2)

        with g1:
            st.markdown("#### Posture Distribution")
            pie_path = plot_posture_pie(df_hist if not df_hist.empty else None)
            if os.path.exists(pie_path):
                st.image(pie_path, use_container_width=True)

        with g2:
            st.markdown("#### Daily Statistics")
            daily_path = plot_daily_stats(df_hist if not df_hist.empty else None)
            if os.path.exists(daily_path):
                st.image(daily_path, use_container_width=True)

        st.markdown("#### Posture Trend Over Time")
        trend_path = plot_posture_trend(df_hist if not df_hist.empty else None)
        if os.path.exists(trend_path):
            st.image(trend_path, use_container_width=True)

        st.markdown("#### Angle History")
        angle_path = plot_angle_history(df_hist if not df_hist.empty else None)
        if os.path.exists(angle_path):
            st.image(angle_path, use_container_width=True)
    else:
        st.info("Click **Generate Graphs** in the sidebar after collecting some posture data.")

    # PDF Report
    if gen_pdf_btn:
        sess_data = st.session_state.last_metrics.get("session", {})
        if not sess_data:
            sess_data = {"good_posture_pct": 0, "bad_posture_pct": 0,
                         "total_frames": 0, "alert_count": 0, "session_duration": 0}
        summary = compute_session_summary(sess_data)
        pdf_path = generate_pdf_report(summary, df_hist)
        if pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "📄 Download PDF Report",
                    data=f.read(),
                    file_name=os.path.basename(pdf_path),
                    mime="application/pdf",
                )
            st.success("PDF report generated!")

# ── TAB 3: Model Info ─────────────────────────────────────────
with tab3:
    st.markdown("### 🤖 Machine Learning Model Info")

    model_info_path = os.path.join(BASE_DIR, "models", "model_info.pkl")
    if os.path.exists(model_info_path):
        with open(model_info_path, "rb") as f:
            info = pickle.load(f)

        mi1, mi2, mi3, mi4 = st.columns(4)
        mi1.metric("Best Model",  info.get("model_name", "N/A"))
        mi2.metric("Accuracy",    f"{info.get('accuracy', 0)*100:.2f}%")
        mi3.metric("F1-Score",    f"{info.get('f1', 0):.4f}")
        mi4.metric("Precision",   f"{info.get('precision', 0):.4f}")

        st.markdown("#### Feature Names")
        feat = info.get("feature_names", [])
        if feat:
            cols = st.columns(3)
            for i, name in enumerate(feat):
                cols[i % 3].markdown(f"- `{name}`")

        # Show model comparison graph if exists
        cmp_path = os.path.join(BASE_DIR, "graphs", "model_comparison.png")
        cm_path  = os.path.join(BASE_DIR, "graphs", "confusion_matrix.png")
        fi_path  = os.path.join(BASE_DIR, "graphs", "feature_importance.png")

        if os.path.exists(cmp_path):
            st.markdown("#### Model Comparison")
            st.image(cmp_path, use_container_width=True)

        gc1, gc2 = st.columns(2)
        if os.path.exists(cm_path):
            with gc1:
                st.markdown("#### Confusion Matrix")
                st.image(cm_path, use_container_width=True)
        if os.path.exists(fi_path):
            with gc2:
                st.markdown("#### Feature Importance")
                st.image(fi_path, use_container_width=True)

        # Train button
        st.markdown("---")
        if st.button("🔄 Re-train Model", key="btn_retrain"):
            with st.spinner("Training ML models... this may take a moment."):
                try:
                    from train_model import run_training
                    result = run_training()
                    st.success(f"Training complete! Best model: **{result['best_model']}** "
                               f"— Accuracy: **{result['accuracy']*100:.2f}%**")
                    st.rerun()
                except Exception as e:
                    st.error(f"Training error: {e}")
    else:
        st.warning("No trained model found.")
        st.info("Run `python train_model.py` in the terminal, or click below:")
        if st.button("🚀 Train Model Now", key="btn_train_now"):
            with st.spinner("Running preprocessing + training pipeline..."):
                try:
                    from train_model import run_training
                    result = run_training()
                    st.success(f"Done! Best: **{result['best_model']}** "
                               f"— Accuracy: **{result['accuracy']*100:.2f}%**")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

# ── TAB 4: About ─────────────────────────────────────────────
with tab4:
    st.markdown("""
### ℹ️ About This Project

**PoseNova — AI-Based Real-Time Human Posture Detection and Correction System**

| Item | Detail |
|---|---|
| **Language** | Python 3.11+ |
| **Pose Engine** | MediaPipe Pose (33 landmarks) |
| **ML Models** | Random Forest, Logistic Regression, SVM, KNN |
| **GUI** | Streamlit |
| **Alert System** | winsound / pygame + pyttsx3 TTS |
| **Output** | Live skeleton overlay, CSV logs, PDF reports, screenshots |

#### How It Works
1. Webcam frame is captured and mirrored  
2. MediaPipe detects 33 body landmarks  
3. Angles are computed (neck, spine, shoulders, head)  
4. ML model classifies posture as GOOD or BAD  
5. Visual overlay drawn: green skeleton (good) / red skeleton (bad)  
6. Alert fires if bad posture persists beyond threshold  
7. All data logged to CSV for analytics  

#### Posture Detection Thresholds
| Angle | Good | Warning | Bad |
|---|---|---|---|
| Neck | < 20° | 20–35° | > 35° |
| Spine | < 15° | 15–30° | > 30° |
| Shoulder Slope | < 8° | 8–18° | > 18° |
| Head Tilt | < 8° | 8–15° | > 15° |

#### Keyboard Shortcuts (OpenCV mode only)
- **Q** — Quit  
- **S** — Save screenshot  
- **T** — Toggle sound  
""")
