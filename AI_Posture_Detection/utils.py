"""
============================================================
utils.py
AI-Based Real-Time Human Posture Detection and Correction System
============================================================
Shared utility functions used across all modules:

- Model loading / saving
- CSV posture history logging
- Screenshot capture
- FPS calculation
- Posture correction tips
- Timestamp utilities
- Session statistics
============================================================
"""

import os
import csv
import pickle
import time
import datetime
import logging
import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR       = os.path.join(BASE_DIR, "models")
SCREENSHOTS_DIR  = os.path.join(BASE_DIR, "screenshots")
LOGS_DIR         = os.path.join(BASE_DIR, "logs")
REPORTS_DIR      = os.path.join(BASE_DIR, "reports")
HISTORY_CSV      = os.path.join(BASE_DIR, "posture_history.csv")

for d in [MODELS_DIR, SCREENSHOTS_DIR, LOGS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "posture_system.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# SECTION 1: MODEL I/O
# ─────────────────────────────────────────────────────────────

def load_model_artifacts():
    """
    Load the trained posture model, scaler, and metadata from disk.

    Returns:
        tuple: (model, scaler, model_info_dict) or (None, None, None) on failure
    """
    model_path = os.path.join(MODELS_DIR, "posture_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    info_path   = os.path.join(MODELS_DIR, "model_info.pkl")

    if not os.path.exists(model_path):
        logger.warning("[Utils] posture_model.pkl not found. Run train_model.py first.")
        return None, None, {}

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        info = {}
        if os.path.exists(info_path):
            with open(info_path, "rb") as f:
                info = pickle.load(f)
        logger.info(f"[Utils] Model loaded: {info.get('model_name', 'Unknown')} | "
                    f"Accuracy: {info.get('accuracy', 0)*100:.1f}%")
        return model, scaler, info
    except Exception as e:
        logger.error(f"[Utils] Failed to load model: {e}")
        return None, None, {}


# ─────────────────────────────────────────────────────────────
# SECTION 2: POSTURE PREDICTION
# ─────────────────────────────────────────────────────────────

def predict_posture(model, scaler, feature_vector):
    """
    Run the ML model on a feature vector to predict posture.

    Parameters:
        model          : Loaded sklearn model
        scaler         : Fitted StandardScaler
        feature_vector (np.ndarray): Raw feature vector (1-D)

    Returns:
        label      (int): 0=bad, 1=good (or based on label encoding)
        confidence (float): Prediction confidence 0.0–1.0
    """
    if model is None or feature_vector is None:
        return 1, 0.5  # default: assume good posture

    try:
        X = feature_vector.reshape(1, -1)
        X_scaled = scaler.transform(X)

        label = model.predict(X_scaled)[0]

        # Get probability if supported
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_scaled)[0]
            confidence = float(proba.max())
        else:
            confidence = 0.75  # default confidence for models without proba

        return int(label), confidence

    except Exception as e:
        logger.error(f"[Utils] Prediction error: {e}")
        return 1, 0.5


def label_to_text(label_int, label_names=None):
    """
    Convert integer label to human-readable posture text.

    Parameters:
        label_int   (int): Numeric label (0 or 1)
        label_names (list): Optional label name list from model_info

    Returns:
        str: "GOOD POSTURE" or "BAD POSTURE"
    """
    if label_names:
        raw = label_names[label_int] if label_int < len(label_names) else "unknown"
    else:
        raw = "good" if label_int == 1 else "bad"

    return "GOOD POSTURE" if "good" in raw.lower() else "BAD POSTURE"


def is_bad_posture(label_int, label_names=None):
    """Return True if the predicted label represents bad posture."""
    text = label_to_text(label_int, label_names)
    return "BAD" in text


# ─────────────────────────────────────────────────────────────
# SECTION 3: CSV POSTURE HISTORY LOGGING
# ─────────────────────────────────────────────────────────────

CSV_HEADER = [
    "timestamp", "date", "time",
    "posture_label", "posture_text",
    "confidence", "neck_angle", "spine_angle",
    "shoulder_slope", "head_tilt", "session_id"
]


def init_history_csv():
    """Create posture_history.csv with headers if it doesn't exist."""
    if not os.path.exists(HISTORY_CSV):
        with open(HISTORY_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)
        logger.info(f"[Utils] Created posture history: {HISTORY_CSV}")


def log_posture_to_csv(posture_text, confidence, angles_dict, session_id="default"):
    """
    Append one posture detection record to posture_history.csv.

    Parameters:
        posture_text (str): "GOOD POSTURE" or "BAD POSTURE"
        confidence   (float): Model confidence 0.0–1.0
        angles_dict  (dict): Angle features from angle_calculation
        session_id   (str): Session identifier
    """
    try:
        now = datetime.datetime.now()
        row = [
            now.isoformat(),
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            0 if "BAD" in posture_text else 1,
            posture_text,
            round(confidence, 4),
            angles_dict.get("neck_avg", 0),
            angles_dict.get("spine_angle", 0),
            angles_dict.get("shoulder_slope", 0),
            angles_dict.get("head_tilt", 0),
            session_id,
        ]
        with open(HISTORY_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
    except Exception as e:
        logger.error(f"[Utils] CSV logging error: {e}")


def load_posture_history():
    """
    Load posture history CSV into a pandas DataFrame.

    Returns:
        pd.DataFrame: Posture history or empty DataFrame
    """
    try:
        import pandas as pd
        if os.path.exists(HISTORY_CSV):
            df = pd.read_csv(HISTORY_CSV)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            return df
        return pd.DataFrame(columns=CSV_HEADER)
    except Exception as e:
        logger.error(f"[Utils] Load history error: {e}")
        import pandas as pd
        return pd.DataFrame(columns=CSV_HEADER)


# ─────────────────────────────────────────────────────────────
# SECTION 4: SCREENSHOT CAPTURE
# ─────────────────────────────────────────────────────────────

def save_screenshot(frame, prefix="posture"):
    """
    Save a screenshot of the current webcam frame to screenshots/.

    Parameters:
        frame  (np.ndarray): OpenCV BGR image frame
        prefix (str): Filename prefix

    Returns:
        str: Saved file path or None on failure
    """
    try:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{prefix}_{ts}.png"
        filepath = os.path.join(SCREENSHOTS_DIR, filename)
        cv2.imwrite(filepath, frame)
        logger.info(f"[Utils] Screenshot saved: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"[Utils] Screenshot error: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# SECTION 5: FPS CALCULATOR
# ─────────────────────────────────────────────────────────────

class FPSCounter:
    """
    Lightweight real-time FPS calculator using exponential moving average.

    Usage:
        fps_counter = FPSCounter()
        fps = fps_counter.update()
    """

    def __init__(self, smoothing=0.9):
        self._prev_time = time.time()
        self._fps       = 0.0
        self._smoothing = smoothing  # EMA factor

    def update(self) -> float:
        """
        Call once per frame. Returns current FPS.

        Returns:
            float: Smoothed frames per second
        """
        now      = time.time()
        delta    = now - self._prev_time
        self._prev_time = now

        instant_fps = 1.0 / (delta + 1e-6)
        self._fps = (self._smoothing * self._fps +
                     (1 - self._smoothing) * instant_fps)
        return round(self._fps, 1)

    @property
    def fps(self) -> float:
        return self._fps


# ─────────────────────────────────────────────────────────────
# SECTION 6: SESSION STATISTICS
# ─────────────────────────────────────────────────────────────

class SessionStats:
    """
    Tracks real-time posture statistics for the current session.

    Attributes:
        total_frames    (int): Total frames processed
        good_frames     (int): Frames where good posture was detected
        bad_frames      (int): Frames where bad posture was detected
        session_start   (float): Unix timestamp when session started
        alert_count     (int): Number of alerts triggered
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_frames  = 0
        self.good_frames   = 0
        self.bad_frames    = 0
        self.session_start = time.time()
        self.alert_count   = 0
        self.screenshot_count = 0

    def update(self, is_good: bool):
        """
        Update stats with the current frame's posture result.

        Parameters:
            is_good (bool): True if good posture detected this frame
        """
        self.total_frames += 1
        if is_good:
            self.good_frames += 1
        else:
            self.bad_frames += 1

    def increment_alerts(self):
        self.alert_count += 1

    def increment_screenshots(self):
        self.screenshot_count += 1

    @property
    def session_duration(self) -> float:
        """Session duration in seconds."""
        return round(time.time() - self.session_start, 1)

    @property
    def good_posture_percent(self) -> float:
        """Percentage of frames with good posture."""
        if self.total_frames == 0:
            return 0.0
        return round(100 * self.good_frames / self.total_frames, 1)

    @property
    def bad_posture_percent(self) -> float:
        """Percentage of frames with bad posture."""
        if self.total_frames == 0:
            return 0.0
        return round(100 * self.bad_frames / self.total_frames, 1)

    def summary(self) -> dict:
        """Return a summary dictionary."""
        return {
            "session_duration":     self.session_duration,
            "total_frames":         self.total_frames,
            "good_frames":          self.good_frames,
            "bad_frames":           self.bad_frames,
            "good_posture_pct":     self.good_posture_percent,
            "bad_posture_pct":      self.bad_posture_percent,
            "alert_count":          self.alert_count,
            "screenshot_count":     self.screenshot_count,
        }


# ─────────────────────────────────────────────────────────────
# SECTION 7: POSTURE CORRECTION TIPS
# ─────────────────────────────────────────────────────────────

POSTURE_TIPS = {
    "neck": [
        "🔵 Keep your head directly above your shoulders.",
        "🔵 Position your monitor at eye level to reduce neck strain.",
        "🔵 Avoid looking down at your phone for extended periods.",
        "🔵 Take neck stretching breaks every 30 minutes.",
    ],
    "shoulder": [
        "🟠 Roll your shoulders back and down to open your chest.",
        "🟠 Avoid rounding your shoulders while typing.",
        "🟠 Keep both shoulders at the same height.",
        "🟠 Adjust your chair armrests to support your forearms.",
    ],
    "spine": [
        "🟢 Sit with your back touching the backrest of your chair.",
        "🟢 Maintain the natural S-curve of your spine.",
        "🟢 Avoid crossing your legs for extended periods.",
        "🟢 Use lumbar support if your chair lacks lower back support.",
    ],
    "general": [
        "⭐ Take a 5-minute standing break every hour.",
        "⭐ Ensure your feet are flat on the floor.",
        "⭐ Keep your keyboard at a height where elbows are at 90°.",
        "⭐ Blink frequently and look away from the screen every 20 min.",
        "⭐ Stay hydrated — dehydration contributes to muscle fatigue.",
    ],
}


def get_posture_tips(angles_dict):
    """
    Generate targeted posture correction tips based on current angles.

    Parameters:
        angles_dict (dict): Angle features from angle_calculation

    Returns:
        list[str]: 2–4 relevant tips
    """
    tips = []

    neck  = angles_dict.get("neck_avg", 0)
    spine = angles_dict.get("spine_angle", 0)
    slope = angles_dict.get("shoulder_slope", 0)

    if neck > 20:
        import random
        tips.append(random.choice(POSTURE_TIPS["neck"]))
    if slope > 10:
        import random
        tips.append(random.choice(POSTURE_TIPS["shoulder"]))
    if spine > 15:
        import random
        tips.append(random.choice(POSTURE_TIPS["spine"]))

    # Always add one general tip
    import random
    tips.append(random.choice(POSTURE_TIPS["general"]))

    return tips[:4]  # limit to 4 tips


# ─────────────────────────────────────────────────────────────
# SECTION 8: TIMESTAMP
# ─────────────────────────────────────────────────────────────

def get_timestamp() -> str:
    """Return formatted current timestamp string."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_session_id() -> str:
    """Generate a unique session ID based on current time."""
    return datetime.datetime.now().strftime("session_%Y%m%d_%H%M%S")


# ─────────────────────────────────────────────────────────────
# SECTION 9: POSTURE RULES (rule-based fallback)
# ─────────────────────────────────────────────────────────────

# Thresholds for rule-based posture detection (used when ML model is unavailable)
POSTURE_THRESHOLDS = {
    "neck_avg":        35.0,   # degrees — neck inclination
    "spine_angle":     22.0,   # degrees — spine inclination
    "shoulder_slope":  12.0,   # degrees — shoulder unevenness
    "head_tilt":       12.0,   # degrees — head sideways tilt
    "hip_alignment":   12.0,   # degrees — hip unevenness
}


def rule_based_posture_check(angles_dict) -> tuple:
    """
    Determine posture using rule-based thresholds (ML model fallback).

    Parameters:
        angles_dict (dict): Angle features

    Returns:
        tuple: (is_bad: bool, reasons: list[str], confidence: float)
    """
    reasons = []

    if angles_dict.get("neck_avg", 0) > POSTURE_THRESHOLDS["neck_avg"]:
        reasons.append("Forward head posture detected")

    if angles_dict.get("spine_angle", 0) > POSTURE_THRESHOLDS["spine_angle"]:
        reasons.append("Slouching / curved spine detected")

    if angles_dict.get("shoulder_slope", 0) > POSTURE_THRESHOLDS["shoulder_slope"]:
        reasons.append("Uneven shoulders detected")

    if angles_dict.get("head_tilt", 0) > POSTURE_THRESHOLDS["head_tilt"]:
        reasons.append("Head tilt detected")

    if angles_dict.get("hip_alignment", 0) > POSTURE_THRESHOLDS["hip_alignment"]:
        reasons.append("Uneven hip alignment detected")

    is_bad = len(reasons) > 0

    # Confidence estimate: based on how many thresholds are exceeded
    confidence = 0.5 + 0.1 * len(reasons)
    confidence = min(confidence, 0.95)

    return is_bad, reasons, confidence
