"""
============================================================
feature_extraction.py
AI-Based Real-Time Human Posture Detection and Correction System
============================================================
Extracts the SAME 19 ML features used during training, from
live MediaPipe landmarks. This ensures the trained model receives
identical input format in real-time detection.

Feature vector (19 features):
  neck_avg, spine_angle, shoulder_slope, head_tilt,
  hip_alignment, ear_sh_left, ear_sh_right,
  lm7_x, lm7_y, lm8_x, lm8_y,
  lm11_x, lm11_y, lm12_x, lm12_y,
  lm23_x, lm23_y, lm24_x, lm24_y
============================================================
"""

import math
import numpy as np
from sklearn.preprocessing import StandardScaler

# Feature column names — MUST match preprocessing.py FEATURE_COLS exactly
FEATURE_NAMES = [
    "neck_avg", "spine_angle", "shoulder_slope", "head_tilt",
    "hip_alignment", "ear_sh_left", "ear_sh_right",
    "lm7_x",  "lm7_y",
    "lm8_x",  "lm8_y",
    "lm11_x", "lm11_y",
    "lm12_x", "lm12_y",
    "lm23_x", "lm23_y",
    "lm24_x", "lm24_y",
]


# ── Angle helpers (same math as preprocessing.py) ─────────────

def _neck_angle(ear_x, ear_y, sh_x, sh_y):
    dx = abs(ear_x - sh_x)
    dy = abs(ear_y - sh_y) + 1e-9
    return math.degrees(math.atan2(dx, dy))

def _torso_angle(sh_mid_x, sh_mid_y, hi_mid_x, hi_mid_y):
    dx = abs(sh_mid_x - hi_mid_x)
    dy = abs(sh_mid_y - hi_mid_y) + 1e-9
    return math.degrees(math.atan2(dx, dy))

def _slope(x1, y1, x2, y2):
    dx = abs(x2 - x1) + 1e-9
    dy = abs(y2 - y1)
    return math.degrees(math.atan2(dy, dx))


# ─────────────────────────────────────────────────────────────
# MAIN FUNCTION: Real-time feature extraction
# ─────────────────────────────────────────────────────────────

def extract_features_realtime(landmarks, frame_width=640, frame_height=480):
    """
    Extract the 19-feature vector from live MediaPipe landmarks.

    This function is called every frame during webcam capture.
    The output matches EXACTLY what the ML model was trained on.

    Parameters:
        landmarks    : pose_landmarks.landmark (list of 33 MediaPipe landmarks)
        frame_width  (int): frame width in pixels
        frame_height (int): frame height in pixels

    Returns:
        feature_vector (np.ndarray): shape (19,) — ML-ready
        angles_dict    (dict): human-readable angle values for HUD display
    """
    if landmarks is None:
        return None, {}

    # Helper: get normalized (0-1) coordinates
    def lm(idx):
        return (landmarks[idx].x, landmarks[idx].y)

    # Key landmark normalized coordinates
    l_ear  = lm(7)   # left ear
    r_ear  = lm(8)   # right ear
    l_sh   = lm(11)  # left shoulder
    r_sh   = lm(12)  # right shoulder
    l_hip  = lm(23)  # left hip
    r_hip  = lm(24)  # right hip

    # Shoulder and hip midpoints
    sh_mid_x = (l_sh[0] + r_sh[0]) / 2
    sh_mid_y = (l_sh[1] + r_sh[1]) / 2
    hi_mid_x = (l_hip[0] + r_hip[0]) / 2
    hi_mid_y = (l_hip[1] + r_hip[1]) / 2

    # ── Calculate angles ──────────────────────────────────────
    # Use standard 16:9 reference coordinates (1280x720) for angle calculation
    # to guarantee exact consistency with preprocessing and training.
    ref_w, ref_h = 1280.0, 720.0
    l_ear_px  = (l_ear[0]*ref_w,  l_ear[1]*ref_h)
    r_ear_px  = (r_ear[0]*ref_w,  r_ear[1]*ref_h)
    l_sh_px   = (l_sh[0]*ref_w,   l_sh[1]*ref_h)
    r_sh_px   = (r_sh[0]*ref_w,   r_sh[1]*ref_h)
    l_hip_px  = (l_hip[0]*ref_w,  l_hip[1]*ref_h)
    r_hip_px  = (r_hip[0]*ref_w,  r_hip[1]*ref_h)
    sh_mid_px = (sh_mid_x*ref_w,  sh_mid_y*ref_h)
    hi_mid_px = (hi_mid_x*ref_w,  hi_mid_y*ref_h)

    neck_l  = _neck_angle(l_ear_px[0], l_ear_px[1], l_sh_px[0], l_sh_px[1])
    neck_r  = _neck_angle(r_ear_px[0], r_ear_px[1], r_sh_px[0], r_sh_px[1])
    neck_avg = (neck_l + neck_r) / 2

    spine_angle     = _torso_angle(sh_mid_px[0], sh_mid_px[1], hi_mid_px[0], hi_mid_px[1])
    shoulder_slope  = _slope(l_sh_px[0], l_sh_px[1], r_sh_px[0], r_sh_px[1])
    head_tilt       = _slope(l_ear_px[0], l_ear_px[1], r_ear_px[0], r_ear_px[1])
    hip_alignment   = _slope(l_hip_px[0], l_hip_px[1], r_hip_px[0], r_hip_px[1])
    ear_sh_left     = abs(l_ear[1] - l_sh[1])
    ear_sh_right    = abs(r_ear[1] - r_sh[1])

    # ── Build the 19-element feature vector ───────────────────
    # Uses NORMALIZED coordinates for landmark positions (matching training)
    feature_vector = np.array([
        neck_avg,               # 0
        spine_angle,            # 1
        shoulder_slope,         # 2
        head_tilt,              # 3
        hip_alignment,          # 4
        ear_sh_left,            # 5  (normalized distance)
        ear_sh_right,           # 6  (normalized distance)
        l_ear[0],  l_ear[1],    # 7-8:   lm7  x,y
        r_ear[0],  r_ear[1],    # 9-10:  lm8  x,y
        l_sh[0],   l_sh[1],     # 11-12: lm11 x,y
        r_sh[0],   r_sh[1],     # 13-14: lm12 x,y
        l_hip[0],  l_hip[1],    # 15-16: lm23 x,y
        r_hip[0],  r_hip[1],    # 17-18: lm24 x,y
    ], dtype=np.float32)

    # Human-readable dict for HUD overlay
    angles_dict = {
        "neck_avg":       round(neck_avg, 1),
        "neck_left":      round(neck_l,   1),
        "neck_right":     round(neck_r,   1),
        "spine_angle":    round(spine_angle, 1),
        "shoulder_slope": round(shoulder_slope, 1),
        "head_tilt":      round(head_tilt, 1),
        "hip_alignment":  round(hip_alignment, 1),
    }

    return feature_vector, angles_dict


def get_feature_names():
    """Return the ordered list of 19 feature names."""
    return FEATURE_NAMES.copy()


def normalize_features(X, scaler=None, fit=False):
    """
    Normalize feature matrix using StandardScaler.

    Parameters:
        X      (np.ndarray): Feature matrix (n_samples, 19)
        scaler : Existing fitted scaler, or None
        fit    (bool): True = fit new scaler; False = transform only

    Returns:
        X_scaled (np.ndarray), scaler (StandardScaler)
    """
    if scaler is None or fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print(f"[FeatureExtraction] Scaler fitted. Mean: {scaler.mean_[:5]}...")
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, scaler
