"""
============================================================
angle_calculation.py
AI-Based Real-Time Human Posture Detection and Correction System
============================================================
This module handles all angle and geometric calculations for posture analysis.
Functions include:
- Calculating angles between three joints
- Neck inclination calculation
- Shoulder slope calculation
- Spine angle calculation
- Head tilt detection
- Euclidean distances
============================================================
"""

import math
import numpy as np


# ─────────────────────────────────────────────────────────────
# SECTION 1: BASIC GEOMETRY UTILITIES
# ─────────────────────────────────────────────────────────────

def calculate_angle(a, b, c):
    """
    Calculate the angle (in degrees) at joint B formed by points A-B-C.

    Parameters:
        a (tuple): (x, y) coordinates of point A (e.g., shoulder)
        b (tuple): (x, y) coordinates of point B — the vertex (e.g., elbow)
        c (tuple): (x, y) coordinates of point C (e.g., wrist)

    Returns:
        float: Angle in degrees (0–180)
    """
    a = np.array(a[:2])  # use only x, y
    b = np.array(b[:2])
    c = np.array(c[:2])

    # Vectors from B to A and B to C
    ba = a - b
    bc = c - b

    # Dot product and magnitudes
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # prevent domain error
    angle = np.degrees(np.arccos(cosine_angle))
    return round(angle, 2)


def euclidean_distance(p1, p2):
    """
    Compute Euclidean distance between two 2D points.

    Parameters:
        p1 (tuple): (x, y) of point 1
        p2 (tuple): (x, y) of point 2

    Returns:
        float: Euclidean distance
    """
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def midpoint(p1, p2):
    """
    Return the midpoint between two 2D points.

    Parameters:
        p1 (tuple): (x, y)
        p2 (tuple): (x, y)

    Returns:
        tuple: (mx, my) midpoint coordinates
    """
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


# ─────────────────────────────────────────────────────────────
# SECTION 2: POSTURE-SPECIFIC ANGLE CALCULATIONS
# ─────────────────────────────────────────────────────────────

def calculate_neck_inclination(ear, shoulder):
    """
    Calculate the inclination angle of the neck relative to vertical.

    The neck inclination is the angle formed between the vertical axis
    and the line from shoulder to ear. A large angle indicates
    forward head posture (bad posture).

    Parameters:
        ear      (tuple): (x, y) ear landmark (left or right)
        shoulder (tuple): (x, y) shoulder landmark (same side)

    Returns:
        float: Neck inclination angle in degrees
    """
    dx = abs(ear[0] - shoulder[0])
    dy = abs(ear[1] - shoulder[1])
    angle = math.degrees(math.atan2(dx, dy + 1e-6))
    return round(angle, 2)


def calculate_torso_inclination(shoulder, hip):
    """
    Calculate the inclination angle of the torso/spine relative to vertical.

    A high torso inclination indicates slouching or leaning forward (bad posture).

    Parameters:
        shoulder (tuple): (x, y) shoulder midpoint
        hip      (tuple): (x, y) hip midpoint

    Returns:
        float: Torso/spine inclination angle in degrees
    """
    dx = abs(shoulder[0] - hip[0])
    dy = abs(shoulder[1] - hip[1])
    angle = math.degrees(math.atan2(dx, dy + 1e-6))
    return round(angle, 2)


def calculate_shoulder_slope(left_shoulder, right_shoulder):
    """
    Calculate the slope angle of the shoulder line.

    If shoulders are uneven, the slope will be non-zero. Used to detect
    uneven shoulders (a sign of bad posture or muscle imbalance).

    Parameters:
        left_shoulder  (tuple): (x, y) left shoulder
        right_shoulder (tuple): (x, y) right shoulder

    Returns:
        float: Shoulder slope angle in degrees (0 = perfectly level)
    """
    dx = right_shoulder[0] - left_shoulder[0]
    dy = right_shoulder[1] - left_shoulder[1]
    angle = math.degrees(math.atan2(abs(dy), abs(dx) + 1e-6))
    return round(angle, 2)


def calculate_head_tilt(left_ear, right_ear):
    """
    Calculate how much the head is tilted sideways.

    Compares the vertical difference between left and right ear positions.

    Parameters:
        left_ear  (tuple): (x, y) left ear landmark
        right_ear (tuple): (x, y) right ear landmark

    Returns:
        float: Head tilt angle in degrees (0 = no tilt)
    """
    dx = abs(right_ear[0] - left_ear[0])
    dy = right_ear[1] - left_ear[1]  # signed for direction
    angle = math.degrees(math.atan2(abs(dy), dx + 1e-6))
    return round(angle, 2)


def calculate_hip_alignment(left_hip, right_hip):
    """
    Calculate the alignment angle of the hips.

    Uneven hips can indicate a pelvic tilt or unequal weight distribution.

    Parameters:
        left_hip  (tuple): (x, y) left hip landmark
        right_hip (tuple): (x, y) right hip landmark

    Returns:
        float: Hip alignment angle in degrees (0 = perfectly level)
    """
    dx = abs(right_hip[0] - left_hip[0])
    dy = abs(right_hip[1] - left_hip[1])
    angle = math.degrees(math.atan2(dy, dx + 1e-6))
    return round(angle, 2)


def calculate_ear_shoulder_distance_ratio(ear, shoulder, image_height):
    """
    Calculate the normalized vertical distance from ear to shoulder.

    A small ratio means the ear is close to the shoulder (forward head posture).

    Parameters:
        ear           (tuple): (x, y) ear landmark
        shoulder      (tuple): (x, y) shoulder landmark
        image_height  (int): height of the image frame in pixels

    Returns:
        float: Normalized ear-to-shoulder vertical distance (0–1)
    """
    vertical_dist = abs(ear[1] - shoulder[1])
    return round(vertical_dist / (image_height + 1e-6), 4)


def calculate_spine_angle(left_shoulder, right_shoulder, left_hip, right_hip):
    """
    Calculate the overall spine angle using shoulder and hip midpoints.

    Parameters:
        left_shoulder  (tuple): (x, y) left shoulder
        right_shoulder (tuple): (x, y) right shoulder
        left_hip       (tuple): (x, y) left hip
        right_hip      (tuple): (x, y) right hip

    Returns:
        float: Spine inclination angle in degrees
    """
    shoulder_mid = midpoint(left_shoulder, right_shoulder)
    hip_mid = midpoint(left_hip, right_hip)
    return calculate_torso_inclination(shoulder_mid, hip_mid)


def calculate_upper_body_balance(left_shoulder, right_shoulder,
                                  left_hip, right_hip):
    """
    Calculate a symmetry score for the upper body.

    Compares horizontal displacement of left vs right side.

    Parameters:
        left_shoulder  (tuple): (x, y)
        right_shoulder (tuple): (x, y)
        left_hip       (tuple): (x, y)
        right_hip      (tuple): (x, y)

    Returns:
        float: Asymmetry score (0 = perfectly symmetric)
    """
    left_offset  = abs(left_shoulder[0]  - left_hip[0])
    right_offset = abs(right_shoulder[0] - right_hip[0])
    return round(abs(left_offset - right_offset), 4)


def calculate_shoulder_symmetry(left_shoulder, right_shoulder, frame_width):
    """
    Measure how symmetric the shoulders are relative to the frame center.

    Parameters:
        left_shoulder (tuple): (x, y)
        right_shoulder (tuple): (x, y)
        frame_width (int): width of the video frame

    Returns:
        float: Symmetry deviation (0 = perfect symmetry)
    """
    center_x = frame_width / 2
    left_dist  = abs(left_shoulder[0]  - center_x)
    right_dist = abs(right_shoulder[0] - center_x)
    return round(abs(left_dist - right_dist), 4)


# ─────────────────────────────────────────────────────────────
# SECTION 3: FEATURE VECTOR BUILDER
# ─────────────────────────────────────────────────────────────

def build_angle_feature_vector(landmarks, frame_width, frame_height):
    """
    Build a compact feature vector of posture angles from MediaPipe landmarks.

    MediaPipe Pose landmark indices:
        0  = nose
        1  = left eye inner   2  = left eye   3  = left eye outer
        4  = right eye inner  5  = right eye  6  = right eye outer
        7  = left ear         8  = right ear
        9  = mouth left       10 = mouth right
        11 = left shoulder    12 = right shoulder
        13 = left elbow       14 = right elbow
        15 = left wrist       16 = right wrist
        23 = left hip         24 = right hip
        25 = left knee        26 = right knee

    Parameters:
        landmarks    : MediaPipe pose landmarks object (.landmark list)
        frame_width  (int): width  of the video frame
        frame_height (int): height of the video frame

    Returns:
        dict: Dictionary of calculated posture angles and metrics
    """

    def lm(idx):
        """Helper: get (x*W, y*H) pixel coordinates for landmark index."""
        l = landmarks[idx]
        return (l.x * frame_width, l.y * frame_height)

    # Extract key landmark coordinates
    left_ear       = lm(7)
    right_ear      = lm(8)
    left_shoulder  = lm(11)
    right_shoulder = lm(12)
    left_elbow     = lm(13)
    right_elbow    = lm(14)
    left_hip       = lm(23)
    right_hip      = lm(24)
    left_knee      = lm(25)
    right_knee     = lm(26)

    # ── Neck angles (both sides) ──────────────────────────────
    neck_left  = calculate_neck_inclination(left_ear,  left_shoulder)
    neck_right = calculate_neck_inclination(right_ear, right_shoulder)
    neck_avg   = round((neck_left + neck_right) / 2, 2)

    # ── Torso / spine angle ───────────────────────────────────
    spine_angle = calculate_spine_angle(
        left_shoulder, right_shoulder, left_hip, right_hip
    )

    # ── Shoulder features ─────────────────────────────────────
    shoulder_slope = calculate_shoulder_slope(left_shoulder, right_shoulder)
    shoulder_sym   = calculate_shoulder_symmetry(
        left_shoulder, right_shoulder, frame_width
    )

    # ── Head tilt ─────────────────────────────────────────────
    head_tilt = calculate_head_tilt(left_ear, right_ear)

    # ── Hip alignment ─────────────────────────────────────────
    hip_alignment = calculate_hip_alignment(left_hip, right_hip)

    # ── Ear-shoulder distance ratio ───────────────────────────
    ear_shoulder_left  = calculate_ear_shoulder_distance_ratio(
        left_ear, left_shoulder, frame_height
    )
    ear_shoulder_right = calculate_ear_shoulder_distance_ratio(
        right_ear, right_shoulder, frame_height
    )
    ear_shoulder_avg   = round((ear_shoulder_left + ear_shoulder_right) / 2, 4)

    # ── Upper body balance ────────────────────────────────────
    upper_body_balance = calculate_upper_body_balance(
        left_shoulder, right_shoulder, left_hip, right_hip
    )

    # ── Elbow angles ─────────────────────────────────────────
    left_elbow_angle  = calculate_angle(left_shoulder,  left_elbow,  left_hip)
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_hip)

    # ── Knee angles ───────────────────────────────────────────
    left_knee_angle  = calculate_angle(left_hip,  left_knee,  lm(27))  # ankle
    right_knee_angle = calculate_angle(right_hip, right_knee, lm(28))

    return {
        "neck_left":           neck_left,
        "neck_right":          neck_right,
        "neck_avg":            neck_avg,
        "spine_angle":         spine_angle,
        "shoulder_slope":      shoulder_slope,
        "shoulder_symmetry":   shoulder_sym,
        "head_tilt":           head_tilt,
        "hip_alignment":       hip_alignment,
        "ear_shoulder_left":   ear_shoulder_left,
        "ear_shoulder_right":  ear_shoulder_right,
        "ear_shoulder_avg":    ear_shoulder_avg,
        "upper_body_balance":  upper_body_balance,
        "left_elbow_angle":    left_elbow_angle,
        "right_elbow_angle":   right_elbow_angle,
        "left_knee_angle":     left_knee_angle,
        "right_knee_angle":    right_knee_angle,
    }
