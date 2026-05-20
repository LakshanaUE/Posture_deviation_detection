"""
============================================================
posture_detection.py
AI-Based Real-Time Human Posture Detection and Correction System
============================================================
Core real-time detection engine using MediaPipe Tasks API.

Uses PoseLandmarker (33 landmarks) with the new tasks API
(mediapipe >= 0.10.30, Python 3.14 compatible).
============================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import os
import math
import threading
import logging

from feature_extraction import extract_features_realtime
from utils import (
    load_model_artifacts,
    predict_posture,
    label_to_text,
    is_bad_posture,
    rule_based_posture_check,
    log_posture_to_csv,
    save_screenshot,
    FPSCounter,
    SessionStats,
    get_posture_tips,
    init_history_csv,
    get_session_id,
)
from alerts import AlertManager

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# MEDIAPIPE TASKS API SETUP
# ─────────────────────────────────────────────────────────────
BaseOptions       = mp.tasks.BaseOptions
PoseLandmarker    = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOpts = mp.tasks.vision.PoseLandmarkerOptions
RunningMode       = mp.tasks.vision.RunningMode

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "pose_landmarker_lite.task")

# ─────────────────────────────────────────────────────────────
# VISUAL CONSTANTS  (matching reference images)
# ─────────────────────────────────────────────────────────────
COLOR_GOOD      = (0,  220,  80)   # bright green (BGR)
COLOR_BAD       = (0,   50, 220)   # red
COLOR_WARN      = (0,  200, 255)   # amber
COLOR_DOT_LEFT  = (200, 220, 0)    # cyan
COLOR_DOT_RIGHT = (0,  165, 255)   # orange

FONT   = cv2.FONT_HERSHEY_SIMPLEX
FONT_B = cv2.FONT_HERSHEY_DUPLEX

# Skeleton connections for drawing
POSTURE_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(24,26),(26,28),
    (7,11),(8,12),(7,8),
]

LEFT_IDS  = {7,11,13,15,17,19,21,23,25,27,29,31}
RIGHT_IDS = {8,12,14,16,18,20,22,24,26,28,30,32}


# ─────────────────────────────────────────────────────────────
# LANDMARK ADAPTER: Convert tasks API landmarks to legacy format
# ─────────────────────────────────────────────────────────────

class _LandmarkAdapter:
    """Wraps a tasks-API NormalizedLandmark to have .x .y .z .visibility."""
    __slots__ = ("x", "y", "z", "visibility")
    def __init__(self, lm):
        self.x = lm.x
        self.y = lm.y
        self.z = lm.z
        self.visibility = getattr(lm, "visibility", 1.0)


def adapt_landmarks(pose_result):
    """
    Convert PoseLandmarkerResult to a list of 33 landmark objects
    compatible with our feature extraction code.

    Returns None if no pose detected.
    """
    if not pose_result.pose_landmarks or len(pose_result.pose_landmarks) == 0:
        return None
    raw = pose_result.pose_landmarks[0]  # first person
    return [_LandmarkAdapter(lm) for lm in raw]


# ─────────────────────────────────────────────────────────────
# DRAWING FUNCTIONS
# ─────────────────────────────────────────────────────────────

def lm_px(landmarks, idx, w, h):
    return (int(landmarks[idx].x * w), int(landmarks[idx].y * h))

def lm_vis(landmarks, idx):
    return landmarks[idx].visibility


def draw_skeleton(frame, landmarks, color):
    h, w = frame.shape[:2]
    for (i, j) in POSTURE_CONNECTIONS:
        if lm_vis(landmarks,i)<0.3 or lm_vis(landmarks,j)<0.3:
            continue
        cv2.line(frame, lm_px(landmarks,i,w,h), lm_px(landmarks,j,w,h), color, 2, cv2.LINE_AA)
    for idx in range(33):
        if lm_vis(landmarks,idx)<0.3:
            continue
        px = lm_px(landmarks,idx,w,h)
        dot_c = COLOR_DOT_LEFT if idx in LEFT_IDS else COLOR_DOT_RIGHT
        cv2.circle(frame, px, 5, dot_c, -1, cv2.LINE_AA)
        cv2.circle(frame, px, 6, (255,255,255), 1, cv2.LINE_AA)


def draw_angle_lines(frame, landmarks, neck_angle, torso_angle):
    h, w = frame.shape[:2]
    if lm_vis(landmarks,11)<0.4 or lm_vis(landmarks,12)<0.4:
        return
    ls = lm_px(landmarks,11,w,h)
    rs = lm_px(landmarks,12,w,h)
    lh = lm_px(landmarks,23,w,h)
    rh = lm_px(landmarks,24,w,h)
    sh_mid = ((ls[0]+rs[0])//2, (ls[1]+rs[1])//2)
    hi_mid = ((lh[0]+rh[0])//2, (lh[1]+rh[1])//2)
    cv2.line(frame, sh_mid, hi_mid, COLOR_GOOD, 3, cv2.LINE_AA)
    vert_top = (sh_mid[0], max(0, sh_mid[1]-120))
    cv2.line(frame, sh_mid, vert_top, (200,200,200), 1, cv2.LINE_AA)
    if lm_vis(landmarks,7)>0.4:
        cv2.line(frame, lm_px(landmarks,7,w,h), ls, (200,220,0), 2, cv2.LINE_AA)
    cv2.line(frame, ls, rs, (200,200,200), 1, cv2.LINE_AA)


def draw_hud(frame, posture_text, confidence, neck_angle, torso_angle,
             fps, bad_duration, session_stats, angles_dict, is_bad, bad_posture_note=None):
    h, w = frame.shape[:2]

    # Border
    border_color = COLOR_BAD if is_bad else COLOR_GOOD
    cv2.rectangle(frame, (0,0), (w-1,h-1), border_color, 4)

    # Top-left angles (reference-image style)
    cv2.rectangle(frame, (5,5), (280,38), (0,0,0), -1)
    cv2.rectangle(frame, (5,5), (280,38), (50,50,50), 1)
    nc = COLOR_BAD if neck_angle >= 35.0 else (COLOR_WARN if neck_angle >= 25.0 else COLOR_GOOD)
    tc = COLOR_BAD if torso_angle >= 22.0 else (COLOR_WARN if torso_angle >= 15.0 else COLOR_GOOD)
    cv2.putText(frame, f"Neck : {int(neck_angle)}", (12,27), FONT, 0.6, nc, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Torso : {int(torso_angle)}", (145,27), FONT, 0.6, tc, 2, cv2.LINE_AA)

    # Top-right alignment label
    align = "Aligned" if not is_bad else "Misaligned"
    ac = COLOR_GOOD if not is_bad else COLOR_BAD
    slope = int(angles_dict.get("shoulder_slope",0))
    info = f"{slope}  {align}"
    tsz = cv2.getTextSize(info, FONT, 0.6, 2)[0]
    cv2.putText(frame, info, (w-tsz[0]-10,27), FONT, 0.6, ac, 2, cv2.LINE_AA)

    # Bottom status bar
    bar_h = 50
    ov = frame.copy()
    cv2.rectangle(ov, (0,h-bar_h), (w,h), (10,10,20), -1)
    cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)
    lc = COLOR_GOOD if not is_bad else COLOR_BAD
    
    display_text = posture_text
    if is_bad and bad_posture_note:
        display_text = f"BAD: {bad_posture_note}"
    
    cv2.putText(frame, display_text, (15,h-bar_h+32), FONT_B, 0.75, lc, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (w//2-60,h-bar_h+32), FONT, 0.6, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.0f}", (w-90,h-bar_h+32), FONT, 0.6, COLOR_WARN, 1, cv2.LINE_AA)

    # Timer
    if is_bad and bad_duration > 0:
        cv2.putText(frame, f"Bad Posture: {bad_duration:.1f}s", (15,h-bar_h-12), FONT, 0.55, COLOR_BAD, 1, cv2.LINE_AA)
    elif not is_bad:
        cv2.putText(frame, f"Good Posture Time : {session_stats.good_frames/max(fps,1):.1f}s", (15,h-bar_h-12), FONT, 0.55, COLOR_GOOD, 1, cv2.LINE_AA)

    # Warning
    if is_bad and bad_duration > 3:
        warn = "Please Sit Straight!"
        tsz = cv2.getTextSize(warn, FONT_B, 0.8, 2)[0]
        tx = (w-tsz[0])//2
        ov2 = frame.copy()
        cv2.rectangle(ov2, (tx-10,h-bar_h-50), (tx+tsz[0]+10,h-bar_h-15), (0,0,150), -1)
        cv2.addWeighted(ov2, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, warn, (tx,h-bar_h-22), FONT_B, 0.8, (255,255,255), 2, cv2.LINE_AA)

    # Live badge
    cv2.rectangle(frame, (5,42), (90,62), (30,30,50), -1)
    cv2.putText(frame, "Live Feed", (10,57), FONT, 0.45, (180,180,180), 1, cv2.LINE_AA)


def draw_no_person(frame):
    h, w = frame.shape[:2]
    cv2.putText(frame, "No person detected", (w//2-130,h//2-10), FONT_B, 0.85, (100,100,150), 2, cv2.LINE_AA)
    cv2.putText(frame, "Ensure your head, ears & shoulders are visible.", (w//2-220,h//2+30), FONT, 0.6, (80,80,100), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────
# POSTURE DETECTOR CLASS
# ─────────────────────────────────────────────────────────────

class PostureDetector:
    """Main posture detection engine using MediaPipe Tasks API."""

    def __init__(self,
                 camera_index=0,
                 sound_enabled=True,
                 voice_enabled=True,
                 use_ml_model=True,
                 bad_posture_threshold=2.0,
                 alert_cooldown=6.0,
                 log_interval=30,
                 screenshot_on_bad=False):
        self.camera_index      = camera_index
        self.use_ml_model      = use_ml_model
        self.log_interval      = log_interval
        self.screenshot_on_bad = screenshot_on_bad

        self.alert_manager = AlertManager(
            sound_enabled=sound_enabled,
            voice_enabled=voice_enabled,
            bad_posture_threshold=bad_posture_threshold,
            alert_cooldown=alert_cooldown,
        )
        self.fps_counter   = FPSCounter()
        self.session_stats = SessionStats()
        self.session_id    = get_session_id()

        self._model       = None
        self._scaler      = None
        self._model_info  = {}
        self._cap         = None
        self._running     = False
        self._frame_count = 0

        init_history_csv()

    def start(self):
        # cv2.CAP_DSHOW is required on Windows to instantly load the camera
        self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
        self._cap.set(cv2.CAP_PROP_FPS,           30)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_index}")
        if self.use_ml_model:
            self._model, self._scaler, self._model_info = load_model_artifacts()
            if self._model is None:
                logger.warning("[Detector] ML model not found - using rule-based.")
                self.use_ml_model = False
        self._running = True
        self.session_stats.reset()
        logger.info(f"[Detector] Camera started | ML: {self.use_ml_model}")

    def stop(self):
        self._running = False
        if self._cap:
            self._cap.release()
        logger.info("[Detector] Camera stopped.")

    def is_running(self):
        return self._running and self._cap is not None and self._cap.isOpened()

    def process_frame(self, pose_landmarker):
        """Process one frame. pose_landmarker is a PoseLandmarker instance."""
        ret, frame = self._cap.read()
        if not ret:
            return None, {}

        self._frame_count += 1
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        fps  = self.fps_counter.update()

        # MediaPipe Tasks API: convert to mp.Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = pose_landmarker.detect(mp_image)

        # Adapt landmarks to our format
        landmarks = adapt_landmarks(result)

        if landmarks is None:
            draw_no_person(frame)
            return frame, {"posture_text": "No Person", "fps": fps, "no_person": True}

        # Feature extraction (angles + ML vector)
        feat_vec, angles_dict = extract_features_realtime(landmarks, w, h)
        neck_avg    = angles_dict.get("neck_avg", 0)
        torso_angle = angles_dict.get("spine_angle", 0)

        # Classification and Sub-type Note determination
        bad_posture_note = None
        voice_msg = "Please sit straight."

        if self.use_ml_model and feat_vec is not None:
            label, confidence = predict_posture(self._model, self._scaler, feat_vec)
            label_names  = self._model_info.get("label_names", None)
            posture_text = label_to_text(label, label_names)
            bad_flag     = is_bad_posture(label, label_names)
        else:
            bad_flag, reasons, confidence = rule_based_posture_check(angles_dict)
            posture_text = "BAD POSTURE" if bad_flag else "GOOD POSTURE"

        if bad_flag:
            # Sub-types logic
            if torso_angle >= 22.0 and neck_avg >= 35.0:
                bad_posture_note = "Bending & Writing"
                voice_msg = "Please do not bend down while writing."
            elif torso_angle >= 22.0:
                bad_posture_note = "Slouching"
                voice_msg = "Please stop slouching."
            elif neck_avg >= 35.0:
                bad_posture_note = "Forward Head Posture"
                voice_msg = "Please pull your head back."
            elif angles_dict.get("shoulder_slope", 0) >= 12.0:
                bad_posture_note = "Uneven Shoulders"
                voice_msg = "Please level your shoulders."
            elif angles_dict.get("head_tilt", 0) >= 12.0:
                bad_posture_note = "Head Tilt"
                voice_msg = "Please straighten your head."
            else:
                bad_posture_note = "Misaligned Posture"
                voice_msg = "Please sit straight."

        # Alerts
        alert_fired  = self.alert_manager.update_bad_posture(bad_flag, message=voice_msg)
        bad_duration = self.alert_manager.get_bad_posture_duration()
        if alert_fired:
            self.session_stats.increment_alerts()
            if self.screenshot_on_bad:
                save_screenshot(frame, prefix="bad_posture")
                self.session_stats.increment_screenshots()

        self.session_stats.update(is_good=not bad_flag)
        skel_color = COLOR_BAD if bad_flag else COLOR_GOOD

        # Draw everything
        draw_skeleton(frame, landmarks, skel_color)
        draw_angle_lines(frame, landmarks, neck_avg, torso_angle)
        draw_hud(frame, posture_text, confidence, neck_avg, torso_angle,
                 fps, bad_duration, self.session_stats, angles_dict, bad_flag, bad_posture_note)

        # CSV logging (every N frames)
        if self._frame_count % self.log_interval == 0:
            log_posture_to_csv(posture_text, confidence, angles_dict, self.session_id)

        metrics = {
            "posture_text":   posture_text,
            "bad_posture_note": bad_posture_note,
            "is_bad":         bad_flag,
            "confidence":     round(confidence, 3),
            "neck_avg":       angles_dict["neck_avg"],
            "spine_angle":    angles_dict["spine_angle"],
            "shoulder_slope": angles_dict["shoulder_slope"],
            "head_tilt":      angles_dict["head_tilt"],
            "fps":            fps,
            "bad_duration":   bad_duration,
            "alert_fired":    alert_fired,
            "session":        self.session_stats.summary(),
            "tips":           get_posture_tips(angles_dict),
            "no_person":      False,
        }
        return frame, metrics

    def run_with_opencv_window(self):
        """Standalone mode with OpenCV window. Press q=quit, s=screenshot, t=toggle sound."""
        options = PoseLandmarkerOpts(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        landmarker = PoseLandmarker.create_from_options(options)

        self.start()
        print("[Detector] Running. q=quit, s=screenshot, t=toggle sound")

        while self.is_running():
            frame, metrics = self.process_frame(landmarker)
            if frame is None:
                break
            cv2.imshow("PoseNova", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                p = save_screenshot(frame)
                print(f"Screenshot: {p}")
            elif key == ord("t"):
                s = self.alert_manager.toggle_sound()
                print(f"Sound: {'ON' if s else 'OFF'}")

        landmarker.close()
        self.stop()
        cv2.destroyAllWindows()
        summary = self.session_stats.summary()
        print("\n" + "="*50 + "\nSESSION SUMMARY\n" + "="*50)
        for k, v in summary.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    detector = PostureDetector(
        camera_index=0, sound_enabled=True, voice_enabled=True,
        use_ml_model=True, bad_posture_threshold=2.0, alert_cooldown=6.0,
    )
    detector.run_with_opencv_window()
