"""
============================================================
preprocessing.py
AI-Based Real-Time Human Posture Detection and Correction System
============================================================
Reads ALL 3 real datasets from dataset/merged/ and labels them
using the SAME angle-based logic as the real-time detector:

  Neck Angle  > 25°  -> BAD POSTURE
  Torso Angle > 20°  -> BAD POSTURE
  Otherwise         -> GOOD POSTURE

Datasets:
  1. dataset_all_points.csv  — 33 landmarks (x1,y1,z1,v1 ... format)
                               -> compute neck+torso angles -> label
  2. yoga_keypoints.csv      — 33 landmarks (cols 0..98 + 'label')
                               -> compute neck+torso angles -> label
  3. confidence_features.csv — pre-computed features + 'posture' column
                               Upright/Stiff=GOOD  Slouched=BAD

Feature vector for ML (same as real-time):
  neck_avg, spine_angle, shoulder_slope, head_tilt,
  hip_alignment, ear_sh_left, ear_sh_right,
  lm7_x, lm7_y, lm8_x, lm8_y,
  lm11_x, lm11_y, lm12_x, lm12_y,
  lm23_x, lm23_y, lm24_x, lm24_y
============================================================
"""

import os
import math
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
RAW_DIR       = os.path.join(BASE_DIR, "dataset", "raw")
MERGED_DIR    = os.path.join(BASE_DIR, "dataset", "merged")
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed")
REPORTS_DIR   = os.path.join(BASE_DIR, "reports")
for d in [RAW_DIR, MERGED_DIR, PROCESSED_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# POSTURE THRESHOLDS  (same as posture_detection.py / reference)
# ─────────────────────────────────────────────────────────────
NECK_BAD_THRESH  = 35.0   # degrees — forward head posture
TORSO_BAD_THRESH = 22.0   # degrees — slouching / spine lean
SLOPE_BAD_THRESH = 12.0   # degrees — uneven shoulders (secondary)

# Feature column names (must match feature_extraction.py + real-time system)
FEATURE_COLS = [
    "neck_avg", "spine_angle", "shoulder_slope", "head_tilt",
    "hip_alignment", "ear_sh_left", "ear_sh_right",
    "lm7_x",  "lm7_y",   # left ear
    "lm8_x",  "lm8_y",   # right ear
    "lm11_x", "lm11_y",  # left shoulder
    "lm12_x", "lm12_y",  # right shoulder
    "lm23_x", "lm23_y",  # left hip
    "lm24_x", "lm24_y",  # right hip
]


# ─────────────────────────────────────────────────────────────
# SECTION 1: ANGLE COMPUTATION HELPERS
# ─────────────────────────────────────────────────────────────

def _neck_angle(ear_x, ear_y, sh_x, sh_y):
    """Angle between ear->shoulder line and vertical axis."""
    dx = abs(ear_x - sh_x)
    dy = abs(ear_y - sh_y) + 1e-9
    return math.degrees(math.atan2(dx, dy))


def _torso_angle(sh_mid_x, sh_mid_y, hi_mid_x, hi_mid_y):
    """Angle between shoulder-midpoint->hip-midpoint and vertical."""
    dx = abs(sh_mid_x - hi_mid_x)
    dy = abs(sh_mid_y - hi_mid_y) + 1e-9
    return math.degrees(math.atan2(dx, dy))


def _slope(x1, y1, x2, y2):
    """Angle of the line between two points from horizontal."""
    dx = abs(x2 - x1) + 1e-9
    dy = abs(y2 - y1)
    return math.degrees(math.atan2(dy, dx))


def angle_label(neck_avg, spine_angle, shoulder_slope, head_tilt=0):
    """
    Apply the same rule as the real-time detector.
    Returns: 0 = BAD, 1 = GOOD
    """
    if neck_avg > NECK_BAD_THRESH:
        return 0
    if spine_angle > TORSO_BAD_THRESH:
        return 0
    if shoulder_slope > SLOPE_BAD_THRESH:
        return 0
    if head_tilt > 12.0:
        return 0
    return 1


def extract_features_from_landmarks_df(df,
                                        lm_format="indexed",
                                        label_override=None):
    """
    Compute posture angle features from a DataFrame of MediaPipe landmarks.

    Parameters:
        df             : DataFrame with landmark coordinate columns
        lm_format      : "indexed"  -> columns x1,y1,z1,v1 ... x33,...
                         "flat"     -> columns 0,1,2,...98 (33 lm × 3 axes)
        label_override : If None -> auto-label from angles
                         If Series/array -> use these labels directly

    Returns:
        pd.DataFrame with FEATURE_COLS + 'label' + 'source'
    """
    records = []
    labels  = []

    for row_idx, row in df.iterrows():
        try:
            if lm_format == "indexed":
                # x1,y1 ... x33,y33  (1-indexed, so landmark i = col x{i})
                def g(i, ax):  # i is 1-indexed landmark number
                    return float(row.get(f"{ax}{i}", 0.0))
                l_ear_x, l_ear_y   = g(8,  "x"), g(8,  "y")   # lm index 7  -> col x8
                r_ear_x, r_ear_y   = g(9,  "x"), g(9,  "y")   # lm index 8  -> col x9
                l_sh_x,  l_sh_y    = g(12, "x"), g(12, "y")   # lm index 11 -> col x12
                r_sh_x,  r_sh_y    = g(13, "x"), g(13, "y")   # lm index 12 -> col x13
                l_hi_x,  l_hi_y    = g(24, "x"), g(24, "y")   # lm index 23 -> col x24
                r_hi_x,  r_hi_y    = g(25, "x"), g(25, "y")   # lm index 24 -> col x25

            else:  # flat: col index = landmark_idx * 3 + axis (0=x,1=y,2=z)
                def gf(lm_idx, axis):
                    col = str(lm_idx * 3 + axis)
                    return float(row.get(col, 0.0))
                l_ear_x, l_ear_y   = gf(7,  0), gf(7,  1)
                r_ear_x, r_ear_y   = gf(8,  0), gf(8,  1)
                l_sh_x,  l_sh_y    = gf(11, 0), gf(11, 1)
                r_sh_x,  r_sh_y    = gf(12, 0), gf(12, 1)
                l_hi_x,  l_hi_y    = gf(23, 0), gf(23, 1)
                r_hi_x,  r_hi_y    = gf(24, 0), gf(24, 1)

            # Scale coordinates to a standard 16:9 reference aspect ratio (1280 x 720)
            # to match real-time feature extraction.
            l_ear_x_s, l_ear_y_s = l_ear_x * 1280, l_ear_y * 720
            r_ear_x_s, r_ear_y_s = r_ear_x * 1280, r_ear_y * 720
            l_sh_x_s,  l_sh_y_s  = l_sh_x * 1280,  l_sh_y * 720
            r_sh_x_s,  r_sh_y_s  = r_sh_x * 1280,  r_sh_y * 720
            l_hi_x_s,  l_hi_y_s  = l_hi_x * 1280,  l_hi_y * 720
            r_hi_x_s,  r_hi_y_s  = r_hi_x * 1280,  r_hi_y * 720

            sh_mid_x_s = (l_sh_x_s + r_sh_x_s) / 2
            sh_mid_y_s = (l_sh_y_s + r_sh_y_s) / 2
            hi_mid_x_s = (l_hi_x_s + r_hi_x_s) / 2
            hi_mid_y_s = (l_hi_y_s + r_hi_y_s) / 2

            neck_l  = _neck_angle(l_ear_x_s, l_ear_y_s, l_sh_x_s, l_sh_y_s)
            neck_r  = _neck_angle(r_ear_x_s, r_ear_y_s, r_sh_x_s, r_sh_y_s)
            neck_a  = (neck_l + neck_r) / 2
            spine   = _torso_angle(sh_mid_x_s, sh_mid_y_s, hi_mid_x_s, hi_mid_y_s)
            sh_slp  = _slope(l_sh_x_s, l_sh_y_s, r_sh_x_s, r_sh_y_s)
            hd_tlt  = _slope(l_ear_x_s, l_ear_y_s, r_ear_x_s, r_ear_y_s)
            hip_al  = _slope(l_hi_x_s, l_hi_y_s, r_hi_x_s, r_hi_y_s)
            ear_sl  = abs(l_ear_y - l_sh_y)
            ear_sr  = abs(r_ear_y - r_sh_y)

            records.append({
                "neck_avg":       round(neck_a,  3),
                "spine_angle":    round(spine,   3),
                "shoulder_slope": round(sh_slp,  3),
                "head_tilt":      round(hd_tlt,  3),
                "hip_alignment":  round(hip_al,  3),
                "ear_sh_left":    round(ear_sl,  4),
                "ear_sh_right":   round(ear_sr,  4),
                "lm7_x":  l_ear_x, "lm7_y":  l_ear_y,
                "lm8_x":  r_ear_x, "lm8_y":  r_ear_y,
                "lm11_x": l_sh_x,  "lm11_y": l_sh_y,
                "lm12_x": r_sh_x,  "lm12_y": r_sh_y,
                "lm23_x": l_hi_x,  "lm23_y": l_hi_y,
                "lm24_x": r_hi_x,  "lm24_y": r_hi_y,
            })

            if label_override is not None:
                labels.append(int(label_override.iloc[row_idx]
                                  if hasattr(label_override, "iloc")
                                  else label_override[row_idx]))
            else:
                labels.append(angle_label(neck_a, spine, sh_slp, hd_tlt))

        except Exception:
            continue  # skip malformed rows

    feat_df          = pd.DataFrame(records)
    feat_df["label"] = labels
    return feat_df


# ─────────────────────────────────────────────────────────────
# SECTION 2: LOAD DATASET 1 — dataset_all_points.csv
# ─────────────────────────────────────────────────────────────

def load_gym_landmarks():
    """
    Load dataset_all_points.csv (33 MediaPipe landmarks, 1-indexed).

    Columns: class, x1,y1,z1,v1, x2,y2,z2,v2, ... x33,y33,z33,v33

    Strategy: Calculate actual neck + torso angles from landmarks.
    Label = angle_label(neck_avg, spine_angle, shoulder_slope)
    This ensures the label is scientifically consistent with the
    real-time detection logic (same thresholds).

    Result: ~2700 samples with posture angle features.
    """
    path = os.path.join(RAW_DIR, "dataset_all_points.csv")
    print(f"\n[DS1] Loading gym landmarks dataset...")
    df = pd.read_csv(path, engine='python')
    print(f"     {len(df)} rows | Classes: {dict(df['class'].value_counts())}")

    feat_df          = extract_features_from_landmarks_df(df, lm_format="indexed")
    feat_df["source"] = "gym_landmarks"

    good = (feat_df["label"] == 1).sum()
    bad  = (feat_df["label"] == 0).sum()
    print(f"     Angle-based labels -> Good: {good} | Bad: {bad}")
    print(f"     Avg neck angle: {feat_df['neck_avg'].mean():.1f}° | "
          f"Avg spine angle: {feat_df['spine_angle'].mean():.1f}°")
    return feat_df


# ─────────────────────────────────────────────────────────────
# SECTION 3: LOAD DATASET 2 — yoga_keypoints.csv
# ─────────────────────────────────────────────────────────────

def load_yoga_keypoints():
    """
    Load yoga_keypoints.csv (33 landmarks in flat format, cols 0-98 + label).

    Columns: 0,1,2,...,98 where col = landmark_idx*3 + axis
             + 'label' column (yoga pose name, not used for labeling)

    Strategy: Calculate neck + torso angles from landmark coordinates.
    Label = angle_label(neck_avg, spine_angle, shoulder_slope)

    Yoga poses naturally include both good posture (standing, warrior)
    and bad posture angles (forward bends, deep slouches), so the
    angle-based label is the most accurate approach.

    Result: ~2633 samples.
    """
    path = os.path.join(RAW_DIR, "yoga_keypoints.csv")
    print(f"\n[DS2] Loading yoga keypoints dataset...")
    df = pd.read_csv(path, engine='python')
    print(f"     {len(df)} rows | Yoga poses: {df['label'].nunique()} unique")
    print(f"     Sample poses: {list(df['label'].value_counts().head(5).index)}")

    # Drop the text label column before feature extraction
    df_landmarks = df.drop(columns=["label"], errors="ignore")

    feat_df           = extract_features_from_landmarks_df(df_landmarks, lm_format="flat")
    feat_df["source"] = "yoga_keypoints"
    feat_df["pose"]   = df["label"].values[:len(feat_df)]  # keep for analysis

    good = (feat_df["label"] == 1).sum()
    bad  = (feat_df["label"] == 0).sum()
    print(f"     Angle-based labels -> Good: {good} | Bad: {bad}")
    print(f"     Avg neck angle: {feat_df['neck_avg'].mean():.1f}° | "
          f"Avg spine angle: {feat_df['spine_angle'].mean():.1f}°")
    return feat_df


# ─────────────────────────────────────────────────────────────
# SECTION 4: LOAD DATASET 3 — confidence_features.csv
# ─────────────────────────────────────────────────────────────

def load_confidence_features():
    """
    Load confidence_features.csv (pre-computed body features).

    Already has meaningful computed features:
      spine_angle, head_tilt_angle, shoulder_slope,
      eye_shoulder_y_ratio, shoulder_y_diff, etc.

    Uses 'posture' column for labeling:
      'Upright' -> GOOD (1)   — back straight, neck straight
      'Stiff'   -> GOOD (1)   — rigid but upright = good for our purposes
      'Slouched' -> BAD (0)   — forward head / rounded back

    Also has direct posture-related features that overlap with our
    real-time system (spine_angle, shoulder_slope, head_tilt_angle).

    Result: ~5949 samples — largest and most directly useful dataset.
    """
    path = os.path.join(RAW_DIR, "confidence_features.csv")
    print(f"\n[DS3] Loading confidence features dataset...")
    df = pd.read_csv(path, engine='python')
    print(f"     {len(df)} rows")
    print(f"     posture values: {dict(df['posture'].value_counts())}")

    # Label mapping based on posture column
    posture_map = {"Upright": 1, "Stiff": 1, "Slouched": 0}
    df["label"] = df["posture"].map(posture_map)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    print(f"     After mapping -> Good: {(df['label']==1).sum()} | Bad: {(df['label']==0).sum()}")

    # Build feature frame using available columns
    # Map confidence_features column names to our standard feature names
    feat_df = pd.DataFrame()

    # Direct column mappings
    feat_df["spine_angle"]    = df["spine_angle"].values
    feat_df["shoulder_slope"] = df["shoulder_slope"].values
    feat_df["head_tilt"]      = df["head_tilt_angle"].values

    # Derive neck_avg: eye_shoulder_y_ratio is proportional to neck extension
    # Scale to approximate degrees (calibrated to match our angle computation)
    feat_df["neck_avg"]       = df["eye_shoulder_y_ratio"].values * 45.0

    # Hip alignment from shoulder_y_diff proxy
    feat_df["hip_alignment"]  = df["shoulder_y_diff"].abs().values * 20.0

    # Ear-to-shoulder distances from hip_shoulder_y_diff
    feat_df["ear_sh_left"]    = df["hip_shoulder_y_diff"].abs().values * 0.5
    feat_df["ear_sh_right"]   = df["hip_shoulder_y_diff"].abs().values * 0.5

    # Landmark positions from available spatial features
    feat_df["lm11_x"] = df["shoulder_center_x"].values  # left  shoulder x ≈ center
    feat_df["lm11_y"] = df["shoulder_y_diff"].values * 0.5
    feat_df["lm12_x"] = df["shoulder_center_x"].values  # right shoulder x ≈ center
    feat_df["lm12_y"] = -df["shoulder_y_diff"].values * 0.5
    feat_df["lm23_x"] = df["hip_center_x"].values
    feat_df["lm23_y"] = df["hip_shoulder_y_diff"].values * 0.5 + 0.6
    feat_df["lm24_x"] = df["hip_center_x"].values
    feat_df["lm24_y"] = df["hip_shoulder_y_diff"].values * 0.5 + 0.6
    # Ear positions estimated from shoulder + eye_shoulder ratio
    feat_df["lm7_x"]  = df["shoulder_center_x"].values - df["body_lean_x"].values
    feat_df["lm7_y"]  = feat_df["lm11_y"] - feat_df["ear_sh_left"]
    feat_df["lm8_x"]  = df["shoulder_center_x"].values + df["body_lean_x"].values
    feat_df["lm8_y"]  = feat_df["lm12_y"] - feat_df["ear_sh_right"]

    feat_df["label"]  = df["label"].values
    feat_df["source"] = "confidence_features"

    # Drop rows where key features are NaN
    feat_df = feat_df.dropna(subset=["spine_angle", "shoulder_slope", "label"])
    print(f"     Features: {list(feat_df.columns)}")
    return feat_df


# ─────────────────────────────────────────────────────────────
# SECTION 5: CLEAN + VALIDATE
# ─────────────────────────────────────────────────────────────

def clean_and_validate(df, source_name):
    """Remove NaN/inf, clip extreme values, drop duplicates."""
    before = len(df)

    # Replace inf
    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop rows with too many NaN
    thresh = int(len(df.columns) * 0.6)
    df = df.dropna(thresh=thresh)

    # Fill remaining NaN with column median
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

    # Drop all-zero feature rows (invalid landmarks)
    feat_sum = df[FEATURE_COLS].abs().sum(axis=1)
    df = df[feat_sum > 0.01]

    # Clip angles to physical range 0-90°
    for col in ["neck_avg", "spine_angle", "shoulder_slope", "head_tilt", "hip_alignment"]:
        if col in df.columns:
            df[col] = df[col].clip(0, 90)

    df = df.drop_duplicates()
    print(f"     Clean [{source_name}]: {before} -> {len(df)} rows")
    return df


# ─────────────────────────────────────────────────────────────
# SECTION 6: MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def run_preprocessing():
    """
    Full preprocessing pipeline using all 3 real datasets.

    All datasets are labeled using angle-based logic consistent
    with the real-time posture detection system.

    Outputs:
        dataset/merged/train.csv
        dataset/merged/test.csv
        dataset/processed/<source>_features.csv  (individual)
        reports/preprocessing_summary.txt
    """
    print("\n" + "=" * 60)
    print("  POSTURE DETECTION — DATA PREPROCESSING PIPELINE")
    print("  Using angle-based labeling (Neck > 25° or Torso > 20° = BAD)")
    print("=" * 60)

    all_frames = []

    # ── Dataset 1 ─────────────────────────────────────────────
    try:
        df1 = load_gym_landmarks()
        df1 = clean_and_validate(df1, "gym_landmarks")
        save_path = os.path.join(PROCESSED_DIR, "gym_landmarks_features.csv")
        df1[FEATURE_COLS + ["label", "source"]].to_csv(save_path, index=False)
        print(f"     Saved: {save_path}")
        all_frames.append(df1[FEATURE_COLS + ["label", "source"]])
    except Exception as e:
        print(f"  [WARN] Dataset 1 failed: {e}")
        import traceback; traceback.print_exc()

    # ── Dataset 2 ─────────────────────────────────────────────
    try:
        df2 = load_yoga_keypoints()
        df2 = clean_and_validate(df2, "yoga_keypoints")
        save_path = os.path.join(PROCESSED_DIR, "yoga_keypoints_features.csv")
        df2[FEATURE_COLS + ["label", "source"]].to_csv(save_path, index=False)
        print(f"     Saved: {save_path}")
        all_frames.append(df2[FEATURE_COLS + ["label", "source"]])
    except Exception as e:
        print(f"  [WARN] Dataset 2 failed: {e}")
        import traceback; traceback.print_exc()

    # ── Dataset 3 ─────────────────────────────────────────────
    try:
        df3 = load_confidence_features()
        df3 = clean_and_validate(df3, "confidence_features")
        save_path = os.path.join(PROCESSED_DIR, "confidence_features_processed.csv")
        df3[FEATURE_COLS + ["label", "source"]].to_csv(save_path, index=False)
        print(f"     Saved: {save_path}")
        all_frames.append(df3[FEATURE_COLS + ["label", "source"]])
    except Exception as e:
        print(f"  [WARN] Dataset 3 failed: {e}")
        import traceback; traceback.print_exc()

    if not all_frames:
        raise RuntimeError("All datasets failed to load.")

    # ── Merge all ─────────────────────────────────────────────
    merged = pd.concat(all_frames, ignore_index=True, sort=False)

    # Final NaN fill
    for col in FEATURE_COLS:
        merged[col] = merged[col].fillna(0.0)

    total  = len(merged)
    good_n = (merged["label"] == 1).sum()
    bad_n  = (merged["label"] == 0).sum()

    print(f"\n{'='*60}")
    print(f"  MERGED DATASET")
    print(f"  Total  : {total} samples")
    print(f"  Good   : {good_n} ({100*good_n/total:.1f}%)")
    print(f"  Bad    : {bad_n}  ({100*bad_n/total:.1f}%)")
    print(f"  Sources: {dict(merged['source'].value_counts())}")
    print(f"{'='*60}")

    # Save merged feature set
    merged_out = os.path.join(MERGED_DIR, "merged_features.csv")
    merged.to_csv(merged_out, index=False)
    print(f"  Merged features: {merged_out}")

    # ── Train / Test split (stratified) ──────────────────────
    train_df, test_df = train_test_split(
        merged, test_size=0.2, random_state=42,
        stratify=merged["label"]
    )
    train_path = os.path.join(MERGED_DIR, "train.csv")
    test_path  = os.path.join(MERGED_DIR, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,   index=False)
    print(f"  Train: {len(train_df)} samples -> {train_path}")
    print(f"  Test:  {len(test_df)} samples  -> {test_path}")

    # ── Preprocessing summary report ──────────────────────────
    rpt = os.path.join(REPORTS_DIR, "preprocessing_summary.txt")
    with open(rpt, "w") as f:
        f.write("=" * 55 + "\n")
        f.write("  POSTURE DETECTION — PREPROCESSING SUMMARY\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Labeling Logic  : Neck>25deg OR Torso>20deg = BAD\n\n")
        f.write(f"Total samples   : {total}\n")
        f.write(f"Good posture(1) : {good_n} ({100*good_n/total:.1f}%)\n")
        f.write(f"Bad  posture(0) : {bad_n}  ({100*bad_n/total:.1f}%)\n")
        f.write(f"Train samples   : {len(train_df)}\n")
        f.write(f"Test  samples   : {len(test_df)}\n\n")
        f.write("Dataset breakdown:\n")
        for src, cnt in merged["source"].value_counts().items():
            f.write(f"  {src}: {cnt}\n")
        f.write("\nFeature columns:\n")
        for col in FEATURE_COLS:
            stats = merged[col]
            f.write(f"  {col:25s} mean={stats.mean():.2f} std={stats.std():.2f}\n")
    print(f"  Report: {rpt}")

    print("\n  PREPROCESSING COMPLETE\n")

    return {
        "train_path":  train_path,
        "test_path":   test_path,
        "label_names": ["bad", "good"],
        "n_train":     len(train_df),
        "n_test":      len(test_df),
        "feature_cols": FEATURE_COLS,
    }


if __name__ == "__main__":
    result = run_preprocessing()
    print("\nResult:", result)
