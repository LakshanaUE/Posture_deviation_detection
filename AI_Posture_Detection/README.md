# 🧠 AI-Based Real-Time Human Posture Detection and Correction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-00BCD4?logo=google&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-5C3EE8?logo=opencv&logoColor=white)

**A professional AI healthcare application for real-time ergonomic posture monitoring**

</div>

---

## 📌 Project Overview

This system uses **MediaPipe Pose** to detect 33 body landmarks in real time, calculates posture angles (neck, spine, shoulders, head), and classifies posture as **GOOD** or **BAD** using trained Machine Learning models. When bad posture is detected, the system triggers audio and voice alerts.

---

## ✨ Features

| Feature | Description |
|---|---|
| Real-time webcam detection | 30 FPS pose tracking |
| 33-landmark skeleton overlay | Cyan/orange dots, colored skeleton lines |
| On-screen angle display | Neck & Torso angles shown live |
| ML classification | Random Forest / SVM / KNN / Logistic Regression |
| Smart alert system | Beep + voice (pyttsx3) with cooldown timer |
| CSV posture logging | Full history with timestamps |
| Analytics dashboard | Pie chart, trend graph, daily stats |
| PDF report generation | Downloadable session report |
| Streamlit GUI | Dark-themed professional dashboard |
| Screenshot capture | Manual + auto on bad posture |

---

## 🏗️ Project Structure

```
AI_Posture_Detection/
│
├── dataset/
│   ├── raw/                ← Place downloaded Kaggle CSVs here
│   ├── processed/
│   └── merged/             ← train.csv, test.csv (auto-generated)
│
├── models/
│   ├── posture_model.pkl   ← Best trained ML model
│   ├── scaler.pkl
│   └── model_info.pkl
│
├── sounds/
│   ├── alert.mp3           ← Custom alert sound (optional)
│   └── beep.wav            ← Auto-generated fallback
│
├── screenshots/            ← Saved screenshots
├── logs/                   ← System logs
├── reports/                ← PDF reports, training report
├── graphs/                 ← Generated visualization PNGs
│
├── app.py                  ← Streamlit dashboard (main GUI)
├── posture_detection.py    ← Core detection engine
├── train_model.py          ← ML training pipeline
├── preprocessing.py        ← Data preprocessing
├── feature_extraction.py   ← Feature engineering
├── angle_calculation.py    ← Geometric angle calculations
├── analytics.py            ← Graphs and PDF generation
├── alerts.py               ← Sound + voice alert system
├── utils.py                ← Shared utilities
├── requirements.txt
├── posture_history.csv     ← Detection log
└── README.md
```

---

## 🔬 MediaPipe Pose Landmarks

The system uses all 33 MediaPipe landmarks. Key landmarks for posture:

| Index | Landmark | Role |
|---|---|---|
| 7 | left_ear | Neck angle calculation |
| 8 | right_ear | Neck angle calculation |
| 11 | left_shoulder | Spine & shoulder analysis |
| 12 | right_shoulder | Spine & shoulder analysis |
| 23 | left_hip | Torso angle |
| 24 | right_hip | Torso angle |

---

## 🤖 Machine Learning

Four models are trained and compared automatically:

| Model | Description |
|---|---|
| Random Forest | 200 trees, balanced class weights |
| Logistic Regression | L2 regularization, max_iter=1000 |
| SVM | RBF kernel, probability output |
| KNN | K=7, distance-weighted |

The best model (highest accuracy) is auto-saved as `posture_model.pkl`.

### Posture Features Extracted

- Neck inclination angle (left + right + average)
- Spine / torso angle
- Shoulder slope angle
- Head tilt angle
- Hip alignment angle
- Ear-to-shoulder distance ratio
- Upper body balance score
- Raw normalized landmark coordinates

---

## 🚦 Posture Detection Logic

```
IF neck_angle > 25°     → Forward head posture (BAD)
IF spine_angle > 20°    → Slouching / curved spine (BAD)
IF shoulder_slope > 12° → Uneven shoulders (BAD)
IF head_tilt > 10°      → Head tilt (BAD)
OTHERWISE               → GOOD POSTURE
```

**Skeleton Colors:**
- 🟢 GREEN = Good posture
- 🔴 RED = Bad posture

---

## 🔔 Alert System

1. **Threshold Timer** — Bad posture must persist for N seconds (default: 3s)
2. **Cooldown** — Min 6 seconds between repeated alerts (prevents spam)
3. **Beep Sound** — winsound / pygame mixer
4. **Voice Alert** — pyttsx3 TTS reads posture correction instruction
5. **Visual Warning** — Red border + flashing "Please Sit Straight!" on frame

---

## 📦 Installation

### 1. Clone / Download
```bash
cd "d:\MINI PROJECT II\AI_Posture_Detection"
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Option A — Streamlit Dashboard (Recommended)
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### Option B — OpenCV Standalone Window
```bash
python posture_detection.py
```

### Option C — Train ML Model First
```bash
# Step 1: Preprocess data
python preprocessing.py

# Step 2: Train models
python train_model.py

# Step 3: Run app
streamlit run app.py
```

---

## 📊 Datasets

Place downloaded CSVs in `dataset/raw/`. The preprocessing script auto-detects and merges them.

| Dataset | Source | Purpose |
|---|---|---|
| Gym Exercise MediaPipe Landmarks | Kaggle: dp5995 | Primary training |
| Yoga Pose Keypoints | Kaggle: suhaniajaythakur | Posture alignment |
| Human Pose Estimation | Kaggle: trainingdatapro | Skeleton detection |
| Exercise Recognition Time Series | Kaggle: muhannadtuameh | Movement analysis |
| Confidence Detection | Kaggle: muhammadkhubaibahmad | Slouch detection |

> ⚡ **No datasets needed!** The system auto-generates a synthetic training dataset if no CSVs are found.

---

## 📸 Output Examples

```
✅ GOOD POSTURE    (green skeleton, no alert)
⚠️ BAD POSTURE     (red skeleton, beep + voice: "Please Sit Straight!")
```

On-screen display (like reference implementation):
```
Neck : 21   Torso : 4        30  Aligned
Good Posture Time : 12.4s
```

---

## 🔮 Future Scope

- Mobile app integration (Flutter + TensorFlow Lite)
- Cloud dashboard (Firebase / AWS)
- Multi-person simultaneous tracking
- Wearable sensor fusion
- 3D pose estimation
- Employer ergonomic monitoring dashboard
- Email/SMS alert integration

---

## 🛠️ Tech Stack

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.11+ | Core language |
| OpenCV | 4.8+ | Video capture & drawing |
| MediaPipe | 0.10+ | Pose landmark detection |
| NumPy | 1.24+ | Numerical computation |
| Pandas | 2.0+ | Data manipulation |
| Scikit-learn | 1.3+ | ML models |
| Matplotlib / Seaborn | Latest | Visualization |
| Streamlit | 1.28+ | GUI dashboard |
| pyttsx3 | 2.90 | Voice alerts |
| pygame | 2.5+ | Audio playback |
| fpdf2 | 2.7+ | PDF generation |

---

## 👨‍💻 Author

**Final Year Engineering Project**  
AI + Machine Learning + Computer Vision  
*AI-Based Real-Time Human Posture Detection and Correction System using MediaPipe*
