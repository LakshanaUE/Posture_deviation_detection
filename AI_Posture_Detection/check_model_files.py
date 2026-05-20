import os
import pickle

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "models", "posture_model.pkl")
scaler_path = os.path.join(base_dir, "models", "scaler.pkl")
info_path = os.path.join(base_dir, "models", "model_info.pkl")

print("=========================================")
print("          POSTURE MODEL DIAGNOSTICS      ")
print("=========================================")

all_ok = True

# Check posture_model.pkl
if os.path.exists(model_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"[OK] Model loaded: {type(model).__name__}")
    except Exception as e:
        print(f"[ERROR] Failed to load posture_model.pkl: {e}")
        all_ok = False
else:
    print("[ERROR] posture_model.pkl not found!")
    all_ok = False

# Check scaler.pkl
if os.path.exists(scaler_path):
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"[OK] Scaler loaded: {type(scaler).__name__}")
        print(f"     Expected features: {scaler.n_features_in_}")
    except Exception as e:
        print(f"[ERROR] Failed to load scaler.pkl: {e}")
        all_ok = False
else:
    print("[ERROR] scaler.pkl not found!")
    all_ok = False

# Check model_info.pkl
if os.path.exists(info_path):
    try:
        with open(info_path, 'rb') as f:
            info = pickle.load(f)
        print("[OK] Model metadata loaded:")
        print(f"     - Name: {info.get('model_name', 'N/A')}")
        print(f"     - Accuracy: {info.get('accuracy', 0)*100:.2f}%")
        print(f"     - F1-Score: {info.get('f1', 0):.4f}")
    except Exception as e:
        print(f"[ERROR] Failed to load model_info.pkl: {e}")
        all_ok = False
else:
    print("[ERROR] model_info.pkl not found!")
    all_ok = False

print("-----------------------------------------")
if all_ok:
    print("SUCCESS: All model files are valid and ready to use!")
else:
    print("WARNING: Some model files are missing or corrupted.")
print("=========================================")
