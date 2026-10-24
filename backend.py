import os
import io
import uuid
import sqlite3
from datetime import datetime
from typing import cast

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from huggingface_hub import hf_hub_download
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles

# --- Configuration & Setup ---
UPLOAD_DIR = "uploaded_images"
REF_DIR = "reference_images"
DB_PATH = "monitoring_history.db"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REF_DIR, exist_ok=True)

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else 
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASSES = [
    "battery", "biological", "brown-glass", "cardboard",
    "clothes", "green-glass", "metal", "paper",
    "plastic", "shoes", "trash", "white-glass"
]

app = FastAPI(title="On-Device Garbage Drift Monitor API")
app.mount("/images", StaticFiles(directory=UPLOAD_DIR), name="images")

# --- Database Setup (SQLite) ---
def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                filename TEXT NOT NULL,
                image_path TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                entropy REAL NOT NULL,
                embedding_distance REAL NOT NULL,
                is_alert INTEGER NOT NULL,
                alert_reason TEXT NOT NULL
            )
        """)
        conn.commit()

init_db()

# --- Load ResNet-50 Model & Register Forward Hook ---
captured_embedding: np.ndarray = np.zeros(2048, dtype=np.float32)

def hook_fn(module: nn.Module, input: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
    global captured_embedding
    vec = output.view(output.size(0), -1)
    vec = F.normalize(vec, p=2, dim=1)
    captured_embedding = vec.detach().cpu().numpy()[0]

def load_garbage_model() -> models.ResNet:
    print("Downloading/loading model from Hugging Face Hub...")
    weight_path = hf_hub_download(
        repo_id="kendrickfff/my_resnet50_garbage_classificationv1.2",
        filename="resnet50_garbage_classificationv1.2.pth"
    )
    
    net = models.resnet50(weights=None)
    net.fc = nn.Linear(net.fc.in_features, len(CLASSES))
    
    ckpt = torch.load(weight_path, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, models.ResNet):
        net = ckpt
    elif isinstance(ckpt, dict):
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        net.load_state_dict(state_dict, strict=False)
        
    net.to(DEVICE)
    net.eval()
    net.avgpool.register_forward_hook(hook_fn)
    return net

model = load_garbage_model()

preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def image_to_tensor(img: Image.Image) -> torch.Tensor:
    tensor = cast(torch.Tensor, preprocess(img))
    return tensor.unsqueeze(0).to(DEVICE)

# --- Hybrid Baseline State (Nearest-Neighbor Reference Bank) ---
class HybridBaselineMonitor:
    def __init__(self, max_dynamic_samples: int = 10) -> None:
        self.static_bank: list[np.ndarray] = []
        self.dynamic_bank: list[np.ndarray] = []
        self.max_dynamic_samples = max_dynamic_samples
        self._init_static_baseline()

    def _init_static_baseline(self) -> None:
        """Stores each pristine reference image vector individually instead of averaging them."""
        for fname in os.listdir(REF_DIR):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                try:
                    img = Image.open(os.path.join(REF_DIR, fname)).convert("RGB")
                    tensor = image_to_tensor(img)
                    with torch.no_grad():
                        _ = model(tensor)
                    self.static_bank.append(captured_embedding.copy())
                except Exception:
                    pass

    def get_min_distance(self, emb: np.ndarray) -> float:
        """Returns the L2 distance to the single closest known waste image in the bank."""
        combined_bank = self.static_bank + self.dynamic_bank
        if not combined_bank:
            self.dynamic_bank.append(emb.copy())
            return 0.0
        distances = [float(np.linalg.norm(emb - ref)) for ref in combined_bank]
        return min(distances)

    def update_dynamic(
        self,
        emb: np.ndarray,
        entropy: float,
        dist: float,
        entropy_cutoff: float,
        dist_cutoff: float
    ) -> None:
        """Only adds a user upload to the dynamic bank if BOTH entropy AND distance are low."""
        if (
            len(self.dynamic_bank) < self.max_dynamic_samples
            and entropy < entropy_cutoff
            and dist < dist_cutoff
        ):
            self.dynamic_bank.append(emb.copy())

monitor = HybridBaselineMonitor(max_dynamic_samples=10)

# --- API Endpoints ---
@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    entropy_threshold: float = Form(1.10),
    distance_threshold: float = Form(0.95)
):
    try:
        raw_bytes = await file.read()
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    # 1. Save uploaded image locally
    ext = os.path.splitext(file.filename or "upload.jpg")[1] or ".jpg"
    unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}{ext}"
    save_path = os.path.join(UPLOAD_DIR, unique_filename)
    img.save(save_path)

    # 2. Forward pass (triggers hook_fn to populate captured_embedding)
    input_tensor = image_to_tensor(img)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    current_emb = captured_embedding.copy()
    top_idx = int(np.argmax(probs))
    pred_class = CLASSES[top_idx]
    confidence = float(probs[top_idx])

    # 3. Calculate Prediction Entropy (Shannon Entropy in nats; max for 12 classes is ln(12) ≈ 2.485)
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

    # 4. Calculate Input Distribution Shift (Closest match in Reference Bank)
    embedding_distance = monitor.get_min_distance(current_emb)
    monitor.update_dynamic(
        current_emb,
        entropy,
        embedding_distance,
        entropy_cutoff=entropy_threshold * 0.5,
        dist_cutoff=distance_threshold * 0.85
    )

    # 5. Evaluate Alert Thresholds
    reasons = []
    if entropy > entropy_threshold:
        reasons.append(f"High Prediction Uncertainty (Entropy {entropy:.2f} > {entropy_threshold:.2f})")
    if embedding_distance > distance_threshold:
        reasons.append(f"Input Distribution Shift (L2 Dist {embedding_distance:.2f} > {distance_threshold:.2f})")

    is_alert = 1 if reasons else 0
    alert_reason = " | ".join(reasons) if reasons else "Normal Operating Conditions"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 6. Log to SQLite Database
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions 
            (timestamp, filename, image_path, predicted_class, confidence, entropy, embedding_distance, is_alert, alert_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp, file.filename, unique_filename, pred_class,
            confidence, entropy, embedding_distance, is_alert, alert_reason
        ))
        conn.commit()
        record_id = cursor.lastrowid

    return {
        "id": record_id,
        "timestamp": timestamp,
        "filename": file.filename,
        "saved_image": unique_filename,
        "predicted_class": pred_class,
        "confidence": round(confidence, 4),
        "entropy": round(entropy, 4),
        "embedding_distance": round(embedding_distance, 4),
        "is_alert": bool(is_alert),
        "alert_reason": alert_reason,
        "calibration_samples": len(monitor.dynamic_bank),
        "has_static_baseline": len(monitor.static_bank) > 0
    }

@app.get("/history")
def get_history(limit: int = 50):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]

@app.post("/reset")
def reset_history():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM predictions")
        conn.commit()
    monitor.dynamic_bank.clear()
    return {"status": "Database and dynamic baseline cleared."}
