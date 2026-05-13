"""
====================================================
  SCRIPT 4B: DEMO MODE (No Camera Required)
  Test model dengan pre-generated test images
====================================================
Jalankan:
  python 4_demo_mode.py

Fitur:
  - Buat dummy test images
  - Run inference pada setiap image
  - Display hasil dengan visualization
  - Simulasi real-time detection
====================================================
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import timm
import numpy as np
import cv2
from datetime import datetime
import random

# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────
MODEL_PATH    = Path("model/best_model.pth")
MAPPING_PATH  = Path("dataset/label_mapping.json")
IMG_SIZE      = 224
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEMO_DIR      = Path("demo_images")

print(f"\n  Device: {DEVICE}")
print(f"  Model: {MODEL_PATH}")


# ─────────────────────────────────────────
#  LOAD LABEL MAPPING
# ─────────────────────────────────────────
def load_mapping():
    with open(MAPPING_PATH) as f:
        return json.load(f)

mapping = load_mapping()
NUM_LVL1 = len(mapping["lvl1_classes"])
NUM_LVL2 = len(mapping["lvl2_classes"])

print(f"  Level 1 classes: {mapping['lvl1_classes']}")
print(f"  Level 2 classes: {mapping['lvl2_classes']}")


# ─────────────────────────────────────────
#  MODEL ARCHITECTURE
# ─────────────────────────────────────────
class HierarchicalGarbageNet(nn.Module):
    def __init__(self, backbone_name="efficientnet_b0",
                 num_lvl1=NUM_LVL1, num_lvl2=NUM_LVL2):
        super().__init__()

        base = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        feat_dim = base.num_features

        self.backbone = base

        self.shared_fc = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        self.head_lvl1 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_lvl1),
        )

        self.head_lvl2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_lvl2),
        )

    def forward(self, x):
        feat   = self.backbone(x)
        shared = self.shared_fc(feat)
        out1   = self.head_lvl1(shared)
        out2   = self.head_lvl2(shared)
        return out1, out2


# ─────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────
def load_model():
    print(f"\n[MODEL] Loading from {MODEL_PATH}...")
    model = HierarchicalGarbageNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    print(f"[MODEL] ✓ Model loaded successfully!")
    return model


model = load_model()

# Transform
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────
#  CREATE DEMO IMAGES
# ─────────────────────────────────────────
def create_demo_images():
    """Generate synthetic demo images for testing"""
    DEMO_DIR.mkdir(exist_ok=True)
    
    print(f"\n[DEMO] Creating synthetic test images...")
    
    colors_by_category = {
        "organik": [(34, 139, 34), (0, 100, 0), (50, 205, 50)],      # Green shades
        "anorganik": [(30, 144, 255), (0, 51, 102), (65, 105, 225)],  # Blue shades
        "b3": [(255, 0, 0), (139, 0, 0), (205, 92, 92)],             # Red shades
    }
    
    for category in mapping["lvl1_classes"]:
        colors = colors_by_category.get(category, [(128, 128, 128)])
        
        for i in range(3):  # 3 images per category
            # Create image
            img = Image.new("RGB", (640, 480), color=(240, 240, 240))
            draw = ImageDraw.Draw(img)
            
            # Draw colored shapes
            color = random.choice(colors)
            x = random.randint(50, 300)
            y = random.randint(50, 300)
            size = random.randint(80, 200)
            
            # Draw random shape
            shape_type = random.choice(["circle", "rect", "polygon"])
            if shape_type == "circle":
                draw.ellipse([x, y, x+size, y+size], fill=color, outline="black", width=2)
            elif shape_type == "rect":
                draw.rectangle([x, y, x+size, y+size], fill=color, outline="black", width=2)
            else:
                points = [(x, y), (x+size, y), (x+size//2, y+size)]
                draw.polygon(points, fill=color, outline="black")
            
            # Add label
            draw.text((10, 10), f"{category.upper()}", fill=(0, 0, 0))
            
            # Save
            filename = DEMO_DIR / f"demo_{category}_{i+1}.jpg"
            img.save(filename)
            print(f"  ✓ Created: {filename}")


# ─────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────
@torch.no_grad()
def predict_frame(frame_bgr):
    """Input: BGR frame dari OpenCV"""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    tensor = val_transform(pil_img).unsqueeze(0).to(DEVICE)
    
    out1, out2 = model(tensor)
    prob1 = torch.softmax(out1, 1)[0].cpu().numpy()
    prob2 = torch.softmax(out2, 1)[0].cpu().numpy()
    
    lvl1_idx = prob1.argmax()
    lvl2_idx = prob2.argmax()
    
    lvl1_pred = mapping["lvl1_classes"][lvl1_idx]
    lvl2_pred = mapping["lvl2_classes"][lvl2_idx]
    lvl1_conf = prob1[lvl1_idx] * 100
    lvl2_conf = prob2[lvl2_idx] * 100
    
    return lvl1_pred, lvl1_conf, lvl2_pred, lvl2_conf, prob1, prob2


# ─────────────────────────────────────────
#  COLORS
# ─────────────────────────────────────────
LEVEL1_COLORS = {
    "organik":   (0, 255, 0),
    "anorganik": (255, 0, 0),
    "b3":        (0, 0, 255),
}

def get_color(lvl1):
    return LEVEL1_COLORS.get(lvl1, (128, 128, 128))


# ─────────────────────────────────────────
#  DEMO LOOP
# ─────────────────────────────────────────
def demo():
    create_demo_images()
    
    # Get demo images
    demo_images = list(DEMO_DIR.glob("*.jpg"))
    
    if not demo_images:
        print("✗ No demo images found!")
        return
    
    print(f"\n[DEMO] Processing {len(demo_images)} demo images...")
    print(f"[KONTROL] Press any key to go to next image, Q to quit\n")
    
    for idx, img_path in enumerate(demo_images, 1):
        print(f"\n[{idx}/{len(demo_images)}] {img_path.name}")
        
        # Read image
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        
        # Predict
        try:
            lvl1, lvl1_conf, lvl2, lvl2_conf, prob1, prob2 = predict_frame(frame)
        except Exception as e:
            print(f"  Error: {e}")
            continue
        
        print(f"  ► Level 1: {lvl1.upper()} ({lvl1_conf:.1f}%)")
        print(f"  ► Level 2: {lvl2} ({lvl2_conf:.1f}%)")
        
        # Draw on image
        color = get_color(lvl1)
        display_frame = frame.copy()
        height, width = display_frame.shape[:2]
        
        # Draw rectangle and text
        cv2.rectangle(display_frame, (10, 10), (width-10, 150), color, 3)
        cv2.putText(display_frame, f"Level 1: {lvl1.upper()}", 
                   (30, 60), cv2.FONT_HERSHEY_BOLD, 1.5, color, 2)
        cv2.putText(display_frame, f"Confidence: {lvl1_conf:.1f}%", 
                   (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(display_frame, f"Level 2: {lvl2}", 
                   (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
        
        # Display
        cv2.imshow("Garbage Detection - DEMO MODE", display_frame)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print("\n✓ Demo completed!")


# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  GARBAGE DETECTION — DEMO MODE (No Camera)")
    print("=" * 60)
    
    demo()
