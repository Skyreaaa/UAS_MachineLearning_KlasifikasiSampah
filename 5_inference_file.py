"""
====================================================
  SCRIPT 5: GARBAGE DETECTION — IMAGE/VIDEO MODE
  Jalankan inference pada file image atau video
  (Alternatif kalau tidak ada kamera)
====================================================
Install:
  pip install opencv-python torch torchvision timm Pillow

Jalankan (image):
  python 5_inference_file.py --image path/ke/gambar.jpg

Jalankan (video):
  python 5_inference_file.py --video path/ke/video.mp4

Jalankan (folder gambar):
  python 5_inference_file.py --folder path/ke/folder/
====================================================
"""

import cv2
import torch
import torch.nn as nn
import json
import argparse
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import timm
import numpy as np
from datetime import datetime
import os

# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────
MODEL_PATH    = Path("model/best_model.pth")
MAPPING_PATH  = Path("dataset/label_mapping.json")
IMG_SIZE      = 224
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
#  DEFINE MODEL ARCHITECTURE
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

# Transform untuk inference
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────
#  INFERENCE FUNCTION
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
#  IMAGE MODE
# ─────────────────────────────────────────
def process_image(image_path):
    print(f"\n[IMAGE] Processing: {image_path}")
    
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"✗ Error: Tidak bisa buka image: {image_path}")
        return
    
    print(f"[INFO] Image size: {frame.shape}")
    
    # Predict
    lvl1, lvl1_conf, lvl2, lvl2_conf, prob1, prob2 = predict_frame(frame)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"  HASIL PREDIKSI")
    print(f"{'='*60}")
    print(f"  Level 1 (Kategori Utama):")
    print(f"    → {lvl1.upper()} ({lvl1_conf:.1f}%)")
    print(f"    • Organik:   {prob1[0]*100:.1f}%")
    print(f"    • Anorganik: {prob1[1]*100:.1f}%")
    print(f"    • B3:        {prob1[2]*100:.1f}%")
    print(f"\n  Level 2 (Subkategori):")
    print(f"    → {lvl2} ({lvl2_conf:.1f}%)")
    print(f"\n  Top 3 Subkategori:")
    top3_idx = np.argsort(prob2)[-3:][::-1]
    for i, idx in enumerate(top3_idx, 1):
        print(f"    {i}. {mapping['lvl2_classes'][idx]}: {prob2[idx]*100:.1f}%")
    print(f"{'='*60}\n")
    
    # Draw on image
    color = get_color(lvl1)
    display_frame = frame.copy()
    
    cv2.rectangle(display_frame, (10, 10), (display_frame.shape[1]-10, 200), color, 2)
    cv2.putText(display_frame, f"Level 1: {lvl1.upper()}", 
               (30, 60), cv2.FONT_HERSHEY_BOLD, 1.5, color, 2)
    cv2.putText(display_frame, f"Confidence: {lvl1_conf:.1f}%", 
               (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(display_frame, f"Level 2: {lvl2} ({lvl2_conf:.1f}%)", 
               (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 1)
    
    # Save output
    output_path = Path(image_path).stem + "_detected.jpg"
    cv2.imwrite(output_path, display_frame)
    print(f"  ✓ Hasil disimpan: {output_path}")
    
    # Display
    cv2.imshow("Garbage Detection - Image", display_frame)
    print(f"  (Tekan tombol apapun untuk tutup...)")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─────────────────────────────────────────
#  VIDEO MODE
# ─────────────────────────────────────────
def process_video(video_path):
    print(f"\n[VIDEO] Processing: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"✗ Error: Tidak bisa buka video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"[VIDEO] FPS: {fps}, Resolution: {width}x{height}, Total frames: {total_frames}")
    print(f"[KONTROL] Q = Quit\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Predict every 5 frames (untuk performa)
        if frame_count % 5 == 0:
            try:
                lvl1, lvl1_conf, lvl2, lvl2_conf, prob1, prob2 = predict_frame(frame)
            except Exception as e:
                print(f"Error: {e}")
                continue
            
            # Draw predictions
            color = get_color(lvl1)
            cv2.rectangle(frame, (10, 10), (400, 120), color, 2)
            cv2.putText(frame, lvl1.upper(), (30, 60), cv2.FONT_HERSHEY_BOLD, 1.5, color, 2)
            cv2.putText(frame, f"{lvl1_conf:.1f}%", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        # Frame info
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", 
                   (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        
        cv2.imshow("Garbage Detection - Video", frame)
        
        if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✓ Video processing complete!")


# ─────────────────────────────────────────
#  FOLDER MODE
# ─────────────────────────────────────────
def process_folder(folder_path):
    print(f"\n[FOLDER] Processing images in: {folder_path}")
    
    folder = Path(folder_path)
    image_files = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg"))
    
    if not image_files:
        print(f"✗ Error: Tidak ada image ditemukan di {folder_path}")
        return
    
    print(f"[INFO] Ditemukan {len(image_files)} image(s)\n")
    
    results = []
    for i, img_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Processing: {img_path.name}")
        
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  ✗ Skip (error membaca)")
            continue
        
        try:
            lvl1, lvl1_conf, lvl2, lvl2_conf, prob1, prob2 = predict_frame(frame)
            print(f"  ✓ {lvl1.upper()} ({lvl1_conf:.1f}%) → {lvl2}")
            results.append({
                "file": img_path.name,
                "level1": lvl1,
                "level1_conf": lvl1_conf,
                "level2": lvl2,
                "level2_conf": lvl2_conf,
            })
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY ({len(results)} images)")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['file']:<30} {r['level1']:<12} {r['level1_conf']:>6.1f}%")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Garbage Detection - Image/Video Mode")
    parser.add_argument("--image", help="Path ke image file")
    parser.add_argument("--video", help="Path ke video file")
    parser.add_argument("--folder", help="Path ke folder berisi image")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  GARBAGE DETECTION — IMAGE/VIDEO MODE")
    print("=" * 60)
    
    if args.image:
        process_image(args.image)
    elif args.video:
        process_video(args.video)
    elif args.folder:
        process_folder(args.folder)
    else:
        print("\n✗ Error: Pilih salah satu: --image, --video, atau --folder")
        print("\nContoh penggunaan:")
        print("  python 5_inference_file.py --image path/ke/gambar.jpg")
        print("  python 5_inference_file.py --video path/ke/video.mp4")
        print("  python 5_inference_file.py --folder path/ke/folder/")
