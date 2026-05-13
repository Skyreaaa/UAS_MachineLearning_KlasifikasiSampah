"""
====================================================
  SCRIPT 4: REAL-TIME GARBAGE DETECTION + AUTO TRACKING
  ─────────────────────────────────────────────────
  Deteksi objek sampah secara otomatis menggunakan kamera.
  Box bounding muncul sendiri & mengikuti objek di frame.

  Cara kerja:
    1. Warmup (40 frame) → model background MOG2 dibangun
    2. Setiap frame: foreground detection via background subtraction
    3. Kontur terbesar dijadikan bounding box (dengan padding & EMA smoothing)
    4. Inference dilakukan pada crop area bounding box
    5. Hasil (Level 1 + Level 2 + confidence %) ditampilkan di dalam box

  Kontrol:
    R     = Reset background model (ulangi warmup)
    C     = Ganti kamera (cycle ke kamera berikutnya)
    SPACE = Capture screenshot
    S     = Toggle Full / Minimal display
    Q     = Quit
====================================================
Install:
  pip install opencv-python torch torchvision timm Pillow
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import cv2
import torch
import torch.nn as nn
import json
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import timm
import numpy as np
from datetime import datetime

# ─────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────
MODEL_PATH   = Path("model/best_model.pth")
MAPPING_PATH = Path("dataset/label_mapping.json")
IMG_SIZE     = 224
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MIN_AREA      = 5000    # luas minimum foreground (pixel^2) agar dianggap objek
BOX_PAD       = 30      # padding di sekitar bounding box objek
ALPHA         = 0.30    # EMA smoothing — kecil=halus, besar=responsif
HIDE_AFTER    = 20      # sembunyikan box setelah N frame tanpa deteksi
WARMUP_FRAMES = 40      # frame awal untuk bangun model background

print(f"\n  Device : {DEVICE}")
print(f"  Model  : {MODEL_PATH}")


# ─────────────────────────────────────────────────
#  LABEL MAPPING
# ─────────────────────────────────────────────────
with open(MAPPING_PATH) as f:
    mapping = json.load(f)

NUM_LVL1 = len(mapping["lvl1_classes"])
NUM_LVL2 = len(mapping["lvl2_classes"])
print(f"  Level 1: {mapping['lvl1_classes']}")
print(f"  Level 2: {mapping['lvl2_classes']}")


# ─────────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────────
class HierarchicalGarbageNet(nn.Module):
    def __init__(self, backbone_name="efficientnet_b0",
                 num_lvl1=NUM_LVL1, num_lvl2=NUM_LVL2):
        super().__init__()
        base     = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        feat_dim = base.num_features
        self.backbone  = base
        self.shared_fc = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4))
        self.head_lvl1 = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_lvl1))
        self.head_lvl2 = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_lvl2))

    def forward(self, x):
        feat   = self.backbone(x)
        shared = self.shared_fc(feat)
        return self.head_lvl1(shared), self.head_lvl2(shared)


def load_model():
    print(f"\n[MODEL] Loading {MODEL_PATH}...")
    m = HierarchicalGarbageNet()
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    m.to(DEVICE).eval()
    print("[MODEL] Loaded!")
    return m

model = load_model()

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────
#  COLORS
# ─────────────────────────────────────────────────
COLORS = {
    "organik":   (30, 210, 30),
    "anorganik": (30, 150, 255),
    "b3":        (40, 40, 255),
}

def get_color(lvl1):
    return COLORS.get(lvl1, (160, 160, 160))


# ─────────────────────────────────────────────────
#  CAMERA
# ─────────────────────────────────────────────────
def scan_cameras():
    """Scan semua kamera yang tersedia (index 0–5, backend DSHOW & MSMF). Return list of (index, backend_id, backend_name, resolution)."""
    print("\n[CAMERA] Scanning...")
    available = []
    found     = set()
    for bid, bname in [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF")]:
        for i in range(6):
            if i in found:
                continue
            try:
                c = cv2.VideoCapture(i, bid)
                if c.isOpened():
                    ret, fr = c.read()
                    if ret and fr is not None:
                        w = int(c.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        available.append((i, bid, bname, f"{w}x{h}"))
                        found.add(i)
                        print(f"  [{len(available)-1}] Kamera {i} ({bname}) - {w}x{h}")
                c.release()
            except Exception:
                pass
    return available


def open_camera(index, backend_id):
    """Buka kamera dengan index & backend tertentu, set resolusi 1280×720 @ 30fps."""
    cap = cv2.VideoCapture(index, backend_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def select_camera(available):
    """Tampilkan pilihan kamera di terminal; return index pilihan dalam list available."""
    if not available:
        return None
    if len(available) == 1:
        return 0
    print("\n  Pilih kamera (nomor + Enter):", end=" ")
    try:
        c = int(input().strip())
        if 0 <= c < len(available):
            return c
    except (ValueError, EOFError):
        pass
    return 0


# ─────────────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────────────
@torch.no_grad()
def predict_roi(frame_bgr, x1, y1, x2, y2):
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    pil     = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    t       = val_transform(pil).unsqueeze(0).to(DEVICE)
    o1, o2  = model(t)
    p1      = torch.softmax(o1, 1)[0].cpu().numpy()
    p2      = torch.softmax(o2, 1)[0].cpu().numpy()
    i1, i2  = p1.argmax(), p2.argmax()
    return (mapping["lvl1_classes"][i1], float(p1[i1]) * 100,
            mapping["lvl2_classes"][i2], float(p2[i2]) * 100,
            p1, p2)


# ─────────────────────────────────────────────────
#  FOREGROUND DETECTION
# ─────────────────────────────────────────────────
def detect_object(frame, fgbg, learning_rate):
    """
    Deteksi objek foreground terbesar menggunakan background subtraction MOG2.

    Pipeline:
        1. apply MOG2 → foreground mask
        2. Morphological open+close+dilate untuk menghilangkan noise
        3. Cari kontur terbesar; filter dengan MIN_AREA
        4. Return bounding box (x1,y1,x2,y2) dengan BOX_PAD, atau None jika tidak ada objek.
    """
    """Deteksi objek foreground terbesar. Return (x1,y1,x2,y2) atau None."""
    H, W = frame.shape[:2]

    fg = fgbg.apply(frame, learningRate=learning_rate)

    # Bersihkan noise
    k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  k, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=3)
    fg = cv2.dilate(fg, k, iterations=2)

    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_AREA:
        return None

    x, y, w, h = cv2.boundingRect(largest)
    x1 = max(0, x - BOX_PAD)
    y1 = max(0, y - BOX_PAD)
    x2 = min(W, x + w + BOX_PAD)
    y2 = min(H, y + h + BOX_PAD)
    return (x1, y1, x2, y2)


# ─────────────────────────────────────────────────
#  DRAW TRACKING BOX
# ─────────────────────────────────────────────────
def draw_box(frame, x1, y1, x2, y2, color, lvl1, c1, lvl2, c2):
    """Gambar bounding box bergaya (sudut dekoratif, area luar digelapkan, label + confidence di dalam box)."""
    box_w = x2 - x1
    box_h = y2 - y1
    corner = min(28, box_w // 4, box_h // 4)
    th     = 3

    # Gelapkan area luar kotak
    mask              = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    dark              = frame.copy()
    dark[mask == 0]   = (dark[mask == 0] * 0.38).astype(np.uint8)
    np.copyto(frame, dark)

    # Border tipis
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    # Sudut dekoratif
    for px, py, dx, dy in [
        (x1, y1, +1, +1), (x2, y1, -1, +1),
        (x1, y2, +1, -1), (x2, y2, -1, -1),
    ]:
        cv2.line(frame, (px, py), (px + dx * corner, py), color, th)
        cv2.line(frame, (px, py), (px, py + dy * corner), color, th)

    # ── Confidence % — besar, di dalam kotak (bawah) ──
    conf_txt   = f"{c1:.1f}%"
    conf_scale = max(0.7, min(2.0, box_w / 190))
    cf_sz, _   = cv2.getTextSize(conf_txt, cv2.FONT_HERSHEY_DUPLEX, conf_scale, 2)
    cf_x = x1 + (box_w - cf_sz[0]) // 2
    cf_y = y2 - 12
    # shadow
    cv2.putText(frame, conf_txt, (cf_x + 2, cf_y + 2),
                cv2.FONT_HERSHEY_DUPLEX, conf_scale, (0, 0, 0), 3)
    cv2.putText(frame, conf_txt, (cf_x, cf_y),
                cv2.FONT_HERSHEY_DUPLEX, conf_scale, color, 2)

    # ── Kategori Level 1 ──
    l1_scale = max(0.6, min(1.3, box_w / 230))
    l1_sz, _ = cv2.getTextSize(lvl1.upper(), cv2.FONT_HERSHEY_DUPLEX, l1_scale, 2)
    l1_x = x1 + (box_w - l1_sz[0]) // 2
    l1_y = cf_y - cf_sz[1] - 8
    cv2.putText(frame, lvl1.upper(), (l1_x + 2, l1_y + 2),
                cv2.FONT_HERSHEY_DUPLEX, l1_scale, (0, 0, 0), 3)
    cv2.putText(frame, lvl1.upper(), (l1_x, l1_y),
                cv2.FONT_HERSHEY_DUPLEX, l1_scale, color, 2)

    # ── Level 2 — di atas kategori ──
    l2_txt  = f"{lvl2}  ({c2:.0f}%)"
    l2_sz, _ = cv2.getTextSize(l2_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    l2_x = x1 + (box_w - l2_sz[0]) // 2
    l2_y = l1_y - l1_sz[1] - 6
    if l2_y > y1 + 10:
        cv2.rectangle(frame,
                      (l2_x - 4, l2_y - l2_sz[1] - 3),
                      (l2_x + l2_sz[0] + 4, l2_y + 3),
                      (0, 0, 0), -1)
        cv2.putText(frame, l2_txt, (l2_x, l2_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)


def draw_no_object_hint(frame):
    """Tampilkan crosshair dan teks panduan di tengah frame saat tidak ada objek terdeteksi."""
    """Tampilkan hint di tengah saat tidak ada objek terdeteksi."""
    H, W = frame.shape[:2]
    msg   = "Letakkan / gerakkan sampah di depan kamera"
    scale = 0.58
    sz, _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    tx = (W - sz[0]) // 2
    ty = H // 2

    # Crosshair kecil di tengah
    cx, cy, arm = W // 2, H // 2, 18
    cv2.line(frame, (cx - arm, cy), (cx + arm, cy), (100, 100, 100), 1)
    cv2.line(frame, (cx, cy - arm), (cx, cy + arm), (100, 100, 100), 1)
    cv2.circle(frame, (cx, cy), 4, (100, 100, 100), 1)

    # Background teks
    cv2.rectangle(frame, (tx - 8, ty + 18 - sz[1] - 4),
                  (tx + sz[0] + 8, ty + 22), (0, 0, 0), -1)
    cv2.putText(frame, msg, (tx, ty + 18),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (160, 160, 160), 1)


# ─────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────
def main():
    available = scan_cameras()
    if not available:
        print("\n[ERROR] Tidak ada kamera terdeteksi!")
        return

    cam_list_idx = select_camera(available)
    cam_idx, cam_bid, cam_bname, _ = available[cam_list_idx]
    cap = open_camera(cam_idx, cam_bid)
    if not cap.isOpened():
        print(f"[ERROR] Gagal buka kamera {cam_idx}")
        return

    print(f"\n[CAMERA] Aktif: index={cam_idx} ({cam_bname})")
    print(f"\n[KONTROL]")
    print(f"  R     = Reset background model")
    print(f"  C     = Ganti kamera")
    print(f"  SPACE = Capture screenshot")
    print(f"  S     = Toggle Full / Minimal")
    print(f"  Q     = Quit")
    print(f"\n{'='*60}\n")
    print("[INFO] Warming up background model...")

    # State
    display_mode  = "full"
    frame_count   = 0
    fgbg          = cv2.createBackgroundSubtractorMOG2(
                        history=150, varThreshold=60, detectShadows=False)
    smooth_box    = None   # (x1,y1,x2,y2) float
    no_det_frames = 0
    last_result   = None   # hasil inference terakhir

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        H, W = frame.shape[:2]

        # ── Warmup: bangun background model dulu ──
        if frame_count <= WARMUP_FRAMES:
            fgbg.apply(frame, learningRate=0.1)
            msg = f"Initializing... ({frame_count}/{WARMUP_FRAMES})"
            cv2.putText(frame, msg, (W // 2 - 150, H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
            cv2.imshow("Garbage Detection - Auto Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # ── Deteksi foreground ──
        # Lambat saat ada objek (freeze background), lebih cepat saat kosong
        lr       = 0.002 if smooth_box is not None else 0.008
        raw_box  = detect_object(frame, fgbg, lr)

        if raw_box is not None:
            no_det_frames = 0
            nb = tuple(float(v) for v in raw_box)
            if smooth_box is None:
                smooth_box = nb
            else:
                smooth_box = tuple(ALPHA * n + (1.0 - ALPHA) * s
                                   for n, s in zip(nb, smooth_box))
        else:
            no_det_frames += 1
            if no_det_frames >= HIDE_AFTER:
                smooth_box = None

        # ── Inference pada box jika ada ──
        display_frame = frame.copy()

        if smooth_box is not None:
            bx1, by1, bx2, by2 = (int(v) for v in smooth_box)
            bx1 = max(0, bx1); by1 = max(0, by1)
            bx2 = min(W, bx2); by2 = min(H, by2)

            if bx2 - bx1 > 30 and by2 - by1 > 30:
                try:
                    result = predict_roi(display_frame, bx1, by1, bx2, by2)
                    if result:
                        last_result = result
                except Exception as e:
                    print(f"[ERROR] {e}")

                if last_result:
                    lvl1, c1, lvl2, c2, p1, p2 = last_result
                    color = get_color(lvl1)
                    draw_box(display_frame, bx1, by1, bx2, by2,
                             color, lvl1, c1, lvl2, c2)

                    if display_mode == "full":
                        # Bar level 1 — kiri bawah
                        bx_bar = 10
                        by_bar = H - 90
                        cv2.putText(display_frame, "Level 1:",
                                    (bx_bar, by_bar - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1)
                        for i, (cls, prob) in enumerate(
                                zip(mapping["lvl1_classes"], p1)):
                            yo  = by_bar + i * 24
                            bl  = int(130 * prob)
                            ci  = get_color(cls)
                            cv2.rectangle(display_frame,
                                          (bx_bar, yo), (bx_bar + 130, yo + 18),
                                          (70, 70, 70), 1)
                            cv2.rectangle(display_frame,
                                          (bx_bar, yo), (bx_bar + bl, yo + 18),
                                          ci, -1)
                            cv2.putText(display_frame,
                                        f"{cls}: {prob*100:.0f}%",
                                        (bx_bar + 138, yo + 13),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, ci, 1)
        else:
            # Tidak ada objek — tampilkan hint
            draw_no_object_hint(display_frame)
            last_result = None

        # ── Panel atas ──
        if display_mode == "full":
            ph = 32
            ov = display_frame.copy()
            cv2.rectangle(ov, (0, 0), (W, ph), (0, 0, 0), -1)
            cv2.addWeighted(ov, 0.55, display_frame, 0.45, 0, display_frame)
            hints = "[R] Reset  [C] Kamera  [S] Mode  [SPACE] Simpan  [Q] Quit"
            cv2.putText(display_frame, hints, (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 170, 170), 1)
            cam_info = f"CAM {cam_idx}({cam_bname})  #{frame_count}"
            ci_sz, _ = cv2.getTextSize(cam_info, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            cv2.putText(display_frame, cam_info, (W - ci_sz[0] - 8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (110, 110, 110), 1)

        cv2.imshow("Garbage Detection - Auto Tracking", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\n[INFO] Keluar...")
            break

        elif key == ord('r'):
            fgbg       = cv2.createBackgroundSubtractorMOG2(
                             history=150, varThreshold=60, detectShadows=False)
            smooth_box    = None
            no_det_frames = 0
            last_result   = None
            frame_count   = 0
            print("[INFO] Background model di-reset")

        elif key == ord(' '):
            fname = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(fname, display_frame)
            print(f"[SAVED] {fname}")
            if last_result:
                lvl1, c1, lvl2, c2, *_ = last_result
                print(f"  -> {lvl1} {c1:.1f}%  |  {lvl2} {c2:.1f}%")

        elif key == ord('s'):
            display_mode = "minimal" if display_mode == "full" else "full"
            print(f"[MODE] {'MINIMAL' if display_mode == 'minimal' else 'FULL'}")

        elif key == ord('c'):
            cap.release()
            cam_list_idx = (cam_list_idx + 1) % len(available)
            cam_idx, cam_bid, cam_bname, _ = available[cam_list_idx]
            cap           = open_camera(cam_idx, cam_bid)
            smooth_box    = None
            no_det_frames = 0
            last_result   = None
            frame_count   = 0
            fgbg          = cv2.createBackgroundSubtractorMOG2(
                                history=150, varThreshold=60, detectShadows=False)
            print(f"[CAMERA] Ganti ke index={cam_idx} ({cam_bname})")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Selesai!")


if __name__ == "__main__":
    print("=" * 60)
    print("  GARBAGE DETECTION - AUTO TRACKING")
    print("=" * 60)
    main()
