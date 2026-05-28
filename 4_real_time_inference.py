"""
====================================================
  SCRIPT 4: REAL-TIME GARBAGE DETECTION (YOLO HYBRID)
  ─────────────────────────────────────────────────
  Cara kerja (Hybrid Option C):
    Detector  → YOLOv8n.pt (COCO pretrained)
                Menemukan objek apa saja dalam frame secara otomatis.
    Classifier→ model/yolo_best.pt (hasil 3_yolo_training.py)
                Mengklasifikasikan crop objek ke 12 kelas sampah.
    Mapping   → Level-2 (subkelas) → Level-1 (Organik/Anorganik/B3)
                via label_mapping.json

  Kontrol:
    C     = Ganti kamera
    SPACE = Capture screenshot
    S     = Toggle Full / Minimal
    Q     = Quit
====================================================
Install:
  pip install ultralytics opencv-python Pillow
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import cv2, json
from pathlib import Path
from datetime import datetime
import numpy as np
from ultralytics import YOLO


# ─────────────────────────────────────────────────
#  KONFIGURASI
# ─────────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent.resolve()
DETECTOR_MODEL   = "yolov8n.pt"                                   # COCO pretrained (auto-download)
CLASSIFIER_MODEL = BASE_DIR / "model" / "yolo_best.pt"            # hasil 3_yolo_training.py
MAPPING_PATH     = BASE_DIR / "dataset" / "label_mapping.json"

DET_CONF  = 0.20   # confidence minimum detector (turunkan jika objek sering tidak terdeteksi)
BOX_PAD   = 20     # padding di sekitar bounding box detector

print(f"\n  Detector  : {DETECTOR_MODEL}")
print(f"  Classifier: {CLASSIFIER_MODEL}")


# ─────────────────────────────────────────────────
#  VALIDASI FILE
# ─────────────────────────────────────────────────
if not CLASSIFIER_MODEL.exists():
    print(f"\n[ERROR] Classifier belum ada: {CLASSIFIER_MODEL}")
    print("  Jalankan dulu: python 3_yolo_training.py")
    sys.exit(1)

if not MAPPING_PATH.exists():
    print(f"\n[ERROR] Label mapping tidak ditemukan: {MAPPING_PATH}")
    sys.exit(1)


# ─────────────────────────────────────────────────
#  LOAD MODEL & MAPPING
# ─────────────────────────────────────────────────
print("\n[MODEL] Loading models...")
detector   = YOLO(DETECTOR_MODEL)
classifier = YOLO(str(CLASSIFIER_MODEL))
print("  ✓ Detector  loaded")
print("  ✓ Classifier loaded")

with open(MAPPING_PATH) as f:
    mapping = json.load(f)

sub_to_lvl1 = mapping["sub_to_lvl1"]   # lvl2_name → lvl1_name
print(f"  Level 1 : {mapping['lvl1_classes']}")
print(f"  Level 2 : {mapping['lvl2_classes']}")


# ─────────────────────────────────────────────────
#  COLORS
# ─────────────────────────────────────────────────
COLORS = {
    "organik"  : (30, 210, 30),
    "anorganik": (30, 150, 255),
    "b3"       : (40,  40, 255),
}

def get_color(lvl1):
    """Return warna BGR berdasarkan kategori Level-1."""
    return COLORS.get(lvl1, (160, 160, 160))


# ─────────────────────────────────────────────────
#  CAMERA
# ─────────────────────────────────────────────────
def scan_cameras():
    """Scan semua kamera yang tersedia (index 0–5, backend DSHOW & MSMF)."""
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
    """Buka kamera dan set resolusi 1280×720 @ 30fps."""
    cap = cv2.VideoCapture(index, backend_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def select_camera(available):
    """Tampilkan pilihan kamera; return index dalam list available."""
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
#  DETEKSI & KLASIFIKASI
# ─────────────────────────────────────────────────
def detect_largest(frame):
    """
    Jalankan YOLO detector pada frame; return bounding box (x1,y1,x2,y2) objek terbesar.
    Return None jika tidak ada deteksi di atas DET_CONF.
    """
    H, W = frame.shape[:2]
    results = detector(frame, conf=DET_CONF, verbose=False)
    boxes   = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return None

    # Ambil bounding box dengan area terbesar
    best_box  = None
    best_area = 0
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area = area
            best_box  = (
                max(0, x1 - BOX_PAD),
                max(0, y1 - BOX_PAD),
                min(W, x2 + BOX_PAD),
                min(H, y2 + BOX_PAD),
            )
    return best_box


def classify_crop(frame, x1, y1, x2, y2):
    """
    Klasifikasikan crop frame[y1:y2, x1:x2] dengan YOLO classifier.
    Return (lvl1_name, lvl2_name, lvl1_conf%, lvl2_conf%) atau None jika crop kosong.
    """
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    results  = classifier(crop, verbose=False)
    probs    = results[0].probs
    top1_idx = probs.top1
    top1_conf= float(probs.top1conf) * 100
    lvl2_name= results[0].names[top1_idx]
    lvl1_name= sub_to_lvl1.get(lvl2_name, "unknown")

    # Top-5 probs untuk bar chart
    top5_idx  = probs.top5
    top5_conf = [float(probs.data[i]) * 100 for i in top5_idx]

    return lvl1_name, top1_conf, lvl2_name, top5_conf[0], probs


# ─────────────────────────────────────────────────
#  DRAW
# ─────────────────────────────────────────────────
def draw_box(frame, x1, y1, x2, y2, color, lvl1, c1, lvl2, c2):
    """Gambar bounding box bergaya dengan label klasifikasi dan confidence %."""
    box_w  = x2 - x1
    corner = min(28, box_w // 4, (y2 - y1) // 4)
    th     = 3

    # Gelapkan area luar kotak
    mask             = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    dark             = frame.copy()
    dark[mask == 0]  = (dark[mask == 0] * 0.38).astype(np.uint8)
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

    # Confidence % — besar di bawah dalam kotak
    conf_txt   = f"{c1:.1f}%"
    conf_scale = max(0.7, min(2.0, box_w / 190))
    cf_sz, _   = cv2.getTextSize(conf_txt, cv2.FONT_HERSHEY_DUPLEX, conf_scale, 2)
    cf_x = x1 + (box_w - cf_sz[0]) // 2
    cf_y = y2 - 12
    cv2.putText(frame, conf_txt, (cf_x + 2, cf_y + 2),
                cv2.FONT_HERSHEY_DUPLEX, conf_scale, (0, 0, 0), 3)
    cv2.putText(frame, conf_txt, (cf_x, cf_y),
                cv2.FONT_HERSHEY_DUPLEX, conf_scale, color, 2)

    # Level 1 kategori
    l1_scale = max(0.6, min(1.3, box_w / 230))
    l1_sz, _ = cv2.getTextSize(lvl1.upper(), cv2.FONT_HERSHEY_DUPLEX, l1_scale, 2)
    l1_x = x1 + (box_w - l1_sz[0]) // 2
    l1_y = cf_y - cf_sz[1] - 8
    cv2.putText(frame, lvl1.upper(), (l1_x + 2, l1_y + 2),
                cv2.FONT_HERSHEY_DUPLEX, l1_scale, (0, 0, 0), 3)
    cv2.putText(frame, lvl1.upper(), (l1_x, l1_y),
                cv2.FONT_HERSHEY_DUPLEX, l1_scale, color, 2)

    # Level 2 subkelas
    l2_txt  = f"{lvl2}  ({c2:.0f}%)"
    l2_sz, _= cv2.getTextSize(l2_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    l2_x = x1 + (box_w - l2_sz[0]) // 2
    l2_y = l1_y - l1_sz[1] - 6
    if l2_y > y1 + 10:
        cv2.rectangle(frame,
                      (l2_x - 4, l2_y - l2_sz[1] - 3),
                      (l2_x + l2_sz[0] + 4, l2_y + 3), (0, 0, 0), -1)
        cv2.putText(frame, l2_txt, (l2_x, l2_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)

    # Label "YOLO" kecil di sudut kiri atas kotak
    cv2.putText(frame, "YOLO", (x1 + 6, y1 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)


def draw_no_object_hint(frame):
    """Tampilkan crosshair dan hint teks saat tidak ada objek terdeteksi."""
    H, W  = frame.shape[:2]
    msg   = "Letakkan / gerakkan sampah di depan kamera"
    scale = 0.58
    sz, _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    tx    = (W - sz[0]) // 2
    ty    = H // 2

    cx, cy, arm = W // 2, H // 2, 18
    cv2.line(frame, (cx - arm, cy), (cx + arm, cy), (100, 100, 100), 1)
    cv2.line(frame, (cx, cy - arm), (cx, cy + arm), (100, 100, 100), 1)
    cv2.circle(frame, (cx, cy), 4, (100, 100, 100), 1)

    cv2.rectangle(frame, (tx - 8, ty + 18 - sz[1] - 4),
                  (tx + sz[0] + 8, ty + 22), (0, 0, 0), -1)
    cv2.putText(frame, msg, (tx, ty + 18),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (160, 160, 160), 1)


# ─────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────
def main():
    available = scan_cameras()
    if not available:
        print("\n[ERROR] Tidak ada kamera terdeteksi!")
        return

    cam_idx_list = select_camera(available)
    cam_idx, cam_bid, cam_bname, _ = available[cam_idx_list]
    cap = open_camera(cam_idx, cam_bid)
    if not cap.isOpened():
        print(f"[ERROR] Gagal buka kamera {cam_idx}")
        return

    print(f"\n[CAMERA] Aktif: index={cam_idx} ({cam_bname})")
    print(f"\n[KONTROL]")
    print(f"  C     = Ganti kamera")
    print(f"  SPACE = Capture screenshot")
    print(f"  S     = Toggle Full / Minimal")
    print(f"  Q     = Quit")
    print(f"\n{'='*60}\n")

    display_mode = "full"
    frame_count  = 0
    last_result  = None   # simpan hasil terakhir

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        H, W = frame.shape[:2]

        # ── Deteksi objek terbesar ──
        bbox = detect_largest(frame)
        display_frame = frame.copy()

        if bbox is not None:
            x1, y1, x2, y2 = bbox

            # ── Klasifikasi crop ──
            result = classify_crop(display_frame, x1, y1, x2, y2)
            if result:
                last_result = result
                lvl1, c1, lvl2, c2, probs = result
                color = get_color(lvl1)
                draw_box(display_frame, x1, y1, x2, y2, color, lvl1, c1, lvl2, c2)

                if display_mode == "full":
                    # Bar top-5 Level-2 — kiri bawah
                    cls_names  = [classifier.names[i] for i in probs.top5]
                    cls_confs  = [float(probs.data[i]) * 100 for i in probs.top5]
                    bx, by     = 10, H - 30 - len(cls_names) * 24
                    cv2.putText(display_frame, "Top subkelas:",
                                (bx, by - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (190, 190, 190), 1)
                    for i, (cn, cf) in enumerate(zip(cls_names, cls_confs)):
                        yo  = by + i * 24
                        lvl = sub_to_lvl1.get(cn, "")
                        ci  = get_color(lvl)
                        bl  = int(130 * cf / 100)
                        cv2.rectangle(display_frame, (bx, yo), (bx + 130, yo + 18), (70, 70, 70), 1)
                        cv2.rectangle(display_frame, (bx, yo), (bx + bl,  yo + 18), ci, -1)
                        cv2.putText(display_frame, f"{cn}: {cf:.0f}%",
                                    (bx + 138, yo + 13),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, ci, 1)
        else:
            draw_no_object_hint(display_frame)
            last_result = None

        # ── Panel atas ──
        if display_mode == "full":
            ph = 32
            ov = display_frame.copy()
            cv2.rectangle(ov, (0, 0), (W, ph), (0, 0, 0), -1)
            cv2.addWeighted(ov, 0.55, display_frame, 0.45, 0, display_frame)
            hints = "[C] Kamera  [S] Mode  [SPACE] Simpan  [Q] Quit  |  YOLO Hybrid"
            cv2.putText(display_frame, hints, (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1)
            ci_txt = f"CAM {cam_idx}({cam_bname})  #{frame_count}"
            ci_sz, _ = cv2.getTextSize(ci_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.putText(display_frame, ci_txt, (W - ci_sz[0] - 8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (110, 110, 110), 1)
        else:
            if last_result:
                lvl1, c1, *_ = last_result
                color = get_color(lvl1)
                ov = display_frame.copy()
                cv2.rectangle(ov, (0, 0), (220, 60), (0, 0, 0), -1)
                cv2.addWeighted(ov, 0.5, display_frame, 0.5, 0, display_frame)
                cv2.putText(display_frame, lvl1.upper(),
                            (10, 44), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)

        cv2.imshow("Garbage Detection - YOLO Hybrid", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\n[INFO] Keluar...")
            break

        elif key == ord(' '):
            fname = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(fname, display_frame)
            print(f"[SAVED] {fname}")
            if last_result:
                lvl1, c1, lvl2, c2, _ = last_result
                print(f"  -> {lvl1} {c1:.1f}%  |  {lvl2} {c2:.1f}%")

        elif key == ord('s'):
            display_mode = "minimal" if display_mode == "full" else "full"
            print(f"[MODE] {'MINIMAL' if display_mode == 'minimal' else 'FULL'}")

        elif key == ord('c'):
            cap.release()
            cam_idx_list = (cam_idx_list + 1) % len(available)
            cam_idx, cam_bid, cam_bname, _ = available[cam_idx_list]
            cap = open_camera(cam_idx, cam_bid)
            print(f"[CAMERA] Ganti ke index={cam_idx} ({cam_bname})")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Selesai!")


# ─────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  GARBAGE DETECTION - YOLO HYBRID (Detect + Classify)")
    print("=" * 60)
    main()
