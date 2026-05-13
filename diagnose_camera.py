"""
====================================================
  CAMERA DIAGNOSTIC TOOL
  Test dan diagnose masalah kamera
====================================================
"""

import cv2
import os
import sys

def test_camera():
    print("\n" + "="*60)
    print("  CAMERA DIAGNOSTIC")
    print("="*60)
    
    # Test 1: Check OpenCV
    print("\n[1] Checking OpenCV...")
    print(f"    OpenCV version: {cv2.__version__}")
    print(f"    ✓ OpenCV installed")
    
    # Test 2: List available cameras
    print("\n[2] Scanning available camera devices...")
    available_cameras = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"    ✓ Camera {i}: {int(width)}x{int(height)}")
            available_cameras.append(i)
            cap.release()
        else:
            cap.release()
    
    if not available_cameras:
        print("    ✗ NO CAMERAS DETECTED!")
        return False
    
    # Test 3: Try to open default camera (index 0)
    print(f"\n[3] Testing default camera (index 0)...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("    ✗ FAILED to open camera 0")
        print("\n[SOLUTIONS]:")
        print("    1. Check if camera is plugged in")
        print("    2. Check Device Manager for camera devices")
        print("    3. Restart the computer")
        print("    4. Update camera drivers")
        print("    5. Try another camera index (see above)")
        return False
    
    print("    ✓ Camera 0 opened successfully")
    
    # Test 4: Try to capture a frame
    print(f"\n[4] Attempting to capture frame...")
    ret, frame = cap.read()
    
    if not ret:
        print("    ✗ FAILED to capture frame")
        cap.release()
        return False
    
    if frame is None:
        print("    ✗ Frame is None")
        cap.release()
        return False
    
    print(f"    ✓ Frame captured successfully")
    print(f"    Frame size: {frame.shape}")
    
    # Test 5: Capture multiple frames
    print(f"\n[5] Capturing 30 frames (test stability)...")
    for i in range(30):
        ret, frame = cap.read()
        if not ret:
            print(f"    ✗ Failed at frame {i}")
            cap.release()
            return False
        if i % 10 == 0:
            print(f"    ✓ Frame {i} captured")
    
    print("    ✓ All frames captured successfully")
    
    cap.release()
    
    print("\n" + "="*60)
    print("  ✓ ALL TESTS PASSED - CAMERA IS WORKING!")
    print("="*60)
    print("\nSekarang bisa jalankan:")
    print("  python 4_real_time_inference.py\n")
    
    return True


def troubleshoot():
    print("\n" + "="*60)
    print("  TROUBLESHOOTING GUIDE")
    print("="*60)
    
    print("\n[CHECKLIST]")
    print("  [ ] 1. Kamera fisik terpasang?")
    print("  [ ] 2. Device Manager mendeteksi kamera?")
    print("  [ ] 3. Kamera tidak digunakan app lain?")
    print("  [ ] 4. Driver kamera ter-update?")
    print("  [ ] 5. Permission sudah diberikan?")
    
    print("\n[COMMON ISSUES]")
    
    print("\n  A. Kamera tidak terdeteksi:")
    print("     - Cek Device Manager → Imaging Devices")
    print("     - Update driver kamera")
    print("     - Unplug & replug kamera")
    print("     - Restart computer")
    
    print("\n  B. Kamera terdeteksi tapi tidak bisa dibuka:")
    print("     - Close aplikasi lain yang pakai kamera (Teams, Zoom, etc)")
    print("     - Restart computer")
    print("     - Check camera permissions di Settings")
    
    print("\n  C. OpenCV tidak bisa akses kamera:")
    print("     - Reinstall opencv-python:")
    print("       pip install --upgrade opencv-python")
    print("     - Atau gunakan opencv-python-headless:")
    print("       pip install opencv-python-headless")
    
    print("\n  D. Multiple camera devices:")
    print("     - Edit script dan ubah:")
    print("       cap = cv2.VideoCapture(0)  # 0 = camera index")
    print("       menjadi:")
    print("       cap = cv2.VideoCapture(1)  # atau 2, 3, dst")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    print("\n  CAMERA DIAGNOSTIC TOOL\n")
    
    if not test_camera():
        print("\n✗ Camera test FAILED\n")
        troubleshoot()
        sys.exit(1)
