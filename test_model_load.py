"""
Quick test: Verify model loads correctly
"""
import sys
sys.path.insert(0, 'd:\\Machine Learning UAS')

print("[TEST] Importing libraries...")
import torch
import torch.nn as nn
import json
from pathlib import Path
import torchvision.transforms as transforms
import timm

print("[TEST] ✓ Libraries imported")

# Load mapping
print("[TEST] Loading mapping...")
mapping = json.load(open("d:\\Machine Learning UAS\\dataset\\label_mapping.json"))
print(f"[TEST] ✓ Mapping loaded: {len(mapping['lvl1_classes'])} Level 1 classes")

# Load model
print("[TEST] Loading model...")
class HierarchicalGarbageNet(nn.Module):
    def __init__(self, num_lvl1=3, num_lvl2=13):
        super().__init__()
        base = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
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
        feat = self.backbone(x)
        shared = self.shared_fc(feat)
        out1 = self.head_lvl1(shared)
        out2 = self.head_lvl2(shared)
        return out1, out2

model = HierarchicalGarbageNet()
model_path = "d:\\Machine Learning UAS\\model\\best_model.pth"
model.load_state_dict(torch.load(model_path, map_location="cpu"))
print(f"[TEST] ✓ Model loaded from {model_path}")

print("\n✓ ALL TESTS PASSED!")
print("\nSekarang bisa jalankan: python 4_real_time_inference.py")
print("dengan kamera yang tersambung ke komputer.")
