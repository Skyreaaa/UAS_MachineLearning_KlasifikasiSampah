"""
====================================================
  SCRIPT 3: MODEL TRAINING – HIERARKIS 2 HEAD
  ─────────────────────────────────────────────
  Backbone : EfficientNet-B0 (ImageNet pretrained, frozen lalu di-unfreeze)
  Head 1   : Level 1 → 3 kelas  (Organik / Anorganik / B3)
  Head 2   : Level 2 → 12 kelas (subkategori per level 1)

  Loss = α × CE(lvl1) + (1-α) × CE(lvl2) + 0.1 × Consistency
  Consistency memastikan prediksi level 2 konsisten dengan level 1.

  Alur training:
    Epoch 1–9  : Hanya head + shared_fc yang dilatih (backbone frozen)
    Epoch 10+  : Backbone di-unfreeze, fine-tuning seluruh jaringan dengan lr lebih kecil
    Checkpoint : best_model.pth (terbaik val_acc_lvl1) + resume_checkpoint.pth (per epoch)
====================================================
Install:
  pip install torch torchvision timm scikit-learn matplotlib seaborn tqdm
"""

import sys, json, time
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm


# ─────────────────────────────────────────────────
#  KONFIGURASI
# ─────────────────────────────────────────────────
DATA_DIR   = Path("dataset/processed")   # output dari 2_preprocessor.py
MODEL_DIR  = Path("model")               # tempat menyimpan model & artefak
MAPPING_F  = Path("dataset/label_mapping.json")

IMG_SIZE   = 224      # resolusi input model (224×224 px)
BATCH_SIZE = 32
NUM_EPOCHS = 30
LR         = 1e-4     # learning rate awal (head only)
LR_MIN     = 1e-6     # lr minimum untuk CosineAnnealing
PATIENCE   = 6        # early stopping: berhenti jika val_loss tidak membaik N epoch berturut
ALPHA      = 0.4      # bobot loss level 1; (1-ALPHA) untuk level 2
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR.mkdir(exist_ok=True)
print(f"  Device : {DEVICE}")


# ─────────────────────────────────────────────────
#  1. LABEL MAPPING
# ─────────────────────────────────────────────────
def load_mapping():
    """Baca label_mapping.json yang dibuat oleh 2_preprocessor.py."""
    with open(MAPPING_F) as f:
        m = json.load(f)
    print(f"  Level 1: {m['lvl1_classes']}")
    print(f"  Level 2: {m['lvl2_classes']}")
    return m

mapping  = load_mapping()
NUM_LVL1 = len(mapping["lvl1_classes"])
NUM_LVL2 = len(mapping["lvl2_classes"])


# ─────────────────────────────────────────────────
#  2. DATASET & AUGMENTASI
# ─────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Augmentasi hanya untuk split train; val/test hanya resize + normalize
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class HierarchicalDataset(Dataset):
    """
    Dataset untuk klasifikasi hierarkis 2 level.

    Struktur folder yang diharapkan:
        dataset/processed/{split}/{lvl1}/{lvl2}/*.jpg

    Setiap sampel menghasilkan (image_tensor, lvl1_label_idx, lvl2_label_idx).
    """

    def __init__(self, split: str, transform=None):
        self.transform   = transform
        self.samples     = []
        self.lvl1_map    = mapping["lvl1_map"]
        self.lvl2_map    = mapping["lvl2_map"]

        root = DATA_DIR / split
        for lvl1_dir in sorted(root.iterdir()):
            if not lvl1_dir.is_dir():
                continue
            lvl1_idx = self.lvl1_map.get(lvl1_dir.name, -1)
            if lvl1_idx == -1:
                continue
            for lvl2_dir in sorted(lvl1_dir.iterdir()):
                if not lvl2_dir.is_dir():
                    continue
                lvl2_idx = self.lvl2_map.get(lvl2_dir.name, -1)
                if lvl2_idx == -1:
                    continue
                for img_path in lvl2_dir.glob("*.jpg"):
                    self.samples.append((img_path, lvl1_idx, lvl2_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, lvl1_idx, lvl2_idx = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, lvl1_idx, lvl2_idx


def get_loaders():
    """Buat DataLoader untuk train, val, dan test split."""
    train_ds = HierarchicalDataset("train", train_tf)
    val_ds   = HierarchicalDataset("val",   val_tf)
    test_ds  = HierarchicalDataset("test",  val_tf)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    print(f"\n  Train  : {len(train_ds)} sampel ({len(train_loader)} batch)")
    print(f"  Val    : {len(val_ds)} sampel")
    print(f"  Test   : {len(test_ds)} sampel")
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────
#  3. ARSITEKTUR MODEL
# ─────────────────────────────────────────────────
class HierarchicalGarbageNet(nn.Module):
    """
    Model klasifikasi sampah hierarkis dengan 2 head output.

    Arsitektur:
        EfficientNet-B0 (backbone, ImageNet pretrained)
            └─► shared_fc  : Linear(1280→512) + BN + ReLU + Dropout(0.4)
                ├─► head_lvl1 : Linear(512→128) → ReLU → Dropout → Linear(128→3)
                └─► head_lvl2 : Linear(512→256) → ReLU → Dropout → Linear(256→12)

    Backbone awalnya di-freeze; di-unfreeze setelah epoch ke-10 dengan lr lebih kecil.
    """

    def __init__(self, backbone_name="efficientnet_b0",
                 num_lvl1=NUM_LVL1, num_lvl2=NUM_LVL2):
        super().__init__()

        base     = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        feat_dim = base.num_features   # 1280 untuk EfficientNet-B0

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

        # Freeze backbone pada awalnya; hanya head yang dilatih dulu
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        feat   = self.backbone(x)
        shared = self.shared_fc(feat)
        return self.head_lvl1(shared), self.head_lvl2(shared)

    def unfreeze_backbone(self, lr_backbone=1e-5):
        """Aktifkan gradient untuk seluruh backbone (fine-tuning phase)."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print(f"  ✓ Backbone di-unfreeze (lr_backbone={lr_backbone})")


def build_model():
    """Inisialisasi model dan print ringkasan parameter."""
    model     = HierarchicalGarbageNet()
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model      : HierarchicalGarbageNet (EfficientNet-B0)")
    print(f"  Total params   : {total:,}")
    print(f"  Trainable      : {trainable:,} ({100*trainable/total:.1f}%)")
    return model.to(DEVICE)


# ─────────────────────────────────────────────────
#  4. LOSS HIERARKIS
# ─────────────────────────────────────────────────
class HierarchicalLoss(nn.Module):
    """
    Loss gabungan untuk klasifikasi hierarkis.

    Formula:
        total = α × CE(lvl1) + (1-α) × CE(lvl2) + 0.1 × Consistency

    Consistency loss memproyeksikan probabilitas level 2 ke ruang level 1
    (via matriks mapping lvl2→lvl1) lalu menghitung CE terhadap label level 1.
    Ini memaksa prediksi level 2 konsisten dengan prediksi level 1.
    """

    def __init__(self, alpha=ALPHA, use_consistency=True):
        super().__init__()
        self.alpha           = alpha
        self.use_consistency = use_consistency
        self.ce              = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Buat matriks proyeksi lvl2 → lvl1  (shape: NUM_LVL2 × NUM_LVL1)
        lvl2_to_lvl1 = torch.zeros(NUM_LVL2, NUM_LVL1)
        for lvl2_name, lvl2_idx in mapping["lvl2_map"].items():
            lvl1_name = mapping["sub_to_lvl1"][lvl2_name]
            lvl1_idx  = mapping["lvl1_map"][lvl1_name]
            lvl2_to_lvl1[lvl2_idx, lvl1_idx] = 1.0
        self.register_buffer("lvl2_to_lvl1", lvl2_to_lvl1)

    def forward(self, out1, out2, label1, label2):
        loss1 = self.ce(out1, label1)
        loss2 = self.ce(out2, label2)

        consistency = 0.0
        if self.use_consistency:
            prob2     = torch.softmax(out2, dim=1)
            proj_lvl1 = prob2 @ self.lvl2_to_lvl1
            consistency = nn.functional.cross_entropy(
                torch.log(proj_lvl1 + 1e-8), label1
            )

        total = self.alpha * loss1 + (1 - self.alpha) * loss2 + 0.1 * consistency
        return total, loss1.item(), loss2.item()


# ─────────────────────────────────────────────────
#  5. TRAINING LOOP
# ─────────────────────────────────────────────────
class EarlyStopping:
    """Hentikan training jika val_loss tidak membaik selama `patience` epoch."""

    def __init__(self, patience=6, min_delta=0.001):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_loss = float("inf")

    def step(self, val_loss) -> bool:
        """Return True jika training boleh lanjut, False jika harus berhenti."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
        return self.counter < self.patience


def train_epoch(model, loader, criterion, optimizer):
    """Satu epoch training; return (avg_loss, acc_lvl1, acc_lvl2)."""
    model.train()
    tot_loss = tot_correct1 = tot_correct2 = n = 0

    for imgs, lbl1, lbl2 in tqdm(loader, desc="  Train", leave=False):
        imgs, lbl1, lbl2 = imgs.to(DEVICE), lbl1.to(DEVICE), lbl2.to(DEVICE)
        optimizer.zero_grad()

        out1, out2   = model(imgs)
        loss, _, _   = criterion(out1, out2, lbl1, lbl2)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        tot_loss     += loss.item() * imgs.size(0)
        tot_correct1 += (out1.argmax(1) == lbl1).sum().item()
        tot_correct2 += (out2.argmax(1) == lbl2).sum().item()
        n += imgs.size(0)

    return tot_loss / n, tot_correct1 / n, tot_correct2 / n


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    """Satu epoch evaluasi tanpa gradient; return (avg_loss, acc_lvl1, acc_lvl2)."""
    model.eval()
    tot_loss = tot_correct1 = tot_correct2 = n = 0

    for imgs, lbl1, lbl2 in tqdm(loader, desc="  Eval ", leave=False):
        imgs, lbl1, lbl2 = imgs.to(DEVICE), lbl1.to(DEVICE), lbl2.to(DEVICE)
        out1, out2       = model(imgs)
        loss, _, _       = criterion(out1, out2, lbl1, lbl2)

        tot_loss     += loss.item() * imgs.size(0)
        tot_correct1 += (out1.argmax(1) == lbl1).sum().item()
        tot_correct2 += (out2.argmax(1) == lbl2).sum().item()
        n += imgs.size(0)

    return tot_loss / n, tot_correct1 / n, tot_correct2 / n


def train(model, train_loader, val_loader):
    """
    Loop training utama dengan resume support, unfreeze scheduler, dan early stopping.

    Resume logic:
        1. Cek resume_checkpoint.pth → lanjut dari epoch terakhir
        2. Cek best_model.pth saja   → mulai ulang dengan backbone unfrozen
        3. Tidak ada checkpoint      → mulai dari awal (backbone frozen)

    Setiap epoch menyimpan resume_checkpoint.pth.
    best_model.pth hanya diperbarui jika val_acc_lvl1 meningkat.
    """
    resume_path     = MODEL_DIR / "resume_checkpoint.pth"
    checkpoint_path = MODEL_DIR / "best_model.pth"

    criterion     = HierarchicalLoss(alpha=ALPHA)
    start_epoch   = 1
    best_acc1     = 0.0
    unfreeze_done = False
    history       = {k: [] for k in
                     ["train_loss", "val_loss", "train_acc1", "val_acc1", "train_acc2", "val_acc2"]}

    if resume_path.exists():
        ckpt          = torch.load(resume_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        start_epoch   = ckpt["epoch"] + 1
        best_acc1     = ckpt["best_acc1"]
        unfreeze_done = ckpt.get("unfreeze_done", False)
        history       = ckpt.get("history", history)
        print(f"\n  ✓ Resume dari epoch {ckpt['epoch']} | best_acc1={best_acc1:.4f}")
    elif checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        unfreeze_done = True
        model.unfreeze_backbone()
        print(f"\n  ✓ Loaded best_model.pth, mulai ulang dengan backbone unfrozen")

    init_lr   = 5e-5 if unfreeze_done else LR
    optimizer = optim.AdamW(
        model.parameters() if unfreeze_done
        else filter(lambda p: p.requires_grad, model.parameters()),
        lr=init_lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=LR_MIN, last_epoch=start_epoch - 2
    )
    es        = EarlyStopping(patience=PATIENCE)

    remaining = NUM_EPOCHS - start_epoch + 1
    print(f"\n{'='*60}")
    print(f"  TRAINING  epoch {start_epoch}–{NUM_EPOCHS} ({remaining} sisa) | batch={BATCH_SIZE}")
    print(f"  LR={init_lr} | Loss = {ALPHA}×CE(lvl1) + {1-ALPHA}×CE(lvl2) + 0.1×Consistency")
    print(f"{'='*60}")

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        # Unfreeze backbone mulai epoch 10
        if epoch >= 10 and not unfreeze_done:
            model.unfreeze_backbone()
            optimizer     = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
            unfreeze_done = True

        t0              = time.time()
        tl, ta1, ta2    = train_epoch(model, train_loader, criterion, optimizer)
        vl, va1, va2    = eval_epoch(model, val_loader,   criterion)
        scheduler.step()

        print(f"  Ep {epoch:02d}/{NUM_EPOCHS} | "
              f"Loss {tl:.4f}/{vl:.4f} | "
              f"Acc-L1 {ta1:.3f}/{va1:.3f} | "
              f"Acc-L2 {ta2:.3f}/{va2:.3f} | "
              f"{time.time()-t0:.1f}s")

        for k, v in zip(history.keys(), [tl, vl, ta1, va1, ta2, va2]):
            history[k].append(v)

        if va1 > best_acc1:
            best_acc1 = va1
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pth")
            print(f"  ✓ Model terbaik disimpan (val_acc_lvl1={best_acc1:.4f})")

        torch.save({
            "epoch"        : epoch,
            "model"        : model.state_dict(),
            "best_acc1"    : best_acc1,
            "unfreeze_done": unfreeze_done,
            "history"      : history,
        }, MODEL_DIR / "resume_checkpoint.pth")

        if not es.step(vl):
            print(f"\n  ⚠ Early stopping di epoch {epoch}")
            break

    model.load_state_dict(torch.load(MODEL_DIR / "best_model.pth", map_location=DEVICE))
    print(f"\n  ✓ Training selesai! Best val_acc_lvl1 = {best_acc1:.4f}")
    return history


# ─────────────────────────────────────────────────
#  6. EVALUASI & VISUALISASI
# ─────────────────────────────────────────────────
@torch.no_grad()
def evaluate_test(model, test_loader):
    """
    Evaluasi model pada test set; cetak classification report dan simpan confusion matrix.
    Output: model/confusion_matrix.png
    """
    print("\n[EVALUASI] Test set...")
    model.eval()
    preds1, preds2, lbls1, lbls2 = [], [], [], []

    for imgs, lbl1, lbl2 in tqdm(test_loader, desc="  Testing"):
        imgs   = imgs.to(DEVICE)
        o1, o2 = model(imgs)
        preds1.extend(o1.argmax(1).cpu().numpy())
        preds2.extend(o2.argmax(1).cpu().numpy())
        lbls1.extend(lbl1.numpy())
        lbls2.extend(lbl2.numpy())

    print("\n  ── LEVEL 1 (Organik / Anorganik / B3) ──")
    print(classification_report(lbls1, preds1, target_names=mapping["lvl1_classes"]))

    print("  ── LEVEL 2 (Subkategori) ──")
    print(classification_report(lbls2, preds2, target_names=mapping["lvl2_classes"]))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.heatmap(confusion_matrix(lbls1, preds1), annot=True, fmt="d", cmap="Greens",
                xticklabels=mapping["lvl1_classes"],
                yticklabels=mapping["lvl1_classes"], ax=axes[0])
    axes[0].set_title("Confusion Matrix – Level 1")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    sns.heatmap(confusion_matrix(lbls2, preds2), annot=True, fmt="d", cmap="Blues",
                xticklabels=mapping["lvl2_classes"],
                yticklabels=mapping["lvl2_classes"], ax=axes[1])
    axes[1].set_title("Confusion Matrix – Level 2")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig(MODEL_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print("  ✓ Confusion matrix disimpan")


def plot_history(history):
    """Plot loss dan accuracy per epoch, simpan ke model/training_history.png."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs    = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], "b-o", ms=4, label="Train")
    axes[0].plot(epochs, history["val_loss"],   "r-o", ms=4, label="Val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [a*100 for a in history["train_acc1"]], "b-o", ms=4, label="Train")
    axes[1].plot(epochs, [a*100 for a in history["val_acc1"]],   "r-o", ms=4, label="Val")
    axes[1].set_title("Accuracy Level 1 (Org/Anorg/B3)")
    axes[1].set_ylabel("%")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, [a*100 for a in history["train_acc2"]], "b-o", ms=4, label="Train")
    axes[2].plot(epochs, [a*100 for a in history["val_acc2"]],   "r-o", ms=4, label="Val")
    axes[2].set_title("Accuracy Level 2 (Subkategori)")
    axes[2].set_ylabel("%")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.suptitle("Training History – HierarchicalGarbageNet", fontsize=13)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "training_history.png", dpi=150)
    plt.close()
    print("  ✓ Training history disimpan")


# ─────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  HIERARCHICAL GARBAGE CLASSIFICATION – TRAINING")
    print("=" * 60)

    train_loader, val_loader, test_loader = get_loaders()
    model   = build_model()
    history = train(model, train_loader, val_loader)
    plot_history(history)
    evaluate_test(model, test_loader)

    print(f"\n  ✓ Model tersimpan di : {MODEL_DIR}/best_model.pth")
    print(f"  ✓ Label mapping      : dataset/label_mapping.json")
    print(f"  ✓ Confusion matrix   : {MODEL_DIR}/confusion_matrix.png")
    print(f"  ✓ Training history   : {MODEL_DIR}/training_history.png")
