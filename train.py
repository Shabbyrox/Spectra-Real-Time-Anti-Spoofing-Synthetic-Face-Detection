import os
import glob
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from main import extract_face_and_spectrum, roi_to_fft

SEED = 42
VAL_FRACTION = 0.2
EPOCHS = 20
BATCH_SIZE = 16
LR = 1e-4
WEIGHT_DECAY = 1e-4

# ImageNet stats — the RGB stream uses a pretrained (ImageNet) backbone.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# 1. Dataset: caches BOTH the RGB face crop and its FFT spectrum, once,
#    keeping only samples where a face was actually detected.
class DirectRawDataset(Dataset):
    def __init__(self, raw_dir="raw_data"):
        self.rgb = []       # cached (H, W, 3) uint8 BGR face crops
        self.fft = []       # cached (H, W, 3) uint8 FFT spectra
        self.labels = []

        sources = [
            (os.path.join(raw_dir, "real"), 0),   # Real / Authentic = 0
            (os.path.join(raw_dir, "fake"), 1),   # Fake / Spoof     = 1
        ]

        skipped = 0
        for folder, label in sources:
            if not os.path.exists(folder):
                continue
            paths = []
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"):
                paths += glob.glob(os.path.join(folder, ext))

            for p in paths:
                frame = cv2.imread(p)
                if frame is None:
                    skipped += 1
                    continue
                roi, spectrum, face_box = extract_face_and_spectrum(frame)
                if face_box is None:
                    skipped += 1     # no face -> background noise; skip
                    continue
                self.rgb.append(roi)
                self.fft.append(spectrum)
                self.labels.append(label)

        n_real = self.labels.count(0)
        n_fake = self.labels.count(1)
        print(f"[*] Usable (face-detected) samples: {len(self.labels)} "
              f"({n_real} Real, {n_fake} Fake) | Skipped (no face / unreadable): {skipped}")

    def __len__(self):
        return len(self.labels)


# Wraps a subset of DirectRawDataset and applies (optional) augmentation.
# For augmented (training) samples the FFT is RECOMPUTED from the augmented
# face crop, so any geometric/photometric transform stays perfectly consistent
# across both streams. Validation uses the untouched cached tensors.
class StreamView(Dataset):
    def __init__(self, base, indices, augment):
        self.base = base
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    @staticmethod
    def _augment_roi(roi):
        """Geometric + photometric augmentation on a uint8 BGR face crop."""
        h, w = roi.shape[:2]

        if random.random() < 0.5:                       # horizontal flip
            roi = roi[:, ::-1, :]

        if random.random() < 0.7:                       # small rotation + scale
            angle = random.uniform(-12, 12)
            scale = random.uniform(0.9, 1.1)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
            roi = cv2.warpAffine(roi, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

        if random.random() < 0.7:                       # brightness / contrast
            alpha = random.uniform(0.85, 1.15)          # contrast
            beta = random.uniform(-12, 12)              # brightness
            roi = cv2.convertScaleAbs(roi, alpha=alpha, beta=beta)

        return np.ascontiguousarray(roi)

    def __getitem__(self, i):
        idx = self.indices[i]

        if self.augment:
            roi = self._augment_roi(self.base.rgb[idx])
            fft_u8 = roi_to_fft(roi)                     # consistent w/ augmented RGB
        else:
            roi = self.base.rgb[idx]
            fft_u8 = self.base.fft[idx]

        # RGB stream: BGR -> RGB, [0,1], ImageNet-normalize for the backbone.
        rgb = np.ascontiguousarray(roi[:, :, ::-1])
        rgb = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD

        fft = torch.from_numpy(np.ascontiguousarray(fft_u8)).float().permute(2, 0, 1) / 255.0
        return rgb, fft, torch.tensor(self.base.labels[idx], dtype=torch.long)


# 2. Two-stream network:
#    - RGB stream : pretrained ResNet18 backbone (transfer learning)
#    - FFT stream : small CNN over the 2D-FFT spectrum
#    Features are concatenated and classified jointly.
class SpectraNet(nn.Module):
    def __init__(self, pretrained=True):
        super(SpectraNet, self).__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = torchvision.models.resnet18(weights=weights)
        backbone.fc = nn.Identity()          # -> 512-dim RGB feature
        # Freeze the early (generic) layers to curb overfitting on a small set;
        # only fine-tune the deeper, more task-specific blocks.
        if pretrained:
            for name, param in backbone.named_parameters():
                if name.startswith(("conv1", "bn1", "layer1", "layer2")):
                    param.requires_grad = False
        self.rgb_stream = backbone

        self.fft_stream = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 128), nn.ReLU(),   # -> 128-dim FFT feature
        )

        self.head = nn.Sequential(
            nn.Linear(512 + 128, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 2),
        )

    def forward(self, rgb, fft):
        r = self.rgb_stream(rgb)
        f = self.fft_stream(fft)
        return self.head(torch.cat([r, f], dim=1))


def stratified_split(labels, val_fraction, seed):
    """Return (train_idx, val_idx) keeping the real/fake ratio in both splits."""
    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for cls in sorted(set(labels)):
        idxs = [i for i, l in enumerate(labels) if l == cls]
        rng.shuffle(idxs)
        n_val = int(round(len(idxs) * val_fraction))
        val_idx += idxs[:n_val]
        train_idx += idxs[n_val:]
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    confusion = [[0, 0], [0, 0]]      # confusion[true][pred]; 0=Real, 1=Fake
    correct = total = 0
    for rgb, fft, labels in loader:
        rgb, fft, labels = rgb.to(device), fft.to(device), labels.to(device)
        preds = model(rgb, fft).argmax(dim=1)
        for t, p in zip(labels.tolist(), preds.tolist()):
            confusion[t][p] += 1
            correct += int(t == p)
            total += 1
    return (correct / total if total else 0.0), confusion


def train_model():
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    dataset = DirectRawDataset(raw_dir="raw_data")
    if len(dataset) == 0:
        print("[!] No usable images found in 'raw_data/real' or 'raw_data/fake'.")
        return

    train_idx, val_idx = stratified_split(dataset.labels, VAL_FRACTION, SEED)
    train_set = StreamView(dataset, train_idx, augment=True)
    val_set = StreamView(dataset, val_idx, augment=False)
    print(f"[*] Split -> train: {len(train_set)}  val: {len(val_set)}")

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[*] Training on device: {device}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    model = SpectraNet(pretrained=True).to(device)

    counts = np.bincount([dataset.labels[i] for i in train_idx], minlength=2)
    weights = torch.tensor((counts.sum() / (2 * np.maximum(counts, 1))),
                            dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    print("\nTraining Spectra two-stream net (ResNet18 RGB + FFT-CNN)...")
    best_val_acc = -1.0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for rgb, fft, labels in train_loader:
            rgb, fft, labels = rgb.to(device), fft.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(rgb, fft), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        val_acc, _ = evaluate(model, val_loader, device)
        flag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "spectra_cnn.pth")
            flag = "  <-- best (saved)"
        print(f"Epoch {epoch + 1:2d}/{EPOCHS} - "
              f"Loss: {running_loss / len(train_loader):.4f} - "
              f"Val Acc: {val_acc * 100:.2f}%{flag}")

    model.load_state_dict(torch.load("spectra_cnn.pth", map_location=device))
    val_acc, cm = evaluate(model, val_loader, device)
    print("\n================ VALIDATION REPORT (best model) ================")
    print(f"Accuracy: {val_acc * 100:.2f}%")
    print("Confusion matrix (rows=true, cols=pred):")
    print(f"                 pred:Real   pred:Fake")
    print(f"  true:Real       {cm[0][0]:>6}      {cm[0][1]:>6}")
    print(f"  true:Fake       {cm[1][0]:>6}      {cm[1][1]:>6}")
    tp, fn, fp = cm[1][1], cm[1][0], cm[0][1]
    recall_spoof = tp / (tp + fn) if (tp + fn) else 0.0
    precision_spoof = tp / (tp + fp) if (tp + fp) else 0.0
    print(f"  Spoof recall (caught):    {recall_spoof * 100:.2f}%")
    print(f"  Spoof precision:          {precision_spoof * 100:.2f}%")
    print("===============================================================")
    print("\n[+] Best model saved as 'spectra_cnn.pth'!")


if __name__ == "__main__":
    train_model()
