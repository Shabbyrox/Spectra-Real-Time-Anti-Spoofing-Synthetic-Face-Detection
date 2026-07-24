# Spectra — Real-Time Anti-Spoofing & Synthetic-Face Detection

Real-time face **anti-spoofing** & **synthetic-face detection** using a two-stream CNN over **RGB + 2D-FFT spectra**.

Spectra decides whether a face in front of the camera is a **live person** or a **spoof** — a photo, a phone/screen replay, or an AI-generated face. Instead of looking at the picture alone, it also reads the face's **frequency signature** (a 2D Fast Fourier Transform), where screens, prints, and GANs leave tell-tale artifacts that a spatial view hides. Hence the name — *Spectra*, from the frequency **spectrum** it analyzes.

---

## How it works

```
 Webcam / image
      │
      ▼
┌─────────────────────┐   YuNet (DNN)  →  Haar (fallback)  →  center-crop (untrusted)
│  Face detection     │   picks the largest face; no face ⇒ refuses to predict
└─────────────────────┘
      │  face ROI, resized 224×224 (BGR)
      ▼
┌───────────────────────── SpectraNet (two-stream) ─────────────────────────┐
│                                                                            │
│   RGB face crop ──▶ ResNet18 (ImageNet, early layers frozen) ──▶ 512-d     │
│                                                                  ╲         │
│                                                                   ▶ concat 640 ─▶ head ─▶ [Real | Fake]
│                                                                  ╱         │
│   2D-FFT spectrum ─▶ FFT-CNN (2 conv blocks) ─────────────────▶ 128-d      │
│                                                                            │
└────────────────────────────────────────────────────────────────────────┘
      │  softmax → temporal smoothing (last N frames)
      ▼
  AUTHENTIC  /  SPOOF  /  UNCERTAIN  /  NO FACE
```

- **Face detection** — [YuNet](https://github.com/opencv/opencv_zoo) (a small ONNX DNN, robust on phone screens and tilted faces), with a classic Haar cascade as a fallback. When no real face is found, Spectra shows **NO FACE** instead of guessing on a center crop.
- **Two representations, two encoders** — the face crop feeds a **pretrained ResNet18** (spatial appearance), and its **2D-FFT magnitude spectrum** feeds a small CNN (frequency signature). Their features are concatenated and classified together.
- **Stable verdicts** — probabilities are averaged over recent frames (temporal smoothing) and low-confidence predictions are reported as **UNCERTAIN** rather than forced into a label.

---

## Technical details

Spectra is **primarily neural-network based** — two CNNs do the work — combined with one classical detector fallback and one deterministic signal-processing transform.

### Components

| Component | Type | Neural net? | Trained here? | ~Params | Role |
|---|---|:---:|:---:|---|---|
| **YuNet** | CNN (ONNX) | ✅ | No — pretrained, frozen | ~85K | Face detection |
| **Haar cascade** | Viola–Jones (AdaBoost) | ❌ | No | — | Fallback detector |
| **2D-FFT** | Signal processing (DSP) | ❌ | Not learned | 0 | RGB → frequency spectrum |
| **SpectraNet** | Deep CNN (two-stream) | ✅ | **Yes** | ~11.4M | Classifier (live vs spoof) |

### SpectraNet architecture

A two-stream network that reads the face two ways and fuses them:

- **Stream A — appearance (`ResNet18`).** An 18-layer residual CNN from torchvision, pretrained on ImageNet with the final classifier removed → a **512-d** feature. Early layers (`conv1`, `bn1`, `layer1`, `layer2`) are **frozen**; deeper layers are fine-tuned (transfer learning), which is what makes high accuracy possible from a small dataset.
- **Stream B — frequency (`FFT-CNN`).** A small CNN trained from scratch (~70K params) over the 2D-FFT spectrum:
  `Conv(3→16)+BN+ReLU+Pool → Conv(16→32)+BN+ReLU+Pool → AdaptiveAvgPool(4×4) → Flatten → Linear(512→128)` → a **128-d** feature.
- **Fusion head.** `concat[512 + 128 = 640] → Linear(640→128) → ReLU → Dropout(0.4) → Linear(128→2)` → 2 logits → softmax → `P(real)`, `P(fake)`.

### The 2D-FFT feature

Per color channel of the face crop:

```
F      = FFT2(channel)                        # 2D Fast Fourier Transform
Fshift = fftshift(F)                          # low frequencies to the center
mag    = 20 · log(|Fshift| + ε)               # log-magnitude spectrum
spec   = clip((mag − 70) / (215 − 70)) · 255  # fixed-window normalize → 0..255
```

A face shown on a **screen** produces *moiré* (interference between the display's pixel grid and the camera sensor grid); prints add paper texture; GANs leave periodic artifacts. These are nearly invisible in pixel space but appear as distinct structure in the frequency domain — the signal Stream B learns from. A **fixed** normalization window (not per-image min/max) preserves absolute magnitude, and the ROI is resized to 224×224 **before** the FFT so every sample shares one frequency scale.

### Training configuration

| Setting | Value |
|---|---|
| Loss | Cross-entropy, class-weighted |
| Optimizer | AdamW · lr 1e-4 · weight-decay 1e-4 |
| LR schedule | StepLR (×0.5 every 8 epochs) |
| Epochs / batch | 20 / 16 |
| Augmentation | flip, rotation ±12°, scale 0.9–1.1, brightness/contrast — applied to the crop, then the FFT is **recomputed** so both streams stay consistent |
| RGB normalization | ImageNet mean/std |
| Split / selection | Stratified 80/20; best **validation-accuracy** checkpoint saved |
| Device | Apple MPS / CUDA / CPU (auto) |

### Inference (per frame)

1. Detect the largest face (YuNet → Haar → refuse if none).
2. Crop & resize to 224×224; build both the RGB and FFT views.
3. SpectraNet forward pass → softmax probabilities.
4. **Temporal smoothing** — average probabilities over the last 4 frames.
5. **Thresholding** — below 60% confidence → `UNCERTAIN`; no face → `NO FACE`; else `AUTHENTIC` / `SPOOF`.

### Design rationale

- **Two streams beat one** — RGB alone plateaued (~72% on the harder task); the FFT stream adds a physical spoof signal that isn't recoverable from pixels. Spatial + frequency = complementary evidence.
- **Transfer learning does the heavy lifting** — a from-scratch small CNN capped near random on the hard task; a pretrained ResNet18 backbone is what makes small-dataset accuracy possible.
- **Classical pieces where learning isn't needed** — the FFT is exact math; Haar is a zero-dependency safety net if the DNN detector is unavailable.

---

## Results

Trained on **LCC-FASD** (real faces vs. replay/print presentation attacks):

| Metric | Value |
|---|---|
| Validation accuracy | **~99%** (held-out split) |
| Spoof recall (attacks caught) | ~98.6% |
| Spoof precision | ~99.8% |

> **Note on real-world use:** ~99% is measured on the dataset's own cameras and conditions. On *your* webcam and phone, expect it to work well but somewhat lower (a domain gap). See [Improving accuracy on your hardware](#improving-accuracy-on-your-hardware).

---

## Project structure

| File | Role |
|---|---|
| `main.py` | Shared engine — face detection (`detect_face_roi`) and the 2D-FFT transform (`roi_to_fft`). Run directly for a live detect + FFT preview. |
| `train.py` | Defines **SpectraNet** (two-stream) and trains it; prints validation accuracy + confusion matrix and saves the best checkpoint. |
| `infer.py` | Real-time webcam inference with the live verdict banner. |
| `collector.py` | Capture your own webcam samples (`a` = live, `s` = spoof) straight into the training folders. |
| `face_detection_yunet_2023mar.onnx` | YuNet detector weights (kept in-repo so it runs after a fresh clone). |
| `spectra_cnn.pth` | Trained model weights (git-ignored; regenerate with `train.py`). |

Data lives in `raw_data/real` (live / label 0) and `raw_data/fake` (spoof / label 1).

---

## Setup

Requires **Python 3.9+**.

```bash
git clone <your-repo-url>
cd Spectra

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> **Important:** Spectra pins **`opencv-python<5`**. OpenCV 5 removed `cv2.CascadeClassifier` and ships an empty Haar-cascade folder, which silently breaks face detection. Stay on OpenCV 4.x.

Runs on Apple **MPS**, **CUDA**, or **CPU** automatically.

---

## Usage

### 1. Run it live
```bash
python infer.py        # press 'q' to quit
```
Left pane = your camera with the verdict banner; right pane = the live 2D-FFT spectrum.

### 2. Train your own model
Put labeled images in `raw_data/real` (live) and `raw_data/fake` (spoof), then:
```bash
python train.py        # saves the best checkpoint to spectra_cnn.pth
```

### 3. Collect your own samples (optional)
```bash
python collector.py
#   'a' → save a LIVE face frame        → raw_data/real
#   's' → save a SPOOF (phone/photo)    → raw_data/fake
#   'q' → quit
```
Only frames with a detected face (green box) are saved.

---

## Dataset

The reference model is trained on **[LCC-FASD](https://www.kaggle.com/datasets/faber24/lcc-fasd)** (Large Crowd-collected Face Anti-Spoofing Dataset) — real live faces vs. print and screen-replay attacks. `train.py` reads images directly from `raw_data/real` and `raw_data/fake`, computes the FFT on the fly, and keeps only samples where a face is detected.

`train.py` caches all samples in memory, so keep the set to a few thousand per class rather than tens of thousands.

---

## Improving accuracy on your hardware

Frequency/replay cues are specific to the **capture device**. If Spectra struggles with a different phone or lighting:

1. Collect ~50–100 of your own frames per class with `collector.py`.
2. Re-run `python train.py`.

Mixing in your own webcam samples adapts the model to your exact camera — the single most effective real-world upgrade.

---

## Notes

- **Anti-spoofing vs. synthetic-face detection** are two related but distinct tasks. The shipped model does **anti-spoofing** (live vs. physical replay/print). The **synthetic-face** capability (detecting GAN-generated faces) uses the same two-stream architecture trained on a synthetic-vs-real dataset — swap the training data in `raw_data/` and retrain.

## Acknowledgements

- [YuNet](https://github.com/opencv/opencv_zoo) face detector · [LCC-FASD](https://www.kaggle.com/datasets/faber24/lcc-fasd) dataset · [torchvision](https://pytorch.org/vision/) ResNet18 (ImageNet).
