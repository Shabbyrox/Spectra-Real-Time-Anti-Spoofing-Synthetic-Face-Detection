import os
import cv2
import numpy as np

# --- Fixed FFT log-magnitude normalization window ---
# Measured empirically across the dataset (p0.5 ~ 77, p99.5 ~ 213).
# Using a FIXED window (instead of per-image MIN-MAX) preserves the
# absolute magnitude differences that distinguish real vs. spoof.
FFT_VMIN = 70.0
FFT_VMAX = 215.0

# --- Primary detector: YuNet (DNN) — far more robust than Haar on screens /
#     phone displays / tilted faces. Falls back to Haar, then center crop. ---
YUNET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "face_detection_yunet_2023mar.onnx")
_yunet = None
if os.path.isfile(YUNET_PATH) and hasattr(cv2, "FaceDetectorYN"):
    try:
        _yunet = cv2.FaceDetectorYN.create(YUNET_PATH, "", (320, 320),
                                           score_threshold=0.6, nms_threshold=0.3,
                                           top_k=5000)
    except Exception:
        _yunet = None
if _yunet is None:
    print("[!] YuNet detector unavailable — using Haar cascade fallback. "
          "Download face_detection_yunet_2023mar.onnx for better detection.")

# Load Haar Cascade (fallback) safely
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml" if hasattr(cv2, 'data') else "haarcascade_frontalface_default.xml"
try:
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        face_cascade = None
except Exception:
    face_cascade = None


def _detect_boxes(frame):
    """Return a list of (x, y, w, h) face boxes using YuNet, else Haar."""
    if _yunet is not None:
        h, w = frame.shape[:2]
        _yunet.setInputSize((w, h))
        _, faces = _yunet.detect(frame)
        if faces is None:
            return []
        return [tuple(int(v) for v in f[:4]) for f in faces]

    if face_cascade is not None and not face_cascade.empty():
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return [tuple(b) for b in face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))]
    return []


def detect_face_roi(frame, target_size=(224, 224)):
    """
    Detect the largest face and return its (color, BGR) ROI resized to
    `target_size`, plus the face box. When no face is found, returns a center
    crop with face_box=None so callers can REFUSE to predict on it.
    """
    boxes = _detect_boxes(frame)

    if boxes:
        # Pick the LARGEST face; clamp the box to the frame.
        x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
        H, W = frame.shape[:2]
        x, y = max(0, x), max(0, y)
        w, h = max(1, min(w, W - x)), max(1, min(h, H - y))
        face_roi = frame[y:y + h, x:x + w]
        face_box = (x, y, w, h)
    else:
        # Fallback to center crop if no face detected (face_box=None => untrusted)
        h_f, w_f = frame.shape[:2]
        face_roi = frame[h_f // 4: 3 * h_f // 4, w_f // 4: 3 * w_f // 4]
        face_box = None

    # Resize the ROI to a FIXED size (so the FFT scale is consistent).
    return cv2.resize(face_roi, target_size), face_box


def roi_to_fft(face_roi):
    """
    Per-channel 2D-FFT magnitude of a fixed-size color ROI -> (H, W, 3) uint8.
    Keeps colour moiré cues (a strong spoof/generation signal) and uses a
    FIXED normalization window so absolute magnitude is preserved.
    """
    channels = []
    for c in range(3):
        f_shift = np.fft.fftshift(np.fft.fft2(face_roi[:, :, c]))
        magnitude = 20 * np.log(np.abs(f_shift) + 1e-8)
        magnitude = (magnitude - FFT_VMIN) / (FFT_VMAX - FFT_VMIN)
        magnitude = np.clip(magnitude, 0.0, 1.0) * 255.0
        channels.append(magnitude.astype(np.uint8))
    return np.stack(channels, axis=-1)  # (H, W, 3)


def extract_face_and_spectrum(frame, target_size=(224, 224)):
    """Return (face_roi_bgr, fft_spectrum, face_box) in a single detection pass."""
    face_roi, face_box = detect_face_roi(frame, target_size)
    return face_roi, roi_to_fft(face_roi), face_box


def extract_fft_spectrum(frame, target_size=(224, 224)):
    """Backwards-compatible helper: returns (fft_spectrum, face_box)."""
    face_roi, face_box = detect_face_roi(frame, target_size)
    return roi_to_fft(face_roi), face_box


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    print("Spectra Pipeline Running... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        vis_frame = cv2.resize(frame, (320, 240))
        spectrum_vis, face_box = extract_fft_spectrum(frame)

        # Draw green bounding box on live video if face is detected
        if face_box is not None:
            x, y, w, h = face_box
            scale_x, scale_y = 320 / frame.shape[1], 240 / frame.shape[0]
            cv2.rectangle(
                vis_frame,
                (int(x * scale_x), int(y * scale_y)),
                (int((x + w) * scale_x), int((y + h) * scale_y)),
                (0, 255, 0), 2
            )

        spectrum_display = cv2.resize(spectrum_vis, (320, 240))
        combined = np.hstack((vis_frame, spectrum_display))

        cv2.imshow("Spectra Engine | Left: Live Face Crop | Right: 2D-FFT", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
