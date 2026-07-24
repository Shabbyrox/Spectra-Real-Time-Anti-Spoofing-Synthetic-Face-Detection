import cv2
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from main import extract_face_and_spectrum
from train import SpectraNet, IMAGENET_MEAN, IMAGENET_STD

# Number of recent frames to average predictions over (temporal smoothing).
# Smaller = snappier reaction, larger = steadier but laggier. At a few FPS,
# 10 frames felt like ~5s; 4 keeps it responsive.
SMOOTH_WINDOW = 4

# Below this smoothed confidence we show UNCERTAIN instead of guessing.
CONF_THRESHOLD = 0.60


def run_spectra_engine():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # pretrained=False: we load our fine-tuned weights, no need to fetch ImageNet.
    model = SpectraNet(pretrained=False).to(device)
    try:
        model.load_state_dict(torch.load("spectra_cnn.pth", map_location=device))
        print("[+] Loaded 'spectra_cnn.pth' successfully!")
    except FileNotFoundError:
        print("[!] Error: 'spectra_cnn.pth' not found. Run train.py first!")
        return

    model.eval()
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Error: Could not access webcam.")
        return

    print("\n=== SPECTRA REAL-TIME SECURITY ENGINE RUNNING ===")
    print("Press 'q' to exit.\n")

    prob_history = deque(maxlen=SMOOTH_WINDOW)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        face_roi, spectrum_vis, face_box = extract_face_and_spectrum(frame)

        if face_box is None:
            # No trustworthy face -> do NOT predict on the center-crop fallback.
            prob_history.clear()
            status_text = "NO FACE"
            banner_color = (128, 128, 128)
        else:
            # RGB stream: BGR -> RGB, [0,1], ImageNet-normalize.
            rgb = face_roi[:, :, ::-1].copy()
            rgb = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            rgb = (rgb - mean) / std

            # FFT stream.
            fft = (
                torch.from_numpy(spectrum_vis).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            ).to(device)

            with torch.no_grad():
                probabilities = F.softmax(model(rgb, fft), dim=1)[0].cpu().numpy()

            # Temporal smoothing so the verdict doesn't flicker frame-to-frame.
            prob_history.append(probabilities)
            avg_probs = np.mean(prob_history, axis=0)
            pred_idx = int(np.argmax(avg_probs))
            conf = float(avg_probs[pred_idx])
            conf_pct = conf * 100

            if conf < CONF_THRESHOLD:
                status_text = f"UNCERTAIN ({conf_pct:.1f}%)"
                banner_color = (0, 165, 255)          # orange
            elif pred_idx == 0:
                status_text = f"AUTHENTIC ({conf_pct:.1f}%)"
                banner_color = (0, 255, 0)
            else:
                status_text = f"SPOOF DETECTED ({conf_pct:.1f}%)"
                banner_color = (0, 0, 255)

        vis_frame = cv2.resize(frame, (320, 240))
        if face_box is not None:
            x, y, w, h = face_box
            scale_x, scale_y = 320 / frame.shape[1], 240 / frame.shape[0]
            cv2.rectangle(
                vis_frame,
                (int(x * scale_x), int(y * scale_y)),
                (int((x + w) * scale_x), int((y + h) * scale_y)),
                (255, 255, 255), 1
            )

        spectrum_display = cv2.resize(spectrum_vis, (320, 240))

        cv2.rectangle(vis_frame, (0, 0), (320, 30), banner_color, -1)
        cv2.putText(
            vis_frame, status_text, (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
        )

        combined = np.hstack((vis_frame, spectrum_display))
        cv2.imshow("Spectra AI Guardrail | Left: Stream | Right: 2D-FFT", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_spectra_engine()
