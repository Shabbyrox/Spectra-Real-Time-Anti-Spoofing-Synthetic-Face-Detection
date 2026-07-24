import cv2
import numpy as np
import os
import time
from main import detect_face_roi, roi_to_fft

# Save RAW webcam frames straight into the folders train.py reads from, so
# collected samples feed the training pipeline directly.
#   'a' -> AUTHENTIC (a real, live face)            -> raw_data/real
#   's' -> SPOOF (a photo / phone / screen replay)  -> raw_data/fake
#
# NOTE: raw_data/{real,fake} may already contain the Kaggle StyleGAN dataset,
# which is a DIFFERENT task (synthetic-face detection). To train a *liveness*
# detector, move that data aside first so you learn live-vs-replay cleanly.
REAL_DIR = "raw_data/real"
FAKE_DIR = "raw_data/fake"
os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(FAKE_DIR, exist_ok=True)


def count_webcam(folder):
    return len([f for f in os.listdir(folder) if f.startswith("webcam_")])


cap = cv2.VideoCapture(0)
auth_count = count_webcam(REAL_DIR)
spoof_count = count_webcam(FAKE_DIR)

print("=== SPECTRA DATA COLLECTOR ===")
print("Press 'a' to save an AUTHENTIC (live face) frame  -> raw_data/real")
print("Press 's' to save a SPOOF (photo/phone/screen) frame -> raw_data/fake")
print("Press 'q' to quit")
print("\nTip: only frames with a detected face (green box) are saved.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    face_roi, face_box = detect_face_roi(frame)
    spectrum_vis = roi_to_fft(face_roi)
    vis_frame = cv2.resize(frame, (320, 240))

    # Highlight detected face region
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

    cv2.putText(combined, f"Auth (real): {auth_count}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(combined, f"Spoof (fake): {spoof_count}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    if face_box is None:
        cv2.putText(combined, "NO FACE - won't save", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

    cv2.imshow("Spectra Collector", combined)
    key = cv2.waitKey(1) & 0xFF

    # Only save frames where a face was actually detected.
    if key in (ord('a'), ord('s')) and face_box is None:
        print("[!] No face detected - frame not saved.")
        continue

    if key == ord('a'):
        path = os.path.join(REAL_DIR, f"webcam_{int(time.time() * 1000)}.jpg")
        cv2.imwrite(path, frame)
        auth_count += 1
        print(f"[+] Saved Authentic Sample #{auth_count} -> {path}")

    elif key == ord('s'):
        path = os.path.join(FAKE_DIR, f"webcam_{int(time.time() * 1000)}.jpg")
        cv2.imwrite(path, frame)
        spoof_count += 1
        print(f"[!] Saved Spoof Sample #{spoof_count} -> {path}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
