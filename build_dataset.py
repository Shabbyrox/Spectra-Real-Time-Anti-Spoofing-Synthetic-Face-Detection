import cv2
import numpy as np
import os
import glob
from main import extract_fft_spectrum

def process_folder(image_folder, save_folder, max_samples=400):
    """
    Reads raw images, extracts face-cropped 2D-FFT spectra, 
    and saves them as training tensors.
    """
    os.makedirs(save_folder, exist_ok=True)
    
    # Grab all JPG and PNG images from the raw folder
    images = glob.glob(os.path.join(image_folder, "*.jpg")) + \
             glob.glob(os.path.join(image_folder, "*.png"))
    
    if not images:
        print(f"[!] No images found in {image_folder}. Please check the folder path.")
        return

    count = 0
    for img_path in images:
        if count >= max_samples:
            break
            
        frame = cv2.imread(img_path)
        if frame is None:
            continue
            
        # Run our custom spatial frequency engine
        spectrum_vis, face_box = extract_fft_spectrum(frame)
        
        # QUALITY CONTROL: Only save the sample if the Haar Cascade actually found a face
        if face_box is not None:
            save_path = os.path.join(save_folder, f"sample_{count}.npy")
            np.save(save_path, spectrum_vis)
            count += 1
            
            # Print progress every 50 images
            if count % 50 == 0:
                print(f"Processed {count}/{max_samples} samples for {save_folder}...")

    print(f"[+] Finished processing {count} valid face samples into {save_folder}")

if __name__ == "__main__":
    print("=== SPECTRA AUTOMATED DATASET BUILDER ===")
    
    # Clear out old manual data
    os.system("rm -rf data/authentic/* data/spoof/*")
    
    print("\nProcessing AUTHENTIC images...")
    # Change "raw_data/real" to match your unzipped folder name if it's different
    process_folder("raw_data/real", "data/authentic", max_samples=400)
    
    print("\nProcessing SPOOF images...")
    # Change "raw_data/fake" to match your unzipped folder name if it's different
    process_folder("raw_data/fake", "data/spoof", max_samples=400)
    
    print("\n[+] Dataset built successfully! You can now run 'python train.py'")