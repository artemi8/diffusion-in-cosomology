import os
import numpy as np
from PIL import Image
import argparse

def verify_conversion_integrity(npy_file, tiff_file):
    if npy_file.endswith(".npy"):

        # Load the original .npy file
        npy_data = np.load(npy_file)

        # Load the corresponding .tiff file
        tiff_image = Image.open(tiff_file)
        tiff_data = np.array(tiff_image, dtype=np.float32)

        # Verify the content
        if np.array_equal(npy_data, tiff_data):
            print(f"Verification passed for {npy_file}")
        else:
            print(f"Verification failed for {npy_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify the integrity of .npy to .tiff conversion.")
    parser.add_argument("npy_file", type=str, help="Path to a .npy file")
    parser.add_argument("tiff_file", type=str, help="Path to a .tiff file")
    
    args = parser.parse_args()
    
    verify_conversion_integrity(args.npy_file, args.tiff_file)
