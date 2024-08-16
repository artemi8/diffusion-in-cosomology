import os
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm

def convert_npy_to_tiff(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for filename in tqdm(os.listdir(src_dir)):
        if filename.endswith(".npy"):
            npy_path = os.path.join(src_dir, filename)
            tiff_path = os.path.join(dst_dir, filename.replace(".npy", ".tiff"))

            # Load the .npy file
            npy_data = np.load(npy_path)

            # Ensure the data is in float32 format
            if npy_data.dtype != np.float32:
                npy_data = npy_data.astype(np.float32)

            # Convert to image and save as .tiff
            tiff_image = Image.fromarray(npy_data)
            tiff_image.save(tiff_path)

     print(f"Converted Files to TIFF format!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .npy files to .tiff format.")
    parser.add_argument("src", type=str, help="Source directory containing .npy files")
    parser.add_argument("dst", type=str, help="Destination directory to save .tiff files")
    
    args = parser.parse_args()
    
    convert_npy_to_tiff(args.src, args.dst)
