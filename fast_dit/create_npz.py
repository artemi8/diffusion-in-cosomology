
from tqdm import tqdm
from PIL import Image
import numpy as np
import argparse
import os

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    real_image_paths = [os.path.join(sample_dir, img) for img in os.listdir(sample_dir)]

    for sample_img in tqdm(real_image_paths, desc="Building .npz file from samples"):
        sample_pil = Image.open(sample_img).convert("RGB")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    print(f'Samples Shape : {samples.shape}')
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", type=str)
    parser.add_argument("--num_samples",  type=int,  default=4000)
    args = parser.parse_args()
    create_npz_from_sample_folder(args.sample_dir, args.num_samples)