import numpy as np
import os

def calculate_min_max_from_npy_folder(folder_path):
    all_min_values = []
    all_max_values = []

    # Get list of all .npy files in the folder
    npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

    for npy_file in npy_files:
        # Load the .npy file
        array = np.load(os.path.join(folder_path, npy_file))

        # Calculate min and max in a vectorized way
        all_min_values.append(np.min(np.log1p(array)))
        all_max_values.append(np.max(np.log1p(array)))

    # Find the overall min and max
    global_min = min(all_min_values)
    global_max = max(all_max_values)

    return global_min, global_max
