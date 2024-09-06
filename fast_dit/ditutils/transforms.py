
import torch
import numpy as np
import os


class Log1pTransform:
    def __call__(self, x):
        return torch.log1p(x)
    
    def inverse_transform(self, x):
        return torch.expm1(x)
    

class InverseNormalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
    
    def __call__(self, tensor):
        mean = self.mean[None, :, None, None]
        std = self.std[None, :, None, None]
        return tensor * std.to('cuda') + mean.to('cuda')  # reverse normalization


class GlobalMinMaxScaleTransform:
    def __init__(self, data_path, global_min=None, global_max=None, min_val=0, max_val=1):

        if global_min is None and global_max is None:
            self.global_min = global_min
            self.global_max = global_max

        else:
            self.data_path = data_path
            self.find_min_max()
        self.min_val = min_val
        self.max_val = max_val
    
    def __call__(self, x):
        scaled_x = (x - self.global_min) / (self.global_max - self.global_min)
        return scaled_x * (self.max_val - self.min_val) + self.min_val
    
    def inverse_transform(self, scaled_x):
        # Inverse min-max scaling
        original_x = (scaled_x - self.min_val) / (self.max_val - self.min_val)
        return original_x * (self.global_max - self.global_min) + self.global_min
    
    def find_min_max(self):
        all_min_values = []
        all_max_values = []

        # Get list of all .npy files in the folder
        npy_files = [f for f in os.listdir(self.data_path) if f.endswith('.npy')]

        for npy_file in npy_files:
            # Load the .npy file
            array = np.load(os.path.join(self.data_path, npy_file))

            # Calculate min and max in a vectorized way
            all_min_values.append(np.min(np.log1p(array)))
            all_max_values.append(np.max(np.log1p(array)))

        # Find the overall min and max
        global_min = min(all_min_values)
        global_max = max(all_max_values)
    

class DuplicateDim:
    def __call__(self, tensor_arr) -> torch.Tensor:
        return tensor_arr.repeat(3, 1, 1)

class ToTensorNoScaling:
    def __call__(self, x):
        # Check if the input is a NumPy array
        if isinstance(x, np.ndarray):
            # Convert the NumPy array to a torch tensor
            tensor = torch.from_numpy(x)
            # If the NumPy array has shape (H, W, C), permute it to (C, H, W)
            if len(tensor.shape) == 3:  # Check if there are 3 dimensions (H, W, C)
                tensor = tensor.permute(2, 0, 1)
            return tensor #tensor.float()
        else:
            raise TypeError("Input should be a NumPy array")
        


def to_numpy(tensor):
    # Ensure the tensor is on CPU and detached from the computation graph
    return tensor.permute(0, 2, 3, 1).cpu().detach().numpy()

def clip_and_average(arr):
    arr = np.mean(arr, axis=3)
    
    #TODO Increasing power two times as post processing, should be changed to 
    # Band wise weighting to smoother enhancement
    return np.clip(arr, 0, 1e13)

def channel_average(arr):
    return np.mean(arr, axis=3)
    

