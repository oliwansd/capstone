import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os, random
from s2cloudless import S2PixelCloudDetector
import random
import rasterio
import pandas as pd


# %%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def load_and_prepare_vis(path):
    """Load a PNG/JPG, drop alpha if present, normalize to [0,1]."""
    arr = np.array(Image.open(path))
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]
    return arr.astype(np.float32) / 255.0  # shape (H,W,3)

def load_and_prepare_nir(path):
    """
    Load a .npy IR stack with ≥3 channels,
    collapse to one band via mean, normalize to [0,1].
    """
    raw = np.load(path).astype(np.float32)   # e.g. shape (H,W,3)
    # collapse channels:
    nir = np.mean(raw[..., :3], axis=-1)     # shape (H,W)
    # min–max normalize:
    mn, mx = nir.min(), nir.max()
    if mx > mn:
        nir = (nir - mn) / (mx - mn)
    else:
        nir = np.zeros_like(nir)
    return nir

def load_and_prepare_ir(path):
    """
    Loads an .npy file, extracts the first 3 channels,
    converts to float32, and min–max normalizes to [0,1].
    """
    data = np.load(path)
    
    # Ensure at least 3 channels
    if data.ndim == 3 and data.shape[2] >= 3:
        rgb = data[..., :3]
    else:
        raise ValueError(f"Expected at least 3 channels, got shape {data.shape}")
    
    # Convert to float32
    rgb = rgb.astype(np.float32)
    
    # Min–max normalize
    min_val = rgb.min()
    max_val = rgb.max()
    if max_val > min_val:
        rgb_norm = (rgb - min_val) / (max_val - min_val)
    else:
        # All pixels identical: return zeros
        rgb_norm = np.zeros_like(rgb)
    
    return rgb_norm

def compute_cloud_mask(vis, nir, blue_thresh=0.6, nir_thresh=0.3):
    """
    Simple rule: cloud if blue is high AND NIR is low.
     - vis[...,2] is blue band
     - nir is normalized [0,1] float
    """
    blue = vis[..., 2]
    return (blue > blue_thresh) & (nir < nir_thresh)
    


def plot_rgb_image(img, title='Image', figsize=(8, 8)):
    """
    Displays a normalized RGB image (values in [0,1]).
    """
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()



def load_and_prep_visual_image(path):
    """
    Load a visible-light PNG, drop alpha if present,
    and normalize pixel values to [0,1].
    
    Parameters:
    - path (str): Path to the image file.
    
    Returns:
    - arr_norm (np.ndarray): Normalized H×W×3 array.
    """
    img = Image.open(path)
    arr = np.array(img)
    
    # Drop alpha channel if there is one
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]
    
    # Normalize to [0,1]
    arr_norm = arr.astype(np.float32) / 255.0
    return arr_norm

def plot_visual_image(img, title='Visible Image', figsize=(8, 8)):
    """
    Display a normalized RGB image array.
    
    Parameters:
    - img (np.ndarray): Image array in [0,1].
    - title (str): Plot title.
    - figsize (tuple): Figure size.
    """
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()


def remove_empty_folders(base_dir):
    for root, dirs, files in os.walk(base_dir, topdown=False):
        for d in dirs:
            folder_path = os.path.join(root, d)
            if not os.listdir(folder_path):
                os.rmdir(folder_path)
                print(f"Removed empty folder: {folder_path}")

# Example usage



def delete_all_white_pngs(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith('.png'):
            path = os.path.join(directory, filename)
            try:
                img = Image.open(path).convert('RGB')
                arr = np.array(img)
                if np.all(arr == 0):
                    os.remove(path)
                    print(f"Deleted all-white image: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")






