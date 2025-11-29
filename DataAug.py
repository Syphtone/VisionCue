#======================================
# Data Augmentation for Time Series
#======================================

import pandas as pd
import numpy as np
import glob
import os
from tsaug import TimeWarp # Ensure tsaug is installed: pip install tsaug

# --- 1. Define paths and create directories ---
original_path = r"D:\RP2\RP2 Projects\AI Interview Coach\Face + SMILE\NonConfident_merged"
augmented_path = os.path.join(original_path, "augmented")
original_files = glob.glob(os.path.join(original_path, "*.csv"))

# Create the augmented directory if it doesn't exist
os.makedirs(augmented_path, exist_ok=True)


# --- 2. Define augmentation functions ---
def jitter(data, sigma=0.05):
    """Adds Gaussian noise to the time series data."""
    noise = np.random.normal(0, sigma, data.shape)
    return data + noise

def scale(data, sigma=0.1):
    """Multiplies the time series data by a random scalar."""
    factor = np.random.normal(loc=1.0, scale=sigma, size=data.shape)
    return data * factor

def time_warp(data):
    """Stretches or compresses the time axis of the data."""
    # tsaug requires a 3D input: (n_series, n_timestamps, n_channels)
    data_reshaped = data.values.reshape(1, data.shape[0], data.shape[1])
    tw_augmentor = TimeWarp(n_speed_change=3)
    augmented_array = tw_augmentor.augment(data_reshaped)
    return pd.DataFrame(augmented_array[0], columns=data.columns)

def window_slice(data, window_size=50):
    """Takes a random sub-sequence from the data."""
    if len(data) <= window_size:
        return data
    start = np.random.randint(0, len(data) - window_size)
    return data.iloc[start:start + window_size].reset_index(drop=True)


# --- 3. Process and save augmented files ---
NUM_AUGMENTATIONS_PER_FILE = 5

for file_path in original_files:
    # Load the original data
    df = pd.read_csv(file_path)

    # Get the original file's base name (e.g., "video1")
    filename_base = os.path.splitext(os.path.basename(file_path))[0]

    # Apply Jittering and save
    for i in range(NUM_AUGMENTATIONS_PER_FILE):
        augmented_df = jitter(df.copy())
        new_file_path = os.path.join(augmented_path, f"{filename_base}_jitter_{i}.csv")
        augmented_df.to_csv(new_file_path, index=False)
        print(f"Saved augmented file: {new_file_path}")

    # Apply Scaling and save
    for i in range(NUM_AUGMENTATIONS_PER_FILE):
        augmented_df = scale(df.copy())
        new_file_path = os.path.join(augmented_path, f"{filename_base}_scale_{i}.csv")
        augmented_df.to_csv(new_file_path, index=False)
        print(f"Saved augmented file: {new_file_path}")

    # Apply Time Warping and save
    # Note: time warping is a more complex operation
    for i in range(NUM_AUGMENTATIONS_PER_FILE):
        try:
            augmented_df = time_warp(df.copy())
            new_file_path = os.path.join(augmented_path, f"{filename_base}_timewarp_{i}.csv")
            augmented_df.to_csv(new_file_path, index=False)
            print(f"Saved augmented file: {new_file_path}")
        except Exception as e:
            print(f"Skipping time_warp for {filename_base}: {e}")

    # Apply Window Slicing and save
    for i in range(NUM_AUGMENTATIONS_PER_FILE):
        augmented_df = window_slice(df.copy())
        new_file_path = os.path.join(augmented_path, f"{filename_base}_slice_{i}.csv")
        augmented_df.to_csv(new_file_path, index=False)
        print(f"Saved augmented file: {new_file_path}")
