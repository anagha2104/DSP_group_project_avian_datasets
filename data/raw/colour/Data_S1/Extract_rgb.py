"""
Dependencies:

Python 3.9.13
OpenCV 4.6.0
NumPy 1.21.5
pandas 1.4.4
Usage: Place your .tif images into a folder and change input_folder path in the script. 
Run: python extract_rgb.py Results will be saved into a subfolder named RGB/ within the same directory.

"""

import numpy as np
import cv2
import random
import pandas as pd
import os

def pooling(img, poolsize=9, poolstride=9, mode='mean'):
    """
    Apply mean pooling to an image channel.

    Parameters:
        img (2D array): Grayscale image or single channel.
        poolsize (int): Pooling window size.
        poolstride (int): Step size for pooling.
        mode (str): Pooling mode (only 'mean' supported).

    Returns:
        2D array: Pooled image.
    """
    row, col = img.shape
    out_row = int(np.ceil(row / poolstride))
    out_col = int(np.ceil(col / poolstride))

    # Padding
    padded_img = np.pad(img, ((0, poolsize), (0, poolsize)), mode='edge')

    # Mean pooling
    output_img = np.zeros((out_row, out_col))
    for i in range(out_row):
        for j in range(out_col):
            startx = j * poolstride
            starty = i * poolstride
            patch = padded_img[starty:starty + poolsize, startx:startx + poolsize]
            output_img[i, j] = np.mean(patch, dtype=int)

    return output_img

def random_sampling(r_pooling, g_pooling, b_pooling, h, w, sample_size=1000):
    """
    Randomly sample RGB values from pooled image channels.

    Parameters:
        r_pooling, g_pooling, b_pooling: 2D arrays of pooled image channels.
        h, w: Dimensions of pooled image.
        sample_size: Number of RGB samples to collect.

    Returns:
        dict: Dictionary with keys 'r', 'g', 'b'.
    """
    r, g, b = [], [], []
    count = 0
    while count < sample_size:
        i = random.randint(0, h - 1)
        j = random.randint(0, w - 1)
        if (r_pooling[i, j], g_pooling[i, j], b_pooling[i, j]) == (255, 255, 255):
            continue
        r.append(r_pooling[i, j])
        g.append(g_pooling[i, j])
        b.append(b_pooling[i, j])
        count += 1
    return {'r': r, 'g': g, 'b': b}

def extract_rgb_from_images(folder_path):
    """
    Extract RGB values from all .tif images in the given folder and save to CSV.

    Parameters:
        folder_path (str): Path to the folder containing .tif files.
    """
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.tif')]
    save_path = os.path.join(folder_path, 'RGB')
    os.makedirs(save_path, exist_ok=True)

    for img_name in image_files:
        name_no_ext = os.path.splitext(img_name)[0]
        save_file = os.path.join(save_path, f"{name_no_ext}.csv")

        if os.path.exists(save_file):
            print(f"{name_no_ext}.csv already exists. Skipping.")
            continue

        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Failed to read {img_path}. Skipping.")
            continue

        r_values = img[:, :, 2]
        g_values = img[:, :, 1]
        b_values = img[:, :, 0]

        r_pooling = pooling(r_values)
        g_pooling = pooling(g_values)
        b_pooling = pooling(b_values)
        h, w = r_pooling.shape

        rgb_data = random_sampling(r_pooling, g_pooling, b_pooling, h, w)
        pd.DataFrame(rgb_data).to_csv(save_file, index=False)
        print(f"Saved: {save_file}")

    print("All images processed successfully.")

if __name__ == "__main__":
    # 📁 Modify the path below to the directory containing your .tif files
    input_folder = "your .tif files"
    extract_rgb_from_images(input_folder)