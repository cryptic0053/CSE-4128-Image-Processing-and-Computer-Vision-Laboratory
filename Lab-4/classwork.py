import cv2
import numpy as np
from tabulate import tabulate
import glob, os

def get_region_descriptors(binary_img):
    # --- Area ---
    area = np.count_nonzero(binary_img)

    # --- Erosion for border ---
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(binary_img, kernel, iterations=1)
    border = binary_img - eroded

    # --- Perimeter ---
    perimeter = np.count_nonzero(border)

    # --- Bounding box for MaxDiameter ---
    ys, xs = np.where(binary_img > 0)
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    max_diameter = max(xmax - xmin, ymax - ymin)

    # --- Form factor, Roundness, Compactness ---
    form_factor = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)
    roundness   = (perimeter ** 2) / (4 * np.pi * area + 1e-6)
    compactness = perimeter ** 2 / (area + 1e-6)

    return [form_factor, roundness, compactness]

# Example: process training and testing binary images
train_images = sorted(glob.glob("train_shapes/*.png"))
test_images  = sorted(glob.glob("test_shapes/*.png"))

train_desc = [get_region_descriptors(cv2.imread(p, 0)) for p in train_images]
test_desc  = [get_region_descriptors(cv2.imread(p, 0)) for p in test_images]

# --- Matching: Euclidean distances between descriptors ---
distances_matrix = []
for t in test_desc:
    row = []
    for g in train_desc:
        d = np.linalg.norm(np.array(t) - np.array(g))
        row.append(d)
    distances_matrix.append(row)

# --- Show as table ---
row_headers = [f"Test {i+1}" for i in range(len(test_desc))]
col_headers = [f"GT {i+1}" for i in range(len(train_desc))]
print(tabulate(distances_matrix, headers=col_headers, showindex=row_headers, tablefmt="grid"))
