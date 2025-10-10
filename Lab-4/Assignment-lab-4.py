import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

# Load grayscale image
img = cv2.imread("gray_input.png", 0)

# Define displacements
distances = [1]  # (horizontal, vertical, diagonal distances)
angles = [0, np.pi/2, np.pi/4]  # 0°=horizontal, 90°=vertical, 45°=diagonal

# Compute GLCMs
glcm = graycomatrix(img, 
                    distances=distances, 
                    angles=angles, 
                    levels=256, 
                    symmetric=True, 
                    normed=True)

# Features
features = {}
for prop in ["contrast", "homogeneity", "energy"]:
    features[prop] = graycoprops(glcm, prop)

# Custom: Maximum probability & Entropy
def max_prob_and_entropy(glcm):
    probs = glcm[:, :, 0, :]
    max_prob = np.max(probs, axis=(0, 1))
    entropy = []
    for k in range(probs.shape[-1]):
        p = probs[:, :, k]
        e = -np.sum(p * np.log2(p + 1e-12))
        entropy.append(e)
    return max_prob, entropy

max_prob, entropy = max_prob_and_entropy(glcm)

# Print results
print("GLCM Texture Features:")
for i, angle in enumerate(["Horizontal", "Vertical", "Diagonal"]):
    print(f"\nDirection: {angle}")
    print(f"Contrast    = {features['contrast'][0,i]:.4f}")
    print(f"Homogeneity = {features['homogeneity'][0,i]:.4f}")
    print(f"Energy      = {features['energy'][0,i]:.4f}")
    print(f"Max Prob    = {max_prob[i]:.4f}")
    print(f"Entropy     = {entropy[i]:.4f}")

# --- For a 5x5 patch example ---
patch = img[0:5, 0:5]
glcm_patch = graycomatrix(patch, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
print("\n5x5 Patch:\n", patch)
print("GLCM of patch:\n", glcm_patch[:, :, 0, 0])
