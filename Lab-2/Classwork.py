import cv2
import numpy as np

# -----------------------------
# Parameters
# -----------------------------
IMG_PATH = "lena.jpg"     # use your input image
sigma = 1                 # σ for Gaussian derivative
T_low, T_high = 100, 150  # thresholds

# -----------------------------
# Helper: show bigger images
# -----------------------------
def show_big(title, img, scale=2.5):
    """
    Resize image for display only (not for processing).
    scale=2.5 means 2.5x bigger display.
    """
    h, w = img.shape[:5]
    resized = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_NEAREST)
    cv2.imshow(title, resized)

# -----------------------------
# Step 1: Load grayscale image
# -----------------------------
img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Image not found. Put lena.jpg in the same folder.")

# -----------------------------
# Step 2: Smooth with Gaussian
# -----------------------------
k = 7 * sigma
blur = cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma)

# -----------------------------
# Step 3: Derivatives (Sobel ≈ Gaussian derivative)
# -----------------------------
gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)

# -----------------------------
# Step 4: Gradient magnitude
# -----------------------------
grad_mag = cv2.magnitude(gx, gy)
grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# -----------------------------
# Step 5: Double thresholding
# -----------------------------
edges = np.zeros_like(grad_mag, dtype=np.uint8)
edges[grad_mag >= T_high] = 255          # strong edge
edges[(grad_mag >= T_low) & (grad_mag < T_high)] = 128  # weak edge

# -----------------------------
# Show results (BIG windows)
# -----------------------------
show_big("Original", img, scale=7)
show_big("Gaussian Blurred", blur, scale=7)
show_big("Gradient Magnitude", grad_mag, scale=7)
show_big("Edges (Double Threshold)", edges, scale=7)

cv2.waitKey(0)
cv2.destroyAllWindows()
