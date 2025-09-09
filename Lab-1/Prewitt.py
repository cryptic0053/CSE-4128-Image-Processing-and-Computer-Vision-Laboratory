import cv2
import numpy as np

# Step 1: Load grayscale image
img = cv2.imread("lena.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Image not found. Put lena.jpg in the same folder as Prewitt.py")

cv2.imshow("Original", img)

# Step 2: Define Prewitt Kernels
Gx = np.array([[-1, 0, 1],
               [-1, 0, 1],
               [-1, 0, 1]], dtype=np.float32)

Gy = np.array([[-1, -1, -1],
               [ 0,  0,  0],
               [ 1,  1,  1]], dtype=np.float32)

# Step 3: Apply convolution
edges_x = cv2.filter2D(img.astype(np.float32), -1, Gx, anchor=(0,0))
edges_y = cv2.filter2D(img.astype(np.float32), -1, Gy, anchor=(0,0))

cv2.imshow("Prewitt Gx (Vertical Edges)", cv2.convertScaleAbs(edges_x))
cv2.imshow("Prewitt Gy (Horizontal Edges)", cv2.convertScaleAbs(edges_y))

# Step 4: Gradient magnitude
grad_mag = np.sqrt(np.square(edges_x) + np.square(edges_y))
grad_mag = np.round(cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)

cv2.imshow("Gradient Magnitude", grad_mag)

cv2.waitKey(0)
cv2.destroyAllWindows()
