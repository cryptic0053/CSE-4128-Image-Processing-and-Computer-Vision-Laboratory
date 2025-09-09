import numpy as np
import cv2

# Load grayscale image
img = cv2.imread('box.jpg', cv2.IMREAD_GRAYSCALE)

# Gaussian kernel (sum = 256) from your template; normalize to keep brightness
kernel1 = np.array([
    [0, 1, 2, 1, 0],
    [1, 3, 5, 3, 1],
    [2, 5, 9, 5, 2],
    [1, 3, 5, 3, 1],
    [0, 1, 2, 1, 0]
], dtype=np.float32)
kernel1 = kernel1 / kernel1.sum()  # preserve average intensity

# ------------------------------------------------------------------
# >>> Filled-in section: padding, convolution, normalization, cropping
h, w = img.shape
pad_h = kernel1.shape[0] // 2   # 2
pad_w = kernel1.shape[1] // 2   # 2

# Reflect padding so the filter works at borders
img_bordered = cv2.copyMakeBorder(
    img, pad_h, pad_h, pad_w, pad_w, borderType=cv2.BORDER_REFLECT
)

# Convolve on the padded image; anchor at center (2,2)
img_conv = cv2.filter2D(
    src=img_bordered.astype(np.float32),
    ddepth=cv2.CV_32F,
    kernel=kernel1,
    anchor=(pad_w, pad_h)
)

# Normalize to 0..255 for viewing
norm = np.round(cv2.normalize(img_conv, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)

# Crop the padding area to return to original size
norm_cropped = norm[pad_h:h + pad_h, pad_w:w + pad_w]
# ------------------------------------------------------------------

# Show all images
cv2.imshow('Original Grayscale Image', img)
cv2.imshow('Bordered Image', img_bordered)
cv2.imshow('Convolution Image', img_conv.astype(np.uint8))  # quick view (will look dark)
cv2.imshow('Normalized Image', norm)
cv2.imshow('Normalized Cropped Image', norm_cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()
