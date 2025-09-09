import numpy as np
import cv2

img = cv2.imread('box.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('grayscaled input image', img)
print(img.max())
print(img.min())

h, w = img.shape
out = np.zeros((h, w), dtype=np.uint8)
cv2.imshow('initial output image', out)

# ------------------------------------------------------------------
# >>> Filled-in section: make a simple filtered version of the image
# Use a 3x3 mean filter as an example processing step
kernel = np.ones((3, 3), np.float32) / 9.0
out = cv2.filter2D(img, ddepth=-1, kernel=kernel)
# ------------------------------------------------------------------

cv2.imshow('new output image', out)
print(out)

out = np.round(cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)
cv2.imshow('normalised output image', out)

img_bordered = cv2.copyMakeBorder(
    src=img, top=25, bottom=25, left=25, right=25, borderType=cv2.BORDER_CONSTANT
)
cv2.imshow('bordered image', img_bordered)

cv2.waitKey(0)
cv2.destroyAllWindows()
