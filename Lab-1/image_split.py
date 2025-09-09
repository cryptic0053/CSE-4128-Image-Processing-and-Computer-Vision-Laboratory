import cv2
import numpy as np
import matplotlib.pyplot as plt

img_color = cv2.imread('box.jpg', 1)
img_gray = cv2.imread('box.jpg', cv2.IMREAD_GRAYSCALE)  # 0 -> gray, 1 -> color
print(img_color.shape)

cv2.imshow('Original image', img_color)
cv2.imshow('Gray_scale image', img_gray)
cv2.waitKey(0)

#%%
b1, g1, r1 = cv2.split(img_color)

# ------------------------------------------------------------------
# >>> Filled-in section: show RGB channels and HSV channels
cv2.imshow("Blue channel", b1)
cv2.imshow("Green channel", g1)
cv2.imshow("Red channel", r1)

# Merge back to verify split/merge works
merged_rgb = cv2.merge((b1, g1, r1))
cv2.imshow("Merged BGR (should look like original)", merged_rgb)

# Convert to HSV and split
img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(img_hsv)
cv2.imshow("HSV - H (Hue)", h)
cv2.imshow("HSV - S (Saturation)", s)
cv2.imshow("HSV - V (Value)", v)
# ------------------------------------------------------------------

# (Optional) quick matplotlib comparison for grayscale vs any channel
# plt.figure(figsize=(8,4))
# plt.subplot(1,2,1); plt.imshow(img_gray, cmap='gray'); plt.title('Grayscale'); plt.axis('off')
# plt.subplot(1,2,2); plt.imshow(v, cmap='gray');       plt.title('HSV - V');  plt.axis('off')
# plt.tight_layout(); plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
