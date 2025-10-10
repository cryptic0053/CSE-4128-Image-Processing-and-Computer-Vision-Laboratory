# =========================
# classwork_rgb_equalization.py
# =========================

# ----- Imports -----
import cv2                     # OpenCV for image IO and image processing
import numpy as np             # NumPy for fast array math
import matplotlib.pyplot as plt# Matplotlib for plotting histograms/PDFs/CDFs

# ----- Helper: compute hist, pdf, cdf for a single channel (0..255) -----
def stats_1ch(channel):
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256])  # 256-bin histogram
    hist = hist.ravel()                                         # flatten to 1D
    total = channel.size                                        # number of pixels
    pdf = hist / total                                          # probability density function
    cdf = np.cumsum(pdf)                                        # cumulative distribution function
    return hist, pdf, cdf

# ----- Read an input color image (BGR order in OpenCV) -----
img_bgr = cv2.imread("col.jpg")   # <-- replace with your image path
if img_bgr is None:
    raise FileNotFoundError("Image not found. Check the path.")

# ----- Split B, G, R channels -----
b, g, r = cv2.split(img_bgr)                # three single-channel grayscale arrays

# ----- Equalize each channel independently (contrast enhancement) -----
b_eq = cv2.equalizeHist(b)                  # histogram equalization for Blue
g_eq = cv2.equalizeHist(g)                  # histogram equalization for Green
r_eq = cv2.equalizeHist(r)                  # histogram equalization for Red

# ----- Merge equalized channels back to a color image -----
img_eq_bgr = cv2.merge([b_eq, g_eq, r_eq])  # recombine channels to color

# ----- Convert to RGB for correct plotting colors in matplotlib -----
img_rgb     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_eq_rgb  = cv2.cvtColor(img_eq_bgr, cv2.COLOR_BGR2RGB)

# ----- Compute stats before/after for each channel -----
hist_b,  pdf_b,  cdf_b  = stats_1ch(b)
hist_g,  pdf_g,  cdf_g  = stats_1ch(g)
hist_r,  pdf_r,  cdf_r  = stats_1ch(r)

hist_be, pdf_be, cdf_be = stats_1ch(b_eq)
hist_ge, pdf_ge, cdf_ge = stats_1ch(g_eq)
hist_re, pdf_re, cdf_re = stats_1ch(r_eq)

# ----- Also compute overall (grayscale) histograms for input vs output (optional) -----
gray_in  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
gray_out = cv2.cvtColor(img_eq_bgr, cv2.COLOR_BGR2GRAY)
hist_in  = cv2.calcHist([gray_in],  [0], None, [256], [0, 256]).ravel()
hist_out = cv2.calcHist([gray_out], [0], None, [256], [0, 256]).ravel()

# ----- Plot everything in a compact dashboard -----
plt.figure(figsize=(14, 16))

# Original vs Equalized images
plt.subplot(5, 2, 1); plt.imshow(img_rgb);    plt.title("Original Image");  plt.axis("off")
plt.subplot(5, 2, 2); plt.imshow(img_eq_rgb); plt.title("Equalized Image"); plt.axis("off")

# Blue channel: Hist (before/after) + PDFs + CDFs
plt.subplot(5, 2, 3); plt.plot(hist_b, label="B Hist (Before)"); plt.plot(hist_be, label="B Hist (After)"); plt.legend(); plt.title("Blue Channel Histogram")
plt.subplot(5, 2, 4); plt.plot(pdf_b,  label="B PDF (Before)");  plt.plot(pdf_be, label="B PDF (After)");  plt.legend(); plt.title("Blue Channel PDF")
plt.subplot(5, 2, 5); plt.plot(cdf_b,  label="B CDF (Before)");  plt.plot(cdf_be, label="B CDF (After)");  plt.legend(); plt.title("Blue Channel CDF")

# Green channel
plt.subplot(5, 2, 6); plt.plot(hist_g, label="G Hist (Before)"); plt.plot(hist_ge, label="G Hist (After)"); plt.legend(); plt.title("Green Channel Histogram")
plt.subplot(5, 2, 7); plt.plot(pdf_g,  label="G PDF (Before)");  plt.plot(pdf_ge, label="G PDF (After)");  plt.legend(); plt.title("Green Channel PDF")
plt.subplot(5, 2, 8); plt.plot(cdf_g,  label="G CDF (Before)");  plt.plot(cdf_ge, label="G CDF (After)");  plt.legend(); plt.title("Green Channel CDF")

# Red channel
plt.subplot(5, 2, 9);  plt.plot(hist_r, label="R Hist (Before)"); plt.plot(hist_re, label="R Hist (After)"); plt.legend(); plt.title("Red Channel Histogram")
plt.subplot(5, 2, 10); plt.plot(pdf_r,  label="R PDF (Before)");  plt.plot(pdf_re, label="R PDF (After)");  plt.legend(); plt.title("Red Channel PDF & CDF?")
# (If you prefer a separate CDF panel for Red, split the last subplot into two; kept compact here.)

plt.tight_layout()
plt.show()
