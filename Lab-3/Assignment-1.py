import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


IMG_PATH = r"color_img.jpg"   


MAX_SIDE = 1000

#UTILITIES
def ensure_image(path_str):
    p = Path(path_str)
    bgr = cv2.imread(str(p))
    if bgr is None:
        raise FileNotFoundError(f"Could not read image at: {p.resolve()}")
    h, w = bgr.shape[:2]
    scale = min(1.0, MAX_SIDE / max(h, w))
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return bgr

def stats_gray(gray):
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    pdf  = hist / hist.sum()
    cdf  = np.cumsum(pdf)
    return hist, pdf, cdf


def main():
    # 1) Read image
    bgr = ensure_image(IMG_PATH)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # PART A: RGB channel equalization (per channel)
    b, g, r = cv2.split(bgr)
    b_eq = cv2.equalizeHist(b)
    g_eq = cv2.equalizeHist(g)
    r_eq = cv2.equalizeHist(r)
    bgr_rgb_eq = cv2.merge([b_eq, g_eq, r_eq])
    rgb_rgb_eq = cv2.cvtColor(bgr_rgb_eq, cv2.COLOR_BGR2RGB)

    fig1 = plt.figure(figsize=(14, 8))
    # Input
    ax = plt.subplot(3, 3, 1); ax.imshow(rgb); ax.set_title("Input"); ax.axis("off")
    # Channels (gray)
    ax = plt.subplot(3, 3, 4); ax.imshow(r, cmap="gray", vmin=0, vmax=255); ax.set_title("Red"); ax.axis("off")
    ax = plt.subplot(3, 3, 5); ax.imshow(g, cmap="gray", vmin=0, vmax=255); ax.set_title("Green"); ax.axis("off")
    ax = plt.subplot(3, 3, 6); ax.imshow(b, cmap="gray", vmin=0, vmax=255); ax.set_title("Blue"); ax.axis("off")
    # Equalized channels (gray)
    ax = plt.subplot(3, 3, 7);  ax.imshow(r_eq, cmap="gray", vmin=0, vmax=255); ax.set_title("Red (Eq)"); ax.axis("off")
    ax = plt.subplot(3, 3, 8); ax.imshow(g_eq, cmap="gray", vmin=0, vmax=255); ax.set_title("Green (Eq)"); ax.axis("off")
    ax = plt.subplot(3, 3, 9); ax.imshow(b_eq, cmap="gray", vmin=0, vmax=255); ax.set_title("Blue (Eq)"); ax.axis("off")
    # Output
    ax = plt.subplot(3, 3, 3); ax.imshow(rgb_rgb_eq); ax.set_title("Output (RGB Eq)"); ax.axis("off")
    plt.tight_layout()
    plt.show()

    # PART B:HSV value-only equalization
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    v_eq = cv2.equalizeHist(v_ch)
    hsv_eq = cv2.merge([h_ch, s_ch, v_eq])
    bgr_hsv_eq = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
    rgb_hsv_eq = cv2.cvtColor(bgr_hsv_eq, cv2.COLOR_BGR2RGB)

    # Hue visualization
    hue_vis = cv2.cvtColor(
        cv2.merge([h_ch, np.full_like(s_ch, 255), np.full_like(v_ch, 255)]),
        cv2.COLOR_HSV2RGB
    )

    fig2 = plt.figure(figsize=(14, 8))
    ax = plt.subplot(2, 3, 1); ax.imshow(rgb);        ax.set_title("Input"); ax.axis("off")
    ax = plt.subplot(2, 3, 2); ax.imshow(hue_vis);    ax.set_title("Hue channel (visualized)"); ax.axis("off")
    ax = plt.subplot(2, 3, 3); ax.imshow(s_ch, cmap="gray", vmin=0, vmax=255); ax.set_title("Saturation"); ax.axis("off")
    ax = plt.subplot(2, 3, 4); ax.imshow(v_ch, cmap="gray", vmin=0, vmax=255); ax.set_title("Value"); ax.axis("off")
    ax = plt.subplot(2, 3, 5); ax.imshow(v_eq, cmap="gray", vmin=0, vmax=255); ax.set_title("Value (Eq)"); ax.axis("off")
    ax = plt.subplot(2, 3, 6); ax.imshow(rgb_hsv_eq); ax.set_title("Output (HSV V-only Eq)"); ax.axis("off")
    plt.tight_layout()
    plt.show()

    # PART C:Histograms+PDF+CDF
    gray_in      = cv2.cvtColor(bgr,         cv2.COLOR_BGR2GRAY)
    gray_rgb_eq  = cv2.cvtColor(bgr_rgb_eq,  cv2.COLOR_BGR2GRAY)
    gray_hsv_eq  = cv2.cvtColor(bgr_hsv_eq,  cv2.COLOR_BGR2GRAY)

    h_in,  p_in,  c_in  = stats_gray(gray_in)
    h_re,  p_re,  c_re  = stats_gray(gray_rgb_eq)
    h_he,  p_he,  c_he  = stats_gray(gray_hsv_eq)

    fig3 = plt.figure(figsize=(12, 10))
    ax = plt.subplot(3,1,1); ax.plot(h_in, label="Original"); ax.plot(h_re, label="RGB Eq"); ax.plot(h_he, label="HSV(V) Eq"); ax.set_title("Histogram (counts)"); ax.legend()
    ax = plt.subplot(3,1,2); ax.plot(p_in, label="Original"); ax.plot(p_re, label="RGB Eq"); ax.plot(p_he, label="HSV(V) Eq"); ax.set_title("PDF"); ax.legend()
    ax = plt.subplot(3,1,3); ax.plot(c_in, label="Original"); ax.plot(c_re, label="RGB Eq"); ax.plot(c_he, label="HSV(V) Eq"); ax.set_title("CDF"); ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
