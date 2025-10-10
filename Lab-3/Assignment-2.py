import cv2
import numpy as np
import matplotlib.pyplot as plt


IMG_PATH = r"histogram.jpg"

#FUNCTIONS
def stats_gray(gray):
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    pdf  = hist / hist.sum()
    cdf  = np.cumsum(pdf)
    return hist, pdf, cdf

#MAIN
def main():
    # 1) Read input image
    img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Could not read image. Check path!")

    # Optional resize for plots
    H, W = img.shape
    scale = 700 / max(H, W)
    if scale < 1.0:
        img = cv2.resize(img, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA)

    # 2) Create double-Gaussian target distribution
    x = np.arange(256)

    # Parameters
    mu1, sigma1 = 30,20
    mu2, sigma2 = 165,8
    w1, w2 = 0.5, 0.5

    g1 = np.exp(-0.5 * ((x - mu1) / sigma1) ** 2) / (sigma1 * np.sqrt(2*np.pi))
    g2 = np.exp(-0.5 * ((x - mu2) / sigma2) ** 2) / (sigma2 * np.sqrt(2*np.pi))
    mix = w1 * g1 + w2 * g2

    target_pdf = mix / mix.sum()
    target_cdf = np.cumsum(target_pdf)
    target_hist_like = target_pdf * img.size

    # 3) Compute input histogram/PDF/CDF
    hist_in, pdf_in, cdf_in = stats_gray(img)

    # 4) Build LUT mapping using CDF matching
    map_lut = np.searchsorted(target_cdf, cdf_in)
    map_lut = np.clip(map_lut, 0, 255).astype(np.uint8)

    # Apply LUT to get output image
    out = cv2.LUT(img, map_lut)

    # Output histogram/PDF/CDF
    hist_out, pdf_out, cdf_out = stats_gray(out)

    # 5) Show results
    fig = plt.figure(figsize=(14, 10))

    # Top row
    ax = plt.subplot(2, 3, 1)
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title("Input Image"); ax.axis("off")

    ax = plt.subplot(2, 3, 3)
    ax.imshow(out, cmap="gray", vmin=0, vmax=255)
    ax.set_title("Output Image"); ax.axis("off")

    # Target histogram
    ax = plt.subplot(2, 3, 2)
    ax.fill_between(x, target_hist_like, step="mid", alpha=0.9)
    ax.set_xlim(0, 255)
    ax.set_title("Target histogram\nμ1=30, σ1=8   μ2=165, σ2=20")

    # Histograms of input/output
    ax = plt.subplot(2, 3, 4)
    ax.bar(x, hist_in, width=1.0)
    ax.set_xlim(0, 255); ax.set_title("Histogram of the Input Image")

    ax = plt.subplot(2, 3, 6)
    ax.bar(x, hist_out, width=1.0)
    ax.set_xlim(0, 255); ax.set_title("Histogram of the Output Image")

    plt.tight_layout()
    plt.show()

    # 6) Show PDFs and CDFs
    fig2 = plt.figure(figsize=(12, 8))

    ax = plt.subplot(2, 2, 1)
    ax.plot(pdf_in, label="Input PDF")
    ax.plot(target_pdf, label="Target PDF")
    ax.plot(pdf_out, label="Output PDF")
    ax.set_xlim(0, 255); ax.set_title("PDFs"); ax.legend()

    ax = plt.subplot(2, 2, 2)
    ax.plot(cdf_in, label="Input CDF")
    ax.plot(target_cdf, label="Target CDF")
    ax.plot(cdf_out, label="Output CDF")
    ax.set_xlim(0, 255); ax.set_title("CDFs"); ax.legend()

    ax = plt.subplot(2, 2, 3)
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title("Input"); ax.axis("off")

    ax = plt.subplot(2, 2, 4)
    ax.imshow(out, cmap="gray", vmin=0, vmax=255)
    ax.set_title("Output"); ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
