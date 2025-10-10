import sys
import os
import cv2
import numpy as np
import matplotlib
try:
    matplotlib.use("Qt5Agg")
except Exception:
    pass
import matplotlib.pyplot as plt

IMG_PATH     = "lena.jpg"
SIGMA        = 1.2   # LoG Gaussian sigma
ZS_THRESH    = 12.0  # Zero-Cross Strength threshold
VAR_THRESH   = 60.0  # Local variance threshold
HOW_MANY     = 6      # 6 panels

# Layout
CAPTION_FONTSIZE = 18
TITLE_FONTSIZE   = 20
FIGSIZE          = (14, 8)   # overall inches
CMAP_GRAY        = "gray"

# Helpers

def ensure_odd(n: int) -> int:
    n = int(round(n))
    return n if (n % 2) else n + 1

def generate_log_kernel(sigma, size=None):
    if size is None:
        size = ensure_odd(max(3, int(round(9 * sigma))))
    else:
        size = ensure_odd(size)
    k = size // 2
    y, x = np.mgrid[-k:k+1, -k:k+1]
    norm = (x**2 + y**2) / (2 * sigma**2)
    LoG = (norm - 1) * np.exp(-norm) / (np.pi * sigma**4)
    LoG -= LoG.mean()
    return LoG.astype(np.float32)

def normalize8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr, dtype=np.uint8)
    out = (arr - mn) / (mx - mn) * 255.0
    return out.astype(np.uint8)

def zero_crossing_4n(resp: np.ndarray) -> np.ndarray:
    h, w = resp.shape
    zc = np.zeros((h, w), dtype=np.uint8)
    s = np.sign(resp)
    for y in range(1, h-1):
        for x in range(1, w-1):
            c = s[y, x]
            if c == 0:
                nb = [s[y-1, x], s[y+1, x], s[y, x-1], s[y, x+1]]
                if any(v > 0 for v in nb) and any(v < 0 for v in nb):
                    zc[y, x] = 255
            else:
                if (c > 0 and (s[y-1, x] < 0 or s[y+1, x] < 0 or s[y, x-1] < 0 or s[y, x+1] < 0)) or \
                   (c < 0 and (s[y-1, x] > 0 or s[y+1, x] > 0 or s[y, x-1] > 0 or s[y, x+1] > 0)):
                    zc[y, x] = 255
    return zc

def zero_cross_strength(img: np.ndarray, zc_mask: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    h, w = img.shape
    ZS = np.zeros((h, w), dtype=np.float32)
    for y in range(1, h-1):
        for x in range(1, w-1):
            if zc_mask[y, x] == 255:
                c = img[y, x]
                nb = [img[y-1, x], img[y+1, x], img[y, x-1], img[y, x+1]]
                ZS[y, x] = np.sum(np.abs(c - np.array(nb)))
    return ZS

def local_variance(img: np.ndarray, win: int = 5) -> np.ndarray:
    assert win % 2 == 1
    f = img.astype(np.float32)
    f2 = f * f
    I  = cv2.integral(f)   # (h+1, w+1)
    I2 = cv2.integral(f2)  # (h+1, w+1)
    h, w = img.shape
    pad = win // 2
    var_map = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        y1, y2 = max(0, y - pad), min(h - 1, y + pad)
        iy1, iy2 = y1, y2 + 1
        for x in range(w):
            x1, x2 = max(0, x - pad), min(w - 1, x + pad)
            ix1, ix2 = x1, x2 + 1
            sum_ = I[iy2, ix2] - I[iy1, ix2] - I[iy2, ix1] + I[iy1, ix1]
            sum2 = I2[iy2, ix2] - I2[iy1, ix2] - I2[iy2, ix1] + I2[iy1, ix1]
            area = (y2 - y1 + 1) * (x2 - x1 + 1)
            mean  = sum_ / area
            mean2 = sum2 / area
            var_map[y, x] = max(0.0, float(mean2 - mean * mean))
    return var_map

# Load 

search = [
    IMG_PATH,
    os.path.join("/mnt/data", IMG_PATH),
    os.path.join("/mnt/data", "Lena.jpg"),
]
gray = None
for p in search:
    if p and os.path.exists(p):
        tmp = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if tmp is not None:
            gray = tmp
            break
if gray is None:
    raise FileNotFoundError(f"Could not load image from {search}")


# Edge pipeline
log_kernel = generate_log_kernel(SIGMA)
log_resp   = cv2.filter2D(gray.astype(np.float32), -1, log_kernel, borderType=cv2.BORDER_REFLECT)
zc_mask    = zero_crossing_4n(log_resp)
ZS         = zero_cross_strength(gray, zc_mask)
edge_simple  = (ZS > ZS_THRESH).astype(np.uint8) * 255
var_map      = local_variance(gray, win=5)
edge_robust  = np.zeros_like(gray, dtype=np.uint8)
edge_robust[(zc_mask == 255) & (var_map > VAR_THRESH)] = 255
edge_combined = np.zeros_like(gray, dtype=np.uint8)
edge_combined[(edge_simple == 255) & (edge_robust == 255)] = 255


candidates = [
    ("Original (Grayscale)", gray),
    ("LoG Response", normalize8(log_resp)),
    ("Zero-Crossing (4N)", zc_mask),
    ("Zero-Cross Strength (ZS)", normalize8(ZS)),
    (f"Robust Edge (σ² > {VAR_THRESH})", edge_robust),
    ("Combined Edge (ZS ∧ Robust)", edge_combined),
]
if HOW_MANY not in (5, 6):
    HOW_MANY = 6
panels = candidates[:HOW_MANY]


# Plot
rows = 2
cols = 3
plt.figure(figsize=FIGSIZE)
plt.suptitle("Robust Laplacian-based Edge Detector",
             fontsize=TITLE_FONTSIZE, fontweight="bold")

for i, (title, img) in enumerate(panels, start=1):
    plt.subplot(rows, cols, i)
    if img.ndim == 2:
        plt.imshow(img, cmap=CMAP_GRAY)
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title, fontsize=CAPTION_FONTSIZE)
    plt.axis("off")

# hide any empty subplots if HOW_MANY==5
if HOW_MANY == 5:
    plt.subplot(rows, cols, 6)
    plt.axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()
