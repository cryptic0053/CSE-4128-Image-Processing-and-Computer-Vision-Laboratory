

import os, cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt
from math import log10

# CONFIG
IMAGE_DIR = "images"
IMAGE_FILES = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]
OUTPUT_DIR = "outputs"   
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPS = 1e-12

# HELPERS
def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

def binarize(gray):
    """Otsu threshold; ensure foreground is white."""
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.sum(th == 255) < np.sum(th == 0):
        th = cv2.bitwise_not(th)
    return th

def hu_1to4_on_mask(gray):
    """Compute Hu moments 1..4 on binarized mask; log transform: -sign(h) * log10(|h|+eps)."""
    mask = binarize(gray)
    m = cv2.moments(mask)
    hu = cv2.HuMoments(m).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + EPS)
    return hu_log[:4]

def chi_squared(f, g):
    f, g = f.astype(float), g.astype(float)
    return float(np.sum(((f - g) ** 2) / (f + g + EPS)))


def rotate(gray, angle=45):
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def translate(gray, tx=30, ty=30):
    h, w = gray.shape
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def scale_canvas(gray, factor=0.5):
    """Scale to factor then paste centered on a black canvas of original size."""
    h, w = gray.shape
    scaled = cv2.resize(gray, (max(1,int(w*factor)), max(1,int(h*factor))), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros_like(gray)
    y = (h - scaled.shape[0]) // 2
    x = (w - scaled.shape[1]) // 2
    canvas[y:y+scaled.shape[0], x:x+scaled.shape[1]] = scaled
    return canvas

def mirror_h(gray):
    return cv2.flip(gray, 1)

def contrast(gray, alpha=1.3, beta=20):
    return cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

def center_patch(gray, k=5):
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    r = k // 2
    patch = gray[max(0,cy-r):min(h,cy+r+1), max(0,cx-r):min(w,cx+r+1)]
    if patch.shape != (k, k):
        patch = cv2.copyMakeBorder(
            patch,
            0, max(0, k - patch.shape[0]),
            0, max(0, k - patch.shape[1]),
            cv2.BORDER_REPLICATE
        )[:k, :k]
    return patch

def build_variants(gray):
    return {
        "Original": gray,
        "Scaled 0.5×": scale_canvas(gray, 0.5),
        "Contrast +20": contrast(gray, 1.3, 20),
        "Mirrored (H)": mirror_h(gray),
        "Rotated 45°": rotate(gray, 45),
        "Translated (+30,+30)": translate(gray, 30, 30),
    }

def montage_3x2(variants_dict, title, savepath=None, upsize=256):
    order = [
        ("Original (Grayscale)", "Original"),
        ("Scaled 0.5×", "Scaled 0.5×"),
        ("Contrast +20", "Contrast +20"),
        ("Mirrored (H)", "Mirrored (H)"),
        ("Rotated 45°", "Rotated 45°"),
        ("Translated (+30,+30)", "Translated (+30,+30)")
    ]

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    fig.suptitle(f"Hu Moments — {title}", fontsize=15, fontweight='bold')

    for ax, (label, key) in zip(axes.ravel(), order):
        img = cv2.resize(variants_dict[key], (upsize, upsize), interpolation=cv2.INTER_LINEAR)
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(label, fontsize=11, pad=15)
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"✅ Saved labeled montage: {savepath}")


def print_hu_table(variants_dict, img_title):
    rows = []
    order = ["Original", "Scaled 0.5×", "Contrast +20", "Mirrored (H)", "Rotated 45°", "Translated (+30,+30)"]
    for k in order:
        phi = hu_1to4_on_mask(variants_dict[k])
        rows.append([k, f"{phi[0]:.6f}", f"{phi[1]:.6f}", f"{phi[2]:.6f}", f"{phi[3]:.6f}"])
    df = pd.DataFrame(rows, columns=["Variant", "ϕ1", "ϕ2", "ϕ3", "ϕ4"])
    print(f"\n=== {img_title} — Hu Moments (log-transformed) ===")
    print(df.to_string(index=False))
    return df

def print_distances(variants_dict, img_title):
    base = hu_1to4_on_mask(variants_dict["Original"])
    queries = ["Rotated 45°", "Scaled 0.5×", "Mirrored (H)"]
    rows = []
    for q in queries:
        d = chi_squared(base, hu_1to4_on_mask(variants_dict[q]))
        rows.append([q, f"{d:.6g}"])
    df = pd.DataFrame(rows, columns=["Query Variant", "Chi-squared vs Original"])
    print(f"\n=== {img_title} — 3 Query Distances (vs Original) ===")
    print(df.to_string(index=False))
    return df

#Main
all_loaded = []
for i, fname in enumerate(IMAGE_FILES, start=1):
    path = os.path.join(IMAGE_DIR, fname)
    try:
        img = load_gray(path)
    except FileNotFoundError as e:
        print(f"⚠️ {e}")
        continue

    title = f"Image {i} ({fname})"
    vars_dict = build_variants(img)

    # 1) Print Hu tables
    print_hu_table(vars_dict, title)

    # 2) Print Chi-squared distances
    print_distances(vars_dict, title)

    # 3) Show montage 
    out_path = os.path.join(OUTPUT_DIR, f"img{i}_montage.png")
    montage_3x2(vars_dict, title, savepath=out_path)

    all_loaded.append(img)

# 5x5 PATCH
if all_loaded:
    patch = center_patch(all_loaded[0], 5)
    # make transforms, but keep size 5x5 after each step for fairness
    vars_patch = {k: cv2.resize(v, (5,5), interpolation=cv2.INTER_NEAREST)
                  for k, v in build_variants(patch).items()}

    # Tables
    print_hu_table(vars_patch, "Patch 5×5 (from img1 center)")
    # Distances
    def _hu(v): return hu_1to4_on_mask(v)
    base_p = _hu(vars_patch["Original"])
    rows_p = []
    for q in ["Rotated 45°", "Scaled 0.5×", "Mirrored (H)"]:
        rows_p.append([q, f"{chi_squared(base_p, _hu(vars_patch[q])):.6g}"])
    df_p = pd.DataFrame(rows_p, columns=["Query Variant", "Chi-squared vs Original (Patch)"])
    print("\n=== Patch 5×5 — 3 Query Distances (vs Original) ===")
    print(df_p.to_string(index=False))

    # Montage
    up = {k: cv2.resize(v, (256,256), interpolation=cv2.INTER_NEAREST) for k, v in vars_patch.items()}
    montage_3x2(up, "Patch 5×5 (upsampled)", savepath=os.path.join(OUTPUT_DIR, "patch5x5_montage.png"))
else:
    print("No images loaded; patch step skipped.")
