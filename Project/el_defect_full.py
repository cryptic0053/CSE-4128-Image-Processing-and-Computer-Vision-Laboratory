#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Electroluminescence (EL) Solar Panel Defect Detection — Full Classical CV Pipeline

Implements ALL requested components:
- Convolution: Gaussian denoising
- Segmentation: Adaptive Gaussian thresholding (cell boundaries)
- Edge Detection: Canny (cracks/line defects)
- Thresholding: Binary (Otsu + Sauvola, combined per-pixel)
- Morphology: open/close + small-object removal
- Frequency Domain: FFT magnitude + optional notch filtering for periodic patterns
- Region Descriptors: area, perimeter, eccentricity, axes, orientation, centroid
- Crack Length: skeleton length
- Hough Line Transform: precise grid/busbar detection (and suppression from defects)
- Busbar Integrity: continuity score from Canny responses along vertical Hough lines
- Sub-millimeter precision: pass --mm_per_px to convert pixel-based metrics

Outputs to --outdir:
  00_input.png
  01_clahe.png
  02_gaussian.png
  03_fft_mag.png
  03b_notch.png            (if --notch yes)
  04_canny.png
  05_cell_boundaries.png
  06_grid_mask.png
  07_crack_score.png
  08_defect_mask_raw.png
  09_defect_mask_clean.png
  10_overlay.png
  11_skeleton.png
  metrics.json
  regions_metrics.csv

Example (PowerShell):
  python .\el_defect_full.py `
    --image ".\ARTS_00001_r6_c2.png" `
    --outdir ".\results_full" `
    --gauss_ksize 5 --gauss_sigma 1.0 `
    --adaptive_block 35 --adaptive_C 5 `
    --canny_low 25 --canny_high 90 `
    --hough_threshold 140 --hough_minlen 110 --hough_gap 8 `
    --angle_dev 15 --grid_dilate 3 `
    --open_kernel 3 --close_kernel 3 --min_area 60 `
    --notch yes --notch_radius 6 --notch_offsets 0 35 0 -35 35 0 -35 0
"""

import os, json, csv, argparse
from typing import List, Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import exposure, feature, measure, morphology
from skimage.filters import threshold_otsu, threshold_sauvola, sato
from skimage.color import label2rgb, gray2rgb


# ---------------------------- I/O helpers ----------------------------

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def imread_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    return img

def to_uint8(img: np.ndarray) -> np.ndarray:
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)

def save_gray(path: str, img: np.ndarray) -> None:
    if img.dtype != np.uint8:
        img = to_uint8(img)
    cv2.imwrite(path, img)

def save_fig_gray(path: str, img: np.ndarray) -> None:
    plt.figure(figsize=(7, 7))
    plt.imshow(img, cmap="gray", vmin=0, vmax=1 if img.dtype!=np.uint8 else None)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()


# ----------------------- Core processing blocks -----------------------

def clahe(img: np.ndarray, clip: float = 2.0, tiles: Tuple[int,int]=(8,8)) -> np.ndarray:
    u8 = to_uint8(img)
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles)
    out = c.apply(u8).astype(np.float32) / 255.0
    return out

def gaussian(img: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    u8 = to_uint8(img)
    g = cv2.GaussianBlur(u8, (ksize, ksize), sigmaX=sigma)
    return g.astype(np.float32) / 255.0

def adaptive_gaussian_threshold(img: np.ndarray, block_size: int = 35, C: int = 5) -> np.ndarray:
    if block_size % 2 == 0: block_size += 1
    block_size = max(3, block_size)
    u8 = to_uint8(img)
    th = cv2.adaptiveThreshold(u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, block_size, C)
    return (th > 0).astype(np.float32)

def canny_edges(img: np.ndarray, low: int, high: int) -> np.ndarray:
    return cv2.Canny(to_uint8(img), threshold1=low, threshold2=high)

def black_hat(img: np.ndarray, radius: int = 3) -> np.ndarray:
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    u8 = to_uint8(img)
    bh = cv2.morphologyEx(u8, cv2.MORPH_BLACKHAT, se)
    return (bh.astype(np.float32) / 255.0)

def fft_magnitude(img: np.ndarray) -> np.ndarray:
    F = np.fft.fftshift(np.fft.fft2(img))
    mag = np.log1p(np.abs(F))
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    return mag

def notch_filter_fft(img: np.ndarray, offsets: List[Tuple[int,int]], radius: int = 6) -> np.ndarray:
    h, w = img.shape
    F = np.fft.fftshift(np.fft.fft2(img))
    Y, X = np.ogrid[:h, :w]
    cy, cx = h//2, w//2
    mask = np.ones((h, w), dtype=np.float32)
    for dy, dx in offsets:
        r1 = np.sqrt((Y - (cy + dy))**2 + (X - (cx + dx))**2)
        r2 = np.sqrt((Y - (cy - dy))**2 + (X - (cx - dx))**2)
        mask *= (r1 > radius).astype(np.float32)
        mask *= (r2 > radius).astype(np.float32)
    Fn = F * mask
    out = np.fft.ifft2(np.fft.ifftshift(Fn))
    out = np.abs(out)
    out = (out - out.min()) / (out.max() - out.min() + 1e-8)
    return out

def hough_lines(edges: np.ndarray, rho=1, theta=np.pi/180, threshold=120,
                min_line_len=80, max_line_gap=8) -> np.ndarray:
    lines = cv2.HoughLinesP(edges, rho=rho, theta=theta, threshold=threshold,
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    if lines is None: return np.empty((0,4), dtype=int)
    return lines.reshape(-1,4)

def make_grid_mask_from_hough(lines: np.ndarray, img_shape: Tuple[int,int],
                              angle_dev_deg: float = 15.0,
                              exclude_angles_deg: Tuple[float,float]=(0.0, 90.0),
                              dilate: int = 3) -> np.ndarray:
    H, W = img_shape
    grid = np.zeros((H, W), dtype=np.uint8)
    for x1,y1,x2,y2 in lines:
        ang = (np.degrees(np.arctan2(y2 - y1, x2 - x1)) + 180) % 180
        if any(abs(ang - a) < angle_dev_deg for a in exclude_angles_deg):
            cv2.line(grid, (x1,y1), (x2,y2), 255, 2)
    if dilate > 0:
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate, dilate))
        grid = cv2.dilate(grid, se)
    return grid > 0

def morphology_cleanup(mask: np.ndarray, open_k=3, close_k=3, min_area=60) -> np.ndarray:
    u8 = (mask.astype(np.uint8) * 255)
    if open_k > 0:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, se)
    if close_k > 0:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, se)
    bin_mask = u8 > 0
    bin_mask = morphology.remove_small_objects(bin_mask, min_size=int(min_area))
    return bin_mask.astype(np.float32)

def crack_score(img_den: np.ndarray) -> np.ndarray:
    """Combine inverted intensity + black-hat + Sato ridge to emphasize cracks."""
    inv = 1.0 - img_den
    bh  = black_hat(img_den, radius=3)
    rid = sato(img_den, sigmas=np.linspace(1, 3, 7), black_ridges=True)
    rid = (rid - rid.min()) / (rid.max() - rid.min() + 1e-8)
    score = 0.5*inv + 0.3*bh + 0.2*rid
    score = (score - score.min()) / (score.max() - score.min() + 1e-8)
    return score

def skeleton_length(mask: np.ndarray) -> Tuple[float, np.ndarray]:
    sk = morphology.skeletonize(mask.astype(bool))
    return float(sk.sum()), sk


# ----------------------------- Main ----------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--resize", type=int, default=None, help="Max dimension; keep aspect (optional)")

    # Filtering / segmentation params
    ap.add_argument("--gauss_ksize", type=int, default=5)
    ap.add_argument("--gauss_sigma", type=float, default=1.0)
    ap.add_argument("--adaptive_block", type=int, default=35)
    ap.add_argument("--adaptive_C", type=int, default=5)
    ap.add_argument("--canny_low", type=int, default=25)
    ap.add_argument("--canny_high", type=int, default=90)

    # Hough (grid/busbar)
    ap.add_argument("--hough_threshold", type=int, default=140)
    ap.add_argument("--hough_minlen", type=int, default=110)
    ap.add_argument("--hough_gap", type=int, default=8)
    ap.add_argument("--angle_dev", type=float, default=15.0)
    ap.add_argument("--grid_dilate", type=int, default=3)

    # Morphology
    ap.add_argument("--open_kernel", type=int, default=3)
    ap.add_argument("--close_kernel", type=int, default=3)
    ap.add_argument("--min_area", type=int, default=60)

    # FFT / notch
    ap.add_argument("--notch", type=str, default="no", choices=["no","yes"])
    ap.add_argument("--notch_radius", type=int, default=6)
    ap.add_argument("--notch_offsets", type=int, nargs="*", default=[0,35, 0,-35, 35,0, -35,0])

    # Units
    ap.add_argument("--mm_per_px", type=float, default=None, help="Pixel size (mm/px) for sub-mm metrics")

    args = ap.parse_args()
    ensure_dir(args.outdir)

    # 0) Load + optional resize
    img0 = imread_gray(args.image)
    if args.resize:
        H, W = img0.shape
        s = args.resize / max(H, W)
        if s < 1.0:
            img0 = cv2.resize(img0, (int(W*s), int(H*s)), interpolation=cv2.INTER_AREA)

    save_gray(os.path.join(args.outdir, "00_input.png"), img0)

    # 1) CLAHE + Gaussian (Convolution)
    img1 = clahe(img0, clip=2.0, tiles=(8,8))
    save_gray(os.path.join(args.outdir, "01_clahe.png"), img1)
    img2 = gaussian(img1, ksize=args.gauss_ksize, sigma=args.gauss_sigma)
    save_gray(os.path.join(args.outdir, "02_gaussian.png"), img2)

    # 2) Frequency domain (FFT magnitude + optional notch)
    fft_mag = fft_magnitude(img2)
    save_fig_gray(os.path.join(args.outdir, "03_fft_mag.png"), fft_mag)
    if args.notch == "yes":
        offs = args.notch_offsets
        if len(offs) % 2 != 0:
            raise ValueError("--notch_offsets must be dy dx pairs")
        pairs = [(offs[i], offs[i+1]) for i in range(0, len(offs), 2)]
        img2n = notch_filter_fft(img2, offsets=pairs, radius=args.notch_radius)
        save_gray(os.path.join(args.outdir, "03b_notch.png"), img2n)
    else:
        img2n = img2

    # 3) Edge detection (Canny) for Hough/grid lines
    edges = canny_edges(img2n, args.canny_low, args.canny_high)
    save_gray(os.path.join(args.outdir, "04_canny.png"), edges)

    # 4) Segmentation of cell boundaries (adaptive Gaussian threshold)
    cells = adaptive_gaussian_threshold(img2n, args.adaptive_block, args.adaptive_C)
    save_gray(os.path.join(args.outdir, "05_cell_boundaries.png"), cells)

    # 5) Hough lines → build grid mask (suppress ~0° / ~90°)
    lines = hough_lines(edges, threshold=args.hough_threshold,
                        min_line_len=args.hough_minlen, max_line_gap=args.hough_gap)
    grid_mask = make_grid_mask_from_hough(lines, img2n.shape,
                                          angle_dev_deg=args.angle_dev,
                                          exclude_angles_deg=(0.0, 90.0),
                                          dilate=args.grid_dilate)
    save_gray(os.path.join(args.outdir, "06_grid_mask.png"), grid_mask.astype(np.float32))

    # 6) Crack score → thresholding (binary classification)
    cscore = crack_score(img2n)
    save_gray(os.path.join(args.outdir, "07_crack_score.png"), cscore)

    # --- FIXED: combine global Otsu (scalar) with local Sauvola (image) per-pixel ---
    t_otsu = threshold_otsu(cscore)  # scalar
    t_sauv = threshold_sauvola(cscore, window_size=max(25, args.adaptive_block), k=0.2)  # array
    t_combined = np.maximum(t_sauv, t_otsu)  # per-pixel stricter threshold
    mask_raw = (cscore > t_combined)
    save_gray(os.path.join(args.outdir, "08_defect_mask_raw.png"), mask_raw.astype(np.float32))

    # Subtract grid/busbar lines; clean with morphology
    mask_ng = mask_raw & (~grid_mask)
    mask = morphology_cleanup(mask_ng, open_k=args.open_kernel,
                              close_k=args.close_kernel, min_area=args.min_area)
    save_gray(os.path.join(args.outdir, "09_defect_mask_clean.png"), mask.astype(np.float32))

    # 7) Region descriptors
    lab = measure.label(mask.astype(bool), connectivity=2)
    props = measure.regionprops(lab)
    regions = []
    for i,p in enumerate(props, 1):
        row = {
            "region_id": i,
            "area_px": int(p.area),
            "perimeter_px": float(p.perimeter),
            "eccentricity": float(getattr(p, "eccentricity", np.nan)),
            "major_axis_length_px": float(getattr(p, "major_axis_length", np.nan)),
            "minor_axis_length_px": float(getattr(p, "minor_axis_length", np.nan)),
            "orientation_rad": float(getattr(p, "orientation", np.nan)),
            "bbox": [int(v) for v in p.bbox],
            "centroid_yx": [float(p.centroid[0]), float(p.centroid[1])],
        }
        regions.append(row)

    # 8) Crack length via skeleton
    crack_len_px, skel = skeleton_length(mask)
    save_gray(os.path.join(args.outdir, "11_skeleton.png"), skel.astype(np.float32))

    # 9) Busbar integrity score (continuity along near-vertical Hough lines)
    busbar_scores = []
    H, W = img2n.shape
    for x1,y1,x2,y2 in lines:
        ang = (np.degrees(np.arctan2(y2 - y1, x2 - x1)) + 180) % 180
        if abs(ang - 90.0) < args.angle_dev:  # vertical-ish
            length = max(1, int(np.hypot(x2 - x1, y2 - y1)))
            xs = np.linspace(x1, x2, length).astype(int)
            ys = np.linspace(y1, y2, length).astype(int)
            xs = np.clip(xs, 0, W-1); ys = np.clip(ys, 0, H-1)
            loc = np.zeros_like(edges, dtype=bool)
            loc[ys, xs] = True
            loc = morphology.binary_dilation(loc, morphology.disk(1))
            score = float((edges > 0)[loc].mean()) if loc.any() else 0.0
            busbar_scores.append({"x1":int(x1),"y1":int(y1),"x2":int(x2),"y2":int(y2),
                                  "continuity_score": score})

    # 10) Overlay visualization
    base = gray2rgb(to_uint8(img2n))
    overlay = base.copy()
    overlay[...,0] = np.maximum(overlay[...,0], (mask*255).astype(np.uint8))  # red = defects
    cv2.imwrite(os.path.join(args.outdir, "10_overlay.png"), overlay[:,:,::-1])  # RGB->BGR

    # 11) Metrics JSON / CSV
    summary = {
        "input_image": os.path.abspath(args.image),
        "num_regions": len(regions),
        "crack_length_px": crack_len_px,
        "hough_lines_total": int(len(lines)),
        "busbar_integrity": busbar_scores,
        "params": vars(args),
    }
    if args.mm_per_px is not None:
        mmpp = float(args.mm_per_px)
        summary["crack_length_mm"] = crack_len_px * mmpp

    with open(os.path.join(args.outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "regions": regions}, f, indent=2)

    if regions:
        keys = list(regions[0].keys())
        if args.mm_per_px is not None:
            keys = keys + ["area_mm2", "perimeter_mm", "major_axis_length_mm", "minor_axis_length_mm"]
        with open(os.path.join(args.outdir, "regions_metrics.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in regions:
                if args.mm_per_px is not None:
                    mmpp = float(args.mm_per_px)
                    r_out = dict(r)
                    r_out["area_mm2"] = r["area_px"] * (mmpp**2)
                    r_out["perimeter_mm"] = r["perimeter_px"] * mmpp
                    r_out["major_axis_length_mm"] = r["major_axis_length_px"] * mmpp
                    r_out["minor_axis_length_mm"] = r["minor_axis_length_px"] * mmpp
                    w.writerow(r_out)
                else:
                    w.writerow(r)

    print("Done. Outputs saved to:", os.path.abspath(args.outdir))


if __name__ == "__main__":
    main()
