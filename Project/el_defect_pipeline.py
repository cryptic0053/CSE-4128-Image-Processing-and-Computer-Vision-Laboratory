#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solar Panel Defect Detection — v5 (robust to texture + grid)

Pipeline
1) Load → optional resize
2) Background removal (rolling-ball; fallback: large Gaussian)
3) CLAHE + bilateral smoothing
4) Crack enhancement: Frangi (line filter) + black-hat + inverted image
5) Threshold: Sauvola + small Otsu union
6) Morphology cleanup
7) Grid suppression: Hough (0°/90°) → dilate → mask out from crack mask
8) Optional notch-filtered Canny for visualization
9) Region descriptors + skeleton length
10) Save intermediates and metrics.json / regions_metrics.csv
"""

import os, json, csv, argparse
import numpy as np
import cv2

from skimage import measure, exposure, morphology, filters, util
from skimage.morphology import remove_small_objects, remove_small_holes, disk, skeletonize
from skimage.color import label2rgb

# ---------------------------- utils ----------------------------

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def imread_gray(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None: raise FileNotFoundError(path)
    if img.ndim == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    if img.max() > 1: img /= 255.0
    return img

def to_u8(img):
    img = np.clip(img, 0, 1)
    return (img * 255.0 + 0.5).astype(np.uint8)

def save_gray(path, img):
    if img.dtype != np.uint8: img = to_u8(img)
    cv2.imwrite(path, img)

def save_bgr(path, img_bgr):
    cv2.imwrite(path, img_bgr)

# -------------------- background / enhancement -----------------

def rolling_ball_bg_sub(img, radius=80):
    """Remove slow illumination with rolling-ball; fallback to large Gaussian."""
    try:
        from skimage.restoration import rolling_ball
        bg = rolling_ball(img, radius=radius)
        out = img - bg
    except Exception:
        k = int(max(3, radius//2)*2+1)
        out = img - cv2.GaussianBlur(img, (k, k), k*0.25)
    out -= out.min()
    out /= (out.max() + 1e-8)
    return out

def clahe(img):
    u8 = to_u8(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    out = clahe.apply(u8).astype(np.float32)/255.0
    return out

def bilateral(img, d=7, sigmaC=25, sigmaS=7):
    u8 = to_u8(img)
    out = cv2.bilateralFilter(u8, d, sigmaC, sigmaS).astype(np.float32)/255.0
    return out

def frangi_enhance(img, sig1=0.8, sig2=2.5, beta=0.5, gamma=15):
    """Enhance line-like (crack) structures."""
    # skimage.filters.frangi expects [0,1]
    f = filters.frangi(img, scale_range=(sig1, sig2),
                       scale_step=2, beta1=beta, beta2=gamma, alpha=0.5, black_ridges=True)
    f = (f - f.min()) / (f.max() - f.min() + 1e-8)
    return f.astype(np.float32)

def notch_filter_fft(img, notches, radius=6):
    h, w = img.shape
    F = np.fft.fftshift(np.fft.fft2(img))
    Y, X = np.ogrid[:h, :w]; cy, cx = h//2, w//2
    mask = np.ones_like(img, dtype=np.float32)
    for dy, dx in notches:
        R1 = np.hypot(Y-(cy+dy), X-(cx+dx))
        R2 = np.hypot(Y-(cy-dy), X-(cx-dx))
        mask *= (R1>radius) * (R2>radius)
    Fn = F * mask
    out = np.abs(np.fft.ifft2(np.fft.ifftshift(Fn)))
    out = (out - out.min()) / (out.max() - out.min() + 1e-8)
    return out.astype(np.float32)

# ------------------------ grid suppression ---------------------

def hough_grid_mask(img_edges, angle_dev_deg=12, dilate=3):
    """Detect near-vertical/horizontal lines and return a dilated mask."""
    lines = cv2.HoughLinesP(img_edges, 1, np.pi/180, threshold=140,
                            minLineLength=100, maxLineGap=6)
    mask = np.zeros_like(img_edges, dtype=np.uint8)
    if lines is None: return mask
    for x1,y1,x2,y2 in lines[:,0,:]:
        ang = (np.degrees(np.arctan2(y2-y1, x2-x1)) + 180) % 180
        if (abs(ang-0) < angle_dev_deg) or (abs(ang-90) < angle_dev_deg):
            cv2.line(mask, (x1,y1), (x2,y2), 255, 2)
    if dilate>0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate,dilate))
        mask = cv2.dilate(mask, k, iterations=1)
    return mask

# --------------------------- main ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--resize", type=int, default=None)

    # background / enhancement
    ap.add_argument("--rb_radius", type=int, default=80)
    ap.add_argument("--bilateral_d", type=int, default=7)
    ap.add_argument("--bilateral_sigmaC", type=float, default=25)
    ap.add_argument("--bilateral_sigmaS", type=float, default=7)
    ap.add_argument("--frangi_s1", type=float, default=0.8)
    ap.add_argument("--frangi_s2", type=float, default=2.5)

    # threshold
    ap.add_argument("--sauvola_w", type=int, default=31)
    ap.add_argument("--sauvola_k", type=float, default=0.25)

    # morphology
    ap.add_argument("--open_kernel", type=int, default=3)
    ap.add_argument("--close_kernel", type=int, default=3)
    ap.add_argument("--min_area", type=int, default=40)

    # grid suppression + edges
    ap.add_argument("--canny_low", type=int, default=25)
    ap.add_argument("--canny_high", type=int, default=90)
    ap.add_argument("--grid_angle_dev", type=float, default=14.0)
    ap.add_argument("--grid_dilate", type=int, default=3)

    # notch (optional used only for edge viz)
    ap.add_argument("--notch", type=str, default="yes", choices=["yes","no"])
    ap.add_argument("--notch_radius", type=int, default=6)
    ap.add_argument("--notch_offsets", type=int, nargs="*", default=[0,35, 0,-35, 35,0, -35,0])

    # region filters
    ap.add_argument("--ecc_min", type=float, default=0.85)
    ap.add_argument("--solidity_max", type=float, default=0.95)

    args = ap.parse_args()
    ensure_dir(args.outdir)

    # load
    img0 = imread_gray(args.image)
    if args.resize:
        h,w = img0.shape
        s = args.resize/max(h,w)
        if s<1: img0 = cv2.resize(img0, (int(w*s), int(h*s)), cv2.INTER_AREA)
    save_gray(os.path.join(args.outdir, "00_input.png"), img0)

    # 1) background removal + CLAHE + bilateral
    img_rb = rolling_ball_bg_sub(img0, radius=args.rb_radius)
    save_gray(os.path.join(args.outdir, "00a_rollingball.png"), img_rb)

    img_eq = clahe(img_rb)
    save_gray(os.path.join(args.outdir, "01_clahe.png"), img_eq)

    img_smooth = bilateral(img_eq, args.bilateral_d, args.bilateral_sigmaC, args.bilateral_sigmaS)
    save_gray(os.path.join(args.outdir, "02_bilateral.png"), img_smooth)

    # 2) crack enhancement
    f_frangi = frangi_enhance(img_smooth, args.frangi_s1, args.frangi_s2)
    save_gray(os.path.join(args.outdir, "03_frangi.png"), f_frangi)

    # black-hat on smoothed image
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    blackhat = cv2.morphologyEx(to_u8(img_smooth), cv2.MORPH_BLACKHAT, se).astype(np.float32)/255.0
    save_gray(os.path.join(args.outdir, "03b_blackhat.png"), blackhat)

    inv = 1.0 - img_smooth
    crack_score = 0.55*f_frangi + 0.35*blackhat + 0.10*inv
    crack_score = (crack_score - crack_score.min())/(crack_score.max()-crack_score.min()+1e-8)
    save_gray(os.path.join(args.outdir, "04_crack_score.png"), crack_score)

    # 3) threshold: Sauvola + small Otsu union
    w = args.sauvola_w + (args.sauvola_w % 2 == 0)
    thr_s = filters.threshold_sauvola(crack_score, window_size=int(w), k=args.sauvola_k)
    mask_s = (crack_score > thr_s)

    thr_o, _ = cv2.threshold(to_u8(crack_score), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask_o = (to_u8(crack_score) > thr_o)

    mask = (mask_s | mask_o)

    # 4) morphology cleanup
    mask = morphology.binary_opening(mask, morphology.disk(args.open_kernel))
    mask = morphology.binary_closing(mask, morphology.disk(args.close_kernel))
    mask = remove_small_objects(mask, args.min_area)
    mask = remove_small_holes(mask, area_threshold=args.min_area)
    save_gray(os.path.join(args.outdir, "05_mask_pre_grid.png"), mask.astype(np.float32))

    # 5) build grid mask from edges and remove it
    if args.notch == "yes":
        offs = args.notch_offsets
        notches = [(offs[i],offs[i+1]) for i in range(0,len(offs),2)]
        img_edges_base = notch_filter_fft(img_smooth, notches, radius=args.notch_radius)
        img_edges_u8 = cv2.Canny(to_u8(img_edges_base), args.canny_low, args.canny_high)
        save_gray(os.path.join(args.outdir, "06a_notched_for_edges.png"), img_edges_base)
    else:
        img_edges_u8 = cv2.Canny(to_u8(img_smooth), args.canny_low, args.canny_high)
    save_gray(os.path.join(args.outdir, "06_edges.png"), img_edges_u8)

    grid_mask = hough_grid_mask(img_edges_u8, angle_dev_deg=args.grid_angle_dev, dilate=args.grid_dilate)
    save_gray(os.path.join(args.outdir, "06b_grid_mask.png"), grid_mask)

    mask_wo_grid = mask & (grid_mask==0)
    save_gray(os.path.join(args.outdir, "07_mask_wo_grid.png"), mask_wo_grid.astype(np.float32))

    # 6) region filtering by geometry (keep thin/elongated, low solidity)
    lab = measure.label(mask_wo_grid)
    props = measure.regionprops(lab)
    keep = np.zeros_like(mask_wo_grid, dtype=bool)
    regions = []
    rid = 1
    for p in props:
        if p.area < args.min_area: continue
        ecc = getattr(p, "eccentricity", 0.0)
        sol = getattr(p, "solidity", 1.0)
        # cracks tend to be elongated (high ecc) and not very solid
        if ecc >= args.ecc_min and sol <= args.solidity_max:
            keep[p.coords[:,0], p.coords[:,1]] = True
            regions.append({
                "region_id": rid,
                "area_px": int(p.area),
                "perimeter_px": float(p.perimeter),
                "eccentricity": float(ecc),
                "solidity": float(sol),
                "major_axis_length": float(getattr(p, "major_axis_length", np.nan)),
                "minor_axis_length": float(getattr(p, "minor_axis_length", np.nan)),
                "orientation_rad": float(getattr(p, "orientation", np.nan)),
                "bbox": [int(v) for v in p.bbox],
                "centroid_yx": [float(p.centroid[0]), float(p.centroid[1])],
            })
            rid += 1

    mask_final = morphology.binary_opening(keep, morphology.disk(1))
    save_gray(os.path.join(args.outdir, "08_mask_final.png"), mask_final.astype(np.float32))

    # 7) skeleton length
    sk = skeletonize(mask_final)
    crack_len_px = float(sk.sum())
    save_gray(os.path.join(args.outdir, "09_skeleton.png"), sk.astype(np.float32))

    # 8) overlay for visualization
    overlay = label2rgb(measure.label(mask_final), image=img0, alpha=0.35, bg_label=0)
    overlay_bgr = cv2.cvtColor((overlay*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    save_bgr(os.path.join(args.outdir, "10_overlay.png"), overlay_bgr)

    # 9) metrics
    summary = {
        "input_image": os.path.abspath(args.image),
        "num_regions": len(regions),
        "crack_length_skeleton_px": crack_len_px,
        "params": vars(args)
    }
    with open(os.path.join(args.outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "regions": regions}, f, indent=2)

    if regions:
        with open(os.path.join(args.outdir, "regions_metrics.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(regions[0].keys()))
            writer.writeheader()
            writer.writerows(regions)

    print("Done. Outputs →", os.path.abspath(args.outdir))


if __name__ == "__main__":
    main()
