import io
import numpy as np
import cv2
import streamlit as st
import matplotlib.pyplot as plt

from skimage import morphology, measure
from skimage.filters import threshold_otsu, threshold_sauvola, sato

# -----------------------
# Small helpers
# -----------------------
def to_float01(img):
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    return img

def to_uint8(img):
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)

def imshow_gray(img, use_uint8=False):
    """Return a PNG bytes image for display in Streamlit."""
    if use_uint8:
        disp = img if img.dtype == np.uint8 else to_uint8(img)
    else:
        disp = to_uint8(img)
    fig = plt.figure(figsize=(4,4))
    plt.imshow(disp, cmap="gray", vmin=0, vmax=255)
    plt.axis('off')
    buf = io.BytesIO()
    plt.tight_layout(pad=0)
    fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf

def clahe(img, clip=2.0, tiles=(8,8)):
    u8 = to_uint8(img)
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles)
    out = c.apply(u8).astype(np.float32) / 255.0
    return out

def gaussian(img, ksize=5, sigma=1.0):
    u8 = to_uint8(img)
    g = cv2.GaussianBlur(u8, (ksize, ksize), sigmaX=sigma)
    return g.astype(np.float32) / 255.0

def canny(img, low, high):
    return cv2.Canny(to_uint8(img), low, high)

def adaptive_gaussian_threshold(img, block=31, C=6):
    if block % 2 == 0:
        block += 1
    u8 = to_uint8(img)
    th = cv2.adaptiveThreshold(u8, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, block, C)
    return (th > 0).astype(np.float32)

def fft_magnitude(img):
    F = np.fft.fftshift(np.fft.fft2(img))
    mag = np.log1p(np.abs(F))
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    return mag

def notch_filter_fft(img, offsets, radius=6):
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

def hough_lines(edges, rho=1, theta=np.pi/180, threshold=140, min_line_len=110, max_line_gap=8):
    lines = cv2.HoughLinesP(edges, rho=rho, theta=theta, threshold=threshold,
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    if lines is None:
        return np.empty((0,4), dtype=int)
    return lines.reshape(-1,4)

def make_grid_mask_from_hough(lines, shape, angle_dev_deg=12.0, exclude=(0.0, 90.0), dilate=3):
    H, W = shape
    grid = np.zeros((H, W), dtype=np.uint8)
    for x1,y1,x2,y2 in lines:
        ang = (np.degrees(np.arctan2(y2-y1, x2-x1)) + 180) % 180
        if any(abs(ang - a) < angle_dev_deg for a in exclude):
            cv2.line(grid, (x1,y1), (x2,y2), 255, 2)
    if dilate > 0:
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate, dilate))
        grid = cv2.dilate(grid, se)
    return grid > 0

def black_hat(img, radius=3):
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    u8 = to_uint8(img)
    bh = cv2.morphologyEx(u8, cv2.MORPH_BLACKHAT, se)
    return bh.astype(np.float32) / 255.0

def crack_score(img_den):
    inv = 1.0 - img_den
    bh  = black_hat(img_den, radius=3)
    rid = sato(img_den, sigmas=np.linspace(1,3,7), black_ridges=True)
    rid = (rid - rid.min()) / (rid.max() - rid.min() + 1e-8)
    score = 0.5*inv + 0.3*bh + 0.2*rid
    score = (score - score.min()) / (score.max() - score.min() + 1e-8)
    return score

def morphology_cleanup(mask, open_k=3, close_k=3, min_area=120):
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

def skeletonize_len(mask):
    sk = morphology.skeletonize(mask.astype(bool))
    return float(sk.sum()), sk

def overlay_red(gray01, mask01):
    base = cv2.cvtColor(to_uint8(gray01), cv2.COLOR_GRAY2BGR)  # BGR
    red = base.copy()
    red[:,:,2] = np.maximum(red[:,:,2], (mask01*255).astype(np.uint8))  # red channel
    return red

def classify_severity(crack_len_px, H, W):
    # normalized crack length (heuristic); tune as needed
    ratio = crack_len_px / float(H*W)
    if ratio < 0.002:
        return "Low", ratio
    elif ratio < 0.01:
        return "Medium", ratio
    else:
        return "High", ratio

# --------------------------------------------
# Streamlit UI
# --------------------------------------------
st.set_page_config(page_title="EL Defect Viewer", layout="wide")
st.title("Electroluminescence (EL) Defect Viewer")

with st.sidebar:
    st.header("Parameters")
    gauss_ksize  = st.slider("Gaussian ksize", 3, 11, 5, step=2)
    gauss_sigma  = st.slider("Gaussian sigma", 0.5, 3.0, 1.0, step=0.1)
    ad_block     = st.slider("Adaptive block", 15, 71, 31, step=2)
    ad_C         = st.slider("Adaptive C", 0, 15, 6, step=1)
    canny_low    = st.slider("Canny low", 5, 100, 25, step=1)
    canny_high   = st.slider("Canny high", 20, 200, 90, step=5)
    hthres       = st.slider("Hough threshold", 50, 250, 150, step=10)
    hminlen      = st.slider("Hough min length", 20, 300, 130, step=5)
    hgap         = st.slider("Hough max gap", 0, 30, 8, step=1)
    angle_dev    = st.slider("Grid angle dev (deg)", 5, 30, 12, step=1)
    grid_dilate  = st.slider("Grid dilate", 0, 7, 4, step=1)
    open_k       = st.slider("Open kernel", 0, 11, 3, step=1)
    close_k      = st.slider("Close kernel", 0, 11, 3, step=1)
    min_area     = st.slider("Min region area (px)", 0, 1000, 120, step=10)
    use_notch    = st.checkbox("Use FFT notch filter", value=True)
    notch_radius = st.slider("Notch radius", 2, 12, 6, step=1)
    offsets_str  = st.text_input("Notch offsets (dy,dx pairs)",
                    value="0 35, 0 -35, 35 0, -35 0")

uploaded = st.file_uploader("Upload EL image (PNG/JPG/BMP)", type=['png','jpg','jpeg','bmp','tif','tiff'])

if uploaded is not None:
    # Read
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img0 = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    if img0 is None:
        st.error("Couldn't read the image. Try another file.")
        st.stop()
    if img0.ndim == 3:
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    img0 = to_float01(img0)

    # 1) CLAHE + Gaussian
    img1 = clahe(img0, clip=2.0, tiles=(8,8))
    img2 = gaussian(img1, ksize=gauss_ksize, sigma=gauss_sigma)

    # 2) FFT
    fft_mag = fft_magnitude(img2)

    # 2b) Notch
    img2n = img2
    if use_notch:
        # parse offsets from text: "0 35, 0 -35, 35 0, -35 0"
        pairs = []
        try:
            parts = [p.strip() for p in offsets_str.split(",")]
            for p in parts:
                dy, dx = p.split()
                pairs.append((int(dy), int(dx)))
        except Exception:
            st.warning("Offsets parse failed; using default [(0,35),(0,-35),(35,0),(-35,0)]")
            pairs = [(0,35),(0,-35),(35,0),(-35,0)]
        img2n = notch_filter_fft(img2, offsets=pairs, radius=notch_radius)

    # 3) Canny for grid lines + 4) adaptive threshold for cells (optional viz)
    edges = canny(img2n, canny_low, canny_high)
    cells = adaptive_gaussian_threshold(img2n, ad_block, ad_C)

    # 5) Hough grid mask
    lines = hough_lines(edges, threshold=hthres, min_line_len=hminlen, max_line_gap=hgap)
    grid_mask = make_grid_mask_from_hough(lines, img2n.shape, angle_dev_deg=angle_dev,
                                          exclude=(0.0, 90.0), dilate=grid_dilate)

    # 6) Crack score & thresholding
    cscore = crack_score(img2n)
    t_otsu = threshold_otsu(cscore)
    t_sauv = threshold_sauvola(cscore, window_size=max(25, ad_block), k=0.2)
    t_comb = np.maximum(t_sauv, t_otsu)
    mask_raw = (cscore > t_comb)

    # remove grid; clean
    mask_ng = mask_raw & (~grid_mask)
    mask = morphology_cleanup(mask_ng, open_k=open_k, close_k=close_k, min_area=min_area)

    # 7) Regions & skeleton
    lab = measure.label(mask.astype(bool), connectivity=2)
    props = measure.regionprops(lab)
    crack_len_px, skel = skeletonize_len(mask)

    # overlay
    overlay = overlay_red(img2n, mask)

    # severity
    H, W = img2n.shape
    sev, ratio = classify_severity(crack_len_px, H, W)

    # -----------------------
    # Display
    # -----------------------
    st.subheader("Results")
    c1, c2, c3 = st.columns(3, gap="small")
    with c1:
        st.caption("Input")
        st.image(imshow_gray(img0), use_column_width=True)
        st.caption("CLAHE")
        st.image(imshow_gray(img1), use_column_width=True)
        st.caption("Gaussian")
        st.image(imshow_gray(img2), use_column_width=True)

    with c2:
        st.caption("FFT magnitude")
        st.image(imshow_gray(fft_mag), use_column_width=True)
        st.caption("Canny edges")
        st.image(imshow_gray(edges, use_uint8=True), use_column_width=True)
        st.caption("Grid mask")
        st.image(imshow_gray(grid_mask.astype(np.float32)), use_column_width=True)

    with c3:
        st.caption("Crack score map")
        st.image(imshow_gray(cscore), use_column_width=True)
        st.caption("Defect mask (clean)")
        st.image(imshow_gray(mask.astype(np.float32)), use_column_width=True)
        st.caption("Overlay (red = defects)")
        st.image(overlay[..., ::-1], use_column_width=True)  # BGR->RGB

    st.caption("Skeleton")
    st.image(imshow_gray(skel.astype(np.float32)), use_column_width=True)

    st.markdown("---")
    st.subheader("Metrics")
    st.write(f"- Image size: **{W}×{H}**")
    st.write(f"- Number of regions: **{len(props)}**")
    st.write(f"- Crack length (px): **{int(crack_len_px)}**")
    st.write(f"- Normalized crack length: **{ratio:.5f}**")
    st.write(f"- **Severity**: :red[**{sev}**]  (heuristic by normalized crack length)")
    st.write(f"- Hough lines (total): **{len(lines)}**")

else:
    st.info("Upload an EL image to run the pipeline.")
