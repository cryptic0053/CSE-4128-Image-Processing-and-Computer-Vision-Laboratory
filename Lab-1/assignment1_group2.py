import sys
import cv2
import numpy as np
import matplotlib
try:
    matplotlib.use("Qt5Agg") 
except Exception:
    pass
import matplotlib.pyplot as plt

#CLI/defaults
IMG_PATH     = "lena.jpg"
SIGMA_SMOOTH = float(sys.argv[2]) if len(sys.argv) >= 3 else 1.0
SIGMA_SHARP  = float(sys.argv[3]) if len(sys.argv) >= 4 else 1.0

def odd_from_sigma(mult, sigma, min_size):
    k = max(int(round(mult * sigma)), min_size)
    return k if k % 2 == 1 else k + 1

def gaussian_5sigma(ch, sigma):
    k = odd_from_sigma(5, sigma, 5)
    return cv2.GaussianBlur(ch, (k, k), sigmaX=sigma, sigmaY=sigma)

def log_7sigma(ch, sigma):
    k = odd_from_sigma(7, sigma, 7)
    blur = cv2.GaussianBlur(ch, (k, k), sigmaX=sigma, sigmaY=sigma)
    return cv2.Laplacian(blur, ddepth=cv2.CV_32F, ksize=3)

def to_u8(x):
    if x.dtype == np.uint8:
        return x
    x = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)
    return np.clip(np.round(x), 0, 255).astype(np.uint8)

#load
bgr = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

#PANELS: RGB
B, G, R = cv2.split(bgr)

#colorized channel views
B_col = cv2.merge([B, np.zeros_like(B), np.zeros_like(B)])
G_col = cv2.merge([np.zeros_like(G), G, np.zeros_like(G)])
R_col = cv2.merge([np.zeros_like(R), np.zeros_like(R), R])

#Gaussian per channel
B_g = to_u8(gaussian_5sigma(B, SIGMA_SMOOTH))
G_g = to_u8(gaussian_5sigma(G, SIGMA_SMOOTH))
R_g = to_u8(gaussian_5sigma(R, SIGMA_SMOOTH))
rgb_gauss = cv2.merge([B_g, G_g, R_g])

#LoG per channel
B_l = to_u8(log_7sigma(B, SIGMA_SHARP))
G_l = to_u8(log_7sigma(G, SIGMA_SHARP))
R_l = to_u8(log_7sigma(R, SIGMA_SHARP))
rgb_log = cv2.merge([B_l, G_l, R_l])

#figure: RGB
plt.figure(figsize=(11, 11))
plt.suptitle("RGB Panels", fontsize=14)

#row 1:original + channels
plt.subplot(3, 4, 1); plt.title("Color image"); plt.imshow(rgb); plt.axis("off")
plt.subplot(3, 4, 2); plt.title("Blue Channel");  plt.imshow(cv2.cvtColor(B_col, cv2.COLOR_BGR2RGB)); plt.axis("off")
plt.subplot(3, 4, 3); plt.title("Green Channel"); plt.imshow(cv2.cvtColor(G_col, cv2.COLOR_BGR2RGB)); plt.axis("off")
plt.subplot(3, 4, 4); plt.title("Red Channel");   plt.imshow(cv2.cvtColor(R_col, cv2.COLOR_BGR2RGB)); plt.axis("off")

#row 2:Gaussian each + merged
plt.subplot(3, 4, 5);  plt.title("Gaussian Blue Channel");  plt.imshow(B_g, cmap="gray"); plt.axis("off")
plt.subplot(3, 4, 6);  plt.title("Gaussian Green Channel"); plt.imshow(G_g, cmap="gray"); plt.axis("off")
plt.subplot(3, 4, 7);  plt.title("Gaussian Red Channel");   plt.imshow(R_g, cmap="gray"); plt.axis("off")
plt.subplot(3, 4, 8);  plt.title("Merged Gaussian");        plt.imshow(cv2.cvtColor(rgb_gauss, cv2.COLOR_BGR2RGB)); plt.axis("off")

#row 3:LoG each + merged
plt.subplot(3, 4, 9);  plt.title("LoG on Blue Channel");   plt.imshow(B_l, cmap="gray"); plt.axis("off")
plt.subplot(3, 4, 10); plt.title("LoG on Green Channel");  plt.imshow(G_l, cmap="gray"); plt.axis("off")
plt.subplot(3, 4, 11); plt.title("LoG on Red Channel");    plt.imshow(R_l, cmap="gray"); plt.axis("off")
plt.subplot(3, 4, 12); plt.title("Merged LoG");            plt.imshow(cv2.cvtColor(rgb_log, cv2.COLOR_BGR2RGB)); plt.axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("assignment1_RGB_panels.png", dpi=220, bbox_inches="tight")

#PANELS:HSV
HSV = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(HSV)

# Gaussian per HSV channel
H_g = to_u8(gaussian_5sigma(H, SIGMA_SMOOTH))
S_g = to_u8(gaussian_5sigma(S, SIGMA_SMOOTH))
V_g = to_u8(gaussian_5sigma(V, SIGMA_SMOOTH))
HSV_g = cv2.merge([H_g, S_g, V_g])
hsv_gauss_bgr = cv2.cvtColor(HSV_g, cv2.COLOR_HSV2BGR)

# LoG per HSV channel
H_l = to_u8(log_7sigma(H, SIGMA_SHARP))
S_l = to_u8(log_7sigma(S, SIGMA_SHARP))
V_l = to_u8(log_7sigma(V, SIGMA_SHARP))
HSV_l = cv2.merge([H_l, S_l, V_l])
hsv_log_bgr = cv2.cvtColor(HSV_l, cv2.COLOR_HSV2BGR)

# optional: visualize H,S,V as grayscale "channels"
H_gray = H
S_gray = S
V_gray = V

# ----- figure: HSV -----
plt.figure(figsize=(11, 11))
plt.suptitle("HSV Panels", fontsize=14)

# row 1: original + H S V channels
plt.subplot(3, 4, 1); plt.title("Color image"); plt.imshow(rgb); plt.axis("off")
plt.subplot(3, 4, 2); plt.title("Hue Channel");        plt.imshow(H_gray, cmap="gray"); plt.axis("off")
plt.subplot(3, 4, 3); plt.title("Saturation Channel"); plt.imshow(S_gray, cmap="gray"); plt.axis("off")
plt.subplot(3, 4, 4); plt.title("Value Channel");      plt.imshow(V_gray, cmap="gray"); plt.axis("off")

# row 2: Gaussian on H,S,V + merged
plt.subplot(3, 4, 5);  plt.title("Gaussian Hue Channel");        plt.imshow(H_g, cmap="gray"); plt.axis("off")
plt.subplot(3, 4, 6);  plt.title("Gaussian Saturation Channel"); plt.imshow(S_g, cmap="gray"); plt.axis("off")
plt.subplot(3, 4, 7);  plt.title("Gaussian Value Channel");      plt.imshow(V_g, cmap="gray"); plt.axis("off")
plt.subplot(3, 4, 8);  plt.title("Merged Gaussian (HSV)");       plt.imshow(cv2.cvtColor(hsv_gauss_bgr, cv2.COLOR_BGR2RGB)); plt.axis("off")

# row 3: LoG on H,S,V + merged
plt.subplot(3, 4, 9);  plt.title("LoG on Hue Channel");          plt.imshow(H_l, cmap="gray"); plt.axis("off")
plt.subplot(3, 4, 10); plt.title("LoG on Saturation Channel");   plt.imshow(S_l, cmap="gray"); plt.axis("off")
plt.subplot(3, 4, 11); plt.title("LoG on Value Channel");        plt.imshow(V_l, cmap="gray"); plt.axis("off")
plt.subplot(3, 4, 12); plt.title("Merged LoG (HSV)");            plt.imshow(cv2.cvtColor(hsv_log_bgr, cv2.COLOR_BGR2RGB)); plt.axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("assignment1_HSV_panels.png", dpi=220, bbox_inches="tight")

plt.show()

print("Saved:", "assignment1_RGB_panels.png", "and", "assignment1_HSV_panels.png")
