import numpy as np
import cv2
import matplotlib.pyplot as plt


def laplacian_to_gaussian(x, y, sigma):
    r = x**2 + y**2
    return - (1 / (np.pi * sigma**4)) * (1 - (r / (2 * sigma**2))) * np.exp(-r / (2 * sigma**2))


def gen_log_ker(sigma):
    s = int(np.ceil(9 * sigma))
    if s % 2 == 0:   
        s += 1
    ax = np.arange(-s // 2 + 1., s // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    ker = laplacian_to_gaussian(xx, yy, sigma)
    return ker


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

def zero_crossing_detect(log_img):
    zc = np.zeros(log_img.shape, dtype=np.uint8)
    h, w = log_img.shape

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            neg = log_img[y-1:y+2, x-1:x+2] < 0
            pos = log_img[y-1:y+2, x-1:x+2] > 0
            if neg.any() and pos.any():
                zc[y, x] = 255
                
    return zc

if __name__ == "__main__":
    img = cv2.imread("lena.jpg", cv2.IMREAD_GRAYSCALE)

    sigma = 0.7
    log_img = log_7sigma(img, sigma)
    zc_img = zero_crossing_detect(log_img)

    # Show results
    plt.figure(figsize=(12, 10))

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Original Lena")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(to_u8(log_img), cmap="gray")
    plt.title("LoG Response")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(zc_img, cmap="gray")
    plt.title("Zero Crossings")
    plt.axis("off")
    
    plt.show()