import sys
import cv2
import numpy as np


IMG_PATH ="box.jpg"

#Arbitrary 2-D kernel

Kernel_1 = np.array([
    [ 2,  1,  0, -1, -2],
    [ 1,  3,  1, -2, -3],
    [ 0,  1,  0, -1,  0],
    [-1, -2, -1,  2,  3],
    [-2, -3,  0,  3,  5]
], dtype=np.float64)

#Helpers
def conv2d(img, K):
    return cv2.filter2D(img.astype(np.float64), ddepth=cv2.CV_64F, kernel=K)

def conv_separable(img, X, Y):
    X = np.asarray(X, dtype=np.float64).reshape((-1, 1))  # column vector
    Y = np.asarray(Y, dtype=np.float64).reshape((1, -1))  # row vector
    t = cv2.filter2D(img.astype(np.float64), ddepth=cv2.CV_64F, kernel=X)
    o = cv2.filter2D(t, ddepth=cv2.CV_64F, kernel=Y)
    return o

def opcount_full(H, W, kh, kw):
    mult = H * W * kh * kw
    add  = H * W * (kh * kw - 1)
    return mult, add

def opcount_sep(H, W, kh, kw):
    mult = H * W * kh + H * W * kw
    add  = H * W * (kh - 1) + H * W * (kw - 1)
    return mult, add

def to_u8(arr):
    return cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

#Load grayscale image
img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
H, W = img.shape
kh, kw = Kernel_1.shape

#(a) Full 2-D convolution
out_full = conv2d(img, Kernel_1)
m_full, a_full = opcount_full(H, W, kh, kw)

print("(a) Full 2-D Convolution")
print("Kernel_1 ({}x{}):\n{}".format(kh, kw, Kernel_1))
print(f"Image size: {H}x{W}")
print(f"Operation count (naive): mult={m_full:,}  add={a_full:,}")

#(b) Exact separable (only if rank==1)
rank = np.linalg.matrix_rank(Kernel_1)
has_exact_sep = (rank == 1)

if has_exact_sep:
    Ue, Se, Vte = np.linalg.svd(Kernel_1, full_matrices=False)
    s0e, u0e, v0e = Se[0], Ue[:, 0], Vte[0, :]
    X_exact = np.sqrt(s0e) * u0e
    Y_exact = np.sqrt(s0e) * v0e
    out_sep_exact = conv_separable(img, X_exact, Y_exact)
    m_sep_exact, a_sep_exact = opcount_sep(H, W, kh, kw)

    print("\n(b) Separable Convolution (exact; rank=1)")
    print("X_exact:\n", X_exact)
    print("Y_exact:\n", Y_exact)
    print(f"Operation count: mult={m_sep_exact:,}  add={a_sep_exact:,}")
else:
    print("\n(b) Separable Convolution")
    print(f"Kernel_1 is NOT exactly separable (matrix rank = {rank}). Skipping exact (b).")

#(c) SVD rank-1 approximation
U, S, Vt = np.linalg.svd(Kernel_1, full_matrices=False)
sigma0, u0, v0 = S[0], U[:, 0], Vt[0, :]

X = np.sqrt(sigma0) * u0
Y = np.sqrt(sigma0) * v0
Kernel_approx = np.outer(X, Y)

Abs_Error = np.abs(Kernel_1 - Kernel_approx)
fro_error = np.linalg.norm(Abs_Error, 'fro')
rel_error = fro_error / np.linalg.norm(Kernel_1, 'fro')
max_error = Abs_Error.max()

out_sep_svd = conv_separable(img, X, Y)
m_sep_svd, a_sep_svd = opcount_sep(H, W, kh, kw)

np.set_printoptions(precision=5, suppress=True)
print("\n(c) SVD Rank-1 Approximation (Additional Task)")
print("Singular values:", S)
print("\nX vector (len {}):\n{}".format(len(X), X))
print("\nY vector (len {}):\n{}".format(len(Y), Y))
print("\nKernel_approx = X ⊗ Y:\n", Kernel_approx)
print("\nAbsolute decomposition error = |Kernel_1 - Kernel_approx|:\n", Abs_Error)
print("\nFrobenius error:", f"{fro_error:.6f}")
print("Relative error :", f"{rel_error:.6f}")
print("Max abs error  :", f"{max_error:.6f}")
print(f"\nOperation count (2×1-D with rank-1): mult={m_sep_svd:,}  add={a_sep_svd:,}")

