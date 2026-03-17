#Nama: Fajira Zahara
#NIM: 24343033
#Class Code: 202523430039

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage.metrics import structural_similarity as ssim

# ==============================
# LOAD CITRA
# ==============================
image = cv2.imread('IMG1111.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ==============================
# TAMBAH NOISE
# ==============================
def add_gaussian_noise(img):
    mean = 0
    std = 25
    noise = np.random.normal(mean, std, img.shape)
    noisy = img + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_pepper_noise(img):
    noisy = img.copy()
    prob = 0.05
    rnd = np.random.rand(*img.shape)

    noisy[rnd < prob/2] = 0
    noisy[rnd > 1 - prob/2] = 255

    return noisy

def add_speckle_noise(img):
    noise = np.random.randn(*img.shape)
    noisy = img + img * noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

gaussian_noise = add_gaussian_noise(image)
sp_noise = add_salt_pepper_noise(image)
speckle_noise = add_speckle_noise(image)

# ==============================
# FILTERING
# ==============================

# Mean Filter
def mean_filter(img, k):
    return cv2.blur(img, (k, k))

# Gaussian Filter
def gaussian_filter(img, sigma):
    return cv2.GaussianBlur(img, (0, 0), sigma)

# Median Filter
def median_filter(img, k):
    return cv2.medianBlur(img, k)

# Min Filter (erosi)
def min_filter(img, k):
    kernel = np.ones((k, k), np.uint8)
    return cv2.erode(img, kernel)

# ==============================
# METRIK EVALUASI
# ==============================
def calculate_metrics(original, filtered):
    mse = np.mean((original - filtered) ** 2)

    if mse == 0:
        psnr = 100
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))

    ssim_value = ssim(original, filtered)

    return mse, psnr, ssim_value

# ==============================
# EVALUASI
# ==============================
def evaluate(noisy_img, label):
    results = []

    filters = [
        ("Mean 3x3", lambda img: mean_filter(img, 3)),
        ("Mean 5x5", lambda img: mean_filter(img, 5)),
        ("Gaussian sigma=1", lambda img: gaussian_filter(img, 1)),
        ("Gaussian sigma=2", lambda img: gaussian_filter(img, 2)),
        ("Median 3x3", lambda img: median_filter(img, 3)),
        ("Median 5x5", lambda img: median_filter(img, 5)),
        ("Min 3x3", lambda img: min_filter(img, 3)),
    ]

    print(f"\n=== Evaluasi untuk {label} ===")

    for name, func in filters:
        start = time.time()
        filtered = func(noisy_img)
        end = time.time()

        mse, psnr, ssim_val = calculate_metrics(image, filtered)

        results.append((name, mse, psnr, ssim_val, end - start))

        print(f"{name}:")
        print(f"  MSE  : {mse:.2f}")
        print(f"  PSNR : {psnr:.2f}")
        print(f"  SSIM : {ssim_val:.4f}")
        print(f"  Time : {end - start:.5f} s")

    return results

# ==============================
# VISUALISASI
# ==============================
def show_results(original, noisy, filtered, title):
    plt.figure(figsize=(10,4))
    plt.suptitle(title)

    plt.subplot(1,3,1)
    plt.imshow(original, cmap='gray')
    plt.title("Original")

    plt.subplot(1,3,2)
    plt.imshow(noisy, cmap='gray')
    plt.title("Noisy")

    plt.subplot(1,3,3)
    plt.imshow(filtered, cmap='gray')
    plt.title("Filtered")

    plt.show()

# ==============================
# JALANKAN EVALUASI
# ==============================
gauss_results = evaluate(gaussian_noise, "Gaussian Noise")
sp_results = evaluate(sp_noise, "Salt & Pepper Noise")
speckle_results = evaluate(speckle_noise, "Speckle Noise")

# Contoh visual (ambil filter terbaik manual)
best_gauss = gaussian_filter(gaussian_noise, 1)
best_sp = median_filter(sp_noise, 3)
best_speckle = gaussian_filter(speckle_noise, 1)

show_results(image, gaussian_noise, best_gauss, "Gaussian Noise Restoration")
show_results(image, sp_noise, best_sp, "Salt & Pepper Restoration")
show_results(image, speckle_noise, best_speckle, "Speckle Restoration")