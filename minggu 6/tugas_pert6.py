#Nama: Fajira Zahara
#NIM: 24343033
#Class Code: 202523430039

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from scipy.signal import convolve2d
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import richardson_lucy

# LOAD IMAGE (AUTO)
img = cv2.imread('img1414.jpg', 0)

if img is None:
    raise Exception("Gambar tidak ditemukan!")

img = img.astype(np.float32)

# PSF MOTION BLUR
def motion_psf(length=15, angle=30):
    psf = np.zeros((length, length))
    center = length // 2

    for i in range(length):
        x = int(center + (i - center) * np.cos(np.deg2rad(angle)))
        y = int(center + (i - center) * np.sin(np.deg2rad(angle)))
        if 0 <= x < length and 0 <= y < length:
            psf[y, x] = 1

    psf /= psf.sum()
    return psf

psf = motion_psf(15, 30)

# FILTERING METHODS
def inverse_filter(img, psf):
    img_fft = np.fft.fft2(img)
    psf_fft = np.fft.fft2(psf, s=img.shape)
    psf_fft[np.abs(psf_fft) < 1e-3] = 1e-3
    result = np.abs(np.fft.ifft2(img_fft / psf_fft))
    return result

def wiener_filter(img, psf, K=0.01):
    img_fft = np.fft.fft2(img)
    psf_fft = np.fft.fft2(psf, s=img.shape)
    psf_conj = np.conj(psf_fft)
    result = np.abs(np.fft.ifft2((psf_conj / (np.abs(psf_fft)**2 + K)) * img_fft))
    return result

def lucy_filter(img, psf):
    # NORMALISASI KE 0–1
    img_norm = img / 255.0

    result = richardson_lucy(img_norm, psf, num_iter=30)

    # KEMBALIKAN KE 0–255
    result = result * 255
    return result   

# DEGRADASI
def degrade(img, psf):
    blur = convolve2d(img, psf, 'same')

    gaussian = blur + np.random.normal(0, 20, img.shape)

    sp = img.copy()
    prob = 0.05
    rand = np.random.rand(*img.shape)
    sp[rand < prob/2] = 0
    sp[rand > 1 - prob/2] = 255
    sp_blur = convolve2d(sp, psf, 'same')

    return {
        "Motion Blur": blur,
        "Gaussian + Blur": gaussian,
        "SaltPepper + Blur": sp_blur
    }

# EVALUASI
def evaluate(original, restored):
    restored = np.clip(restored, 0, 255)
    return mse(original, restored), psnr(original, restored, data_range=255), ssim(original, restored, data_range=255)

# PROSES
data = []
datasets = degrade(img, psf)

for name, degraded in datasets.items():
    methods = {
        "Inverse": inverse_filter,
        "Wiener": wiener_filter,
        "Lucy": lucy_filter
    }

    for m_name, func in methods.items():
        start = time.time()
        result = func(degraded, psf)
        t = time.time() - start

        m, p, s = evaluate(img, result)

        data.append([name, m_name, m, p, s, t])

# TABEL HASIL
df = pd.DataFrame(data, columns=["Degradasi", "Metode", "MSE", "PSNR", "SSIM", "Waktu"])

print("\n=== HASIL EVALUASI ===")
print(df)

df.to_csv("hasil_restorasi.csv", index=False)
print("\nDisimpan ke hasil_restorasi.csv")

# GRAFIK
for metric in ["PSNR", "SSIM"]:
    plt.figure()
    for degradasi in df["Degradasi"].unique():
        subset = df[df["Degradasi"] == degradasi]
        plt.plot(subset["Metode"], subset[metric])

    plt.title(f"Perbandingan {metric}")
    plt.xlabel("Metode")
    plt.ylabel(metric)
    plt.show()

# VISUALISASI GAMBAR
plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(2,2,2)
plt.imshow(datasets["Motion Blur"], cmap='gray')
plt.title("Motion Blur")

plt.subplot(2,2,3)
plt.imshow(datasets["Gaussian + Blur"], cmap='gray')
plt.title("Gaussian + Blur")

plt.subplot(2,2,4)
plt.imshow(datasets["SaltPepper + Blur"], cmap='gray')
plt.title("Salt & Pepper + Blur")

plt.show()

# GRAFIK
for metric in ["PSNR", "SSIM"]:
    plt.figure()
    for degradasi in df["Degradasi"].unique():
        subset = df[df["Degradasi"] == degradasi]
        plt.plot(subset["Metode"], subset[metric])

    plt.title(f"Perbandingan {metric}")
    plt.xlabel("Metode")
    plt.ylabel(metric)
    plt.show()

# VISUALISASI RESTORASI
for name, degraded in datasets.items():
    plt.figure(figsize=(10,6))

    plt.subplot(2,2,1)
    plt.imshow(degraded, cmap='gray')
    plt.title(f"{name}")

    inv = inverse_filter(degraded, psf)
    inv = np.clip(inv, 0, 255).astype(np.uint8)
    plt.subplot(2,2,2)
    plt.imshow(inv, cmap='gray', vmin=0, vmax=255)
    plt.title("Inverse")

    wie = wiener_filter(degraded, psf)
    wie = np.clip(wie, 0, 255).astype(np.uint8)
    plt.subplot(2,2,3)
    plt.imshow(wie, cmap='gray', vmin=0, vmax=255)
    plt.title("Wiener")

    lucy = lucy_filter(degraded, psf)
    lucy = np.clip(lucy, 0, 255).astype(np.uint8)
    plt.subplot(2,2,4)
    plt.imshow(lucy, cmap='gray', vmin=0, vmax=255)
    plt.title("Lucy")

    plt.suptitle(f"Restorasi: {name}")
    plt.show()