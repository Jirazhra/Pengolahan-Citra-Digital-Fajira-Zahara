#Nama: Fajira Zahara
#NIM: 24343033
#Class Code: 202523430039

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# LOAD 2 CITRA
img_nat = cv2.imread('IMG1213.jpg', 0)
img_pat = cv2.imread('IMG12131.jpg', 0)

if img_nat is None or img_pat is None:
    raise Exception("pastikan gambar tersedia!")

img_nat = cv2.resize(img_nat, (512, 512))
img_pat = cv2.resize(img_pat, (512, 512))

img_nat = img_nat.astype(np.float32)
img_pat = img_pat.astype(np.float32)

# NORMALISASI
def normalize(img):
    img = img - np.min(img)
    if np.max(img) != 0:
        img = img / np.max(img)
    return img

# FFT + VISUALISASI
def fft_analysis(img, title):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    magnitude = np.log(np.abs(fshift) + 1)
    phase = np.angle(fshift)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"{title}")

    plt.subplot(1, 3, 2)
    plt.imshow(magnitude, cmap='gray')
    plt.title("Magnitude Spectrum")

    plt.subplot(1, 3, 3)
    plt.imshow(phase, cmap='gray')
    plt.title("Phase Spectrum")

    plt.tight_layout()
    plt.show()
    return fshift

# REKONSTRUKSI (FIXED)
def reconstruct(fshift):
    magnitude = np.abs(fshift)
    phase = np.angle(fshift)

    # --- Magnitude Only ---
    mag_only = magnitude * np.exp(1j * 0)
    img_mag = np.fft.ifft2(np.fft.ifftshift(mag_only))
    img_mag = np.abs(img_mag)         
    img_mag = normalize(img_mag)       

    # --- Phase Only ---
    phase_only = 1.0 * np.exp(1j * phase)
    img_phase = np.fft.ifft2(np.fft.ifftshift(phase_only))
    img_phase = np.real(img_phase)
    img_phase = normalize(img_phase)    

    print("MAG range:   ", img_mag.min(), img_mag.max())
    print("PHASE range: ", img_phase.min(), img_phase.max())

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(normalize(fshift.__abs__()), cmap='gray')  
    plt.title("Magnitude Spectrum (log)")

    plt.subplot(1, 3, 2)
    plt.imshow(img_mag, cmap='gray')
    plt.title("Magnitude Only\n(tanpa phase info)")

    plt.subplot(1, 3, 3)
    plt.imshow(img_phase, cmap='gray')
    plt.title("Phase Only\n(magnitude uniform)")

    plt.tight_layout()
    plt.show()

# FILTER FREKUENSI
def ideal_lowpass(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2

    Y, X = np.ogrid[:rows, :cols]
    dist = np.sqrt((Y - crow)**2 + (X - ccol)**2)
    mask = (dist <= cutoff).astype(np.float32)
    return mask

def gaussian_lowpass(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    d2 = (Y - crow)**2 + (X - ccol)**2
    mask = np.exp(-d2 / (2 * (cutoff**2)))
    return mask.astype(np.float32)

def apply_filter(img, mask):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    filtered = fshift * mask
    img_back = np.fft.ifft2(np.fft.ifftshift(filtered))
    return normalize(np.real(img_back)) * 255

# NOTCH FILTER
def notch_filter(shape, centers, radius=10):
    mask = np.ones(shape, dtype=np.float32)
    rows, cols = shape
    Y, X = np.ogrid[:rows, :cols]

    for (cx, cy) in centers:
        dist = np.sqrt((Y - cx)**2 + (X - cy)**2)
        mask[dist < radius] = 0
    return mask

# WAVELET
def wavelet_process(img, wavelet='haar'):
    coeffs = pywt.wavedec2(img, wavelet, level=2)
    cA, (cH, cV, cD), *_ = coeffs

    plt.figure(figsize=(8, 6))
    plt.subplot(2, 2, 1); plt.imshow(normalize(cA), cmap='gray'); plt.title("Approx (cA)")
    plt.subplot(2, 2, 2); plt.imshow(normalize(cH), cmap='gray'); plt.title("Horizontal (cH)")
    plt.subplot(2, 2, 3); plt.imshow(normalize(cV), cmap='gray'); plt.title("Vertical (cV)")
    plt.subplot(2, 2, 4); plt.imshow(normalize(cD), cmap='gray'); plt.title("Diagonal (cD)")
    plt.suptitle(f"Wavelet: {wavelet}")
    plt.tight_layout()
    plt.show()

    # Rekonstruksi dari aproksimasi saja (zero out detail coeffs)
    coeffs_mod = [cA] + [(np.zeros_like(h), np.zeros_like(v), np.zeros_like(d))
                         for (h, v, d) in coeffs[1:]]
    rec = pywt.waverec2(coeffs_mod, wavelet)
    # Crop jika ukuran berubah sedikit
    rec = rec[:img.shape[0], :img.shape[1]]
    return normalize(rec) * 255

# PROSES UTAMA

# FFT Analysis
f_nat = fft_analysis(img_nat, "Natural Image")
f_pat = fft_analysis(img_pat, "Pattern Image")

# Rekonstruksi (sekarang magnitude tidak gelap)
print("=== Rekonstruksi Natural ===")
reconstruct(f_nat)
print("=== Rekonstruksi Pattern ===")
reconstruct(f_pat)

# Filtering
cutoff = 30
mask_ideal = ideal_lowpass(img_nat.shape, cutoff)
mask_gauss = gaussian_lowpass(img_nat.shape, cutoff)

ideal_img = apply_filter(img_nat, mask_ideal)
gauss_img = apply_filter(img_nat, mask_gauss)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1); plt.imshow(img_nat, cmap='gray');   plt.title("Original")
plt.subplot(1, 3, 2); plt.imshow(ideal_img, cmap='gray'); plt.title(f"Ideal LPF (r={cutoff})")
plt.subplot(1, 3, 3); plt.imshow(gauss_img, cmap='gray'); plt.title(f"Gaussian LPF (r={cutoff})")
plt.tight_layout()
plt.show()

# Notch filter
notch = notch_filter(img_pat.shape, [(256, 200), (256, 300)], 10)
notch_img = apply_filter(img_pat, notch)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1); plt.imshow(img_pat, cmap='gray');   plt.title("Original Pattern")
plt.subplot(1, 2, 2); plt.imshow(notch_img, cmap='gray'); plt.title("Notch Filtered")
plt.tight_layout()
plt.show()

# Wavelet
rec_haar = wavelet_process(img_nat, 'haar')
rec_db4  = wavelet_process(img_nat, 'db4')

plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1); plt.imshow(img_nat,    cmap='gray'); plt.title("Original")
plt.subplot(1, 3, 2); plt.imshow(rec_haar,   cmap='gray'); plt.title("Rekonstruksi Haar")
plt.subplot(1, 3, 3); plt.imshow(rec_db4,    cmap='gray'); plt.title("Rekonstruksi db4")
plt.tight_layout()
plt.show()

# EVALUASI
print("\n=== EVALUASI KUALITAS ===")
print(f"PSNR Ideal LPF   : {psnr(img_nat, ideal_img, data_range=255):.4f} dB")
print(f"SSIM Ideal LPF   : {ssim(img_nat, ideal_img, data_range=255):.4f}")
print(f"PSNR Gaussian LPF: {psnr(img_nat, gauss_img, data_range=255):.4f} dB")
print(f"SSIM Gaussian LPF: {ssim(img_nat, gauss_img, data_range=255):.4f}")
print(f"PSNR Haar Wavelet: {psnr(img_nat, rec_haar,  data_range=255):.4f} dB")
print(f"SSIM Haar Wavelet: {ssim(img_nat, rec_haar,  data_range=255):.4f}")
print(f"PSNR db4 Wavelet : {psnr(img_nat, rec_db4,   data_range=255):.4f} dB")
print(f"SSIM db4 Wavelet : {ssim(img_nat, rec_db4,   data_range=255):.4f}")