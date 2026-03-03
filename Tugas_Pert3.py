#Nama: Fajira Zahara
#NIM: 24343033
#Class Code: 202523430039

# ============================================
# PRAKTIKUM: TRANSFORMASI GEOMETRIK & INTERPOLASI
# ============================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

print("=== PRAKTIKUM 3 ===")
print("Kasus: Perbandingan Dokumen Lurus vs Dokumen Miring\n")

# ================= FUNGSI METRIK =================
def mse_psnr(img1, img2):
    mse = np.mean((img1.astype(float) - img2.astype(float))**2)
    psnr = 10*np.log10(255**2/mse) if mse > 0 else float('inf')
    return mse, psnr

# ================= 1. LOAD DUA GAMBAR =================
print("1. Menampilkan Dua Citra")

ref_img = cv2.imread("foto_lurus.jpg", 0)
moving_img = cv2.imread("foto_miring.jpg", 0)

h, w = ref_img.shape

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(ref_img, cmap='gray')
plt.title("Gambar Lurus")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(moving_img, cmap='gray')
plt.title("Gambar Miring")
plt.axis("off")
plt.show()


# ================= 2. TRANSFORMASI HOMOGEN =================
print("2. Translasi, Rotasi, Scaling (Homogen)")

# Matriks
translation = np.array([[1,0,40],[0,1,30],[0,0,1]])
theta = np.pi/10
rotation = np.array([[np.cos(theta),-np.sin(theta),0],
                     [np.sin(theta),np.cos(theta),0],
                     [0,0,1]])
scaling = np.array([[1.1,0,0],[0,1.1,0],[0,0,1]])

composite = translation @ rotation @ scaling

homo_ref = cv2.warpPerspective(ref_img, composite, (w,h))
homo_mov = cv2.warpPerspective(moving_img, composite, (w,h))

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(homo_ref, cmap='gray')
plt.title("Homogen - Lurus")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(homo_mov, cmap='gray')
plt.title("Homogen - Miring")
plt.axis("off")
plt.show()


# ================= 3. AFFINE =================
print("3. Transformasi Affine (3 titik)")

pts_ref = np.float32([[100,100],[w-100,100],[100,h-100]])
pts_mov = np.float32([[80,120],[w-60,80],[120,h-60]])

M_aff = cv2.getAffineTransform(pts_mov, pts_ref)

aff_result = cv2.warpAffine(moving_img, M_aff, (w,h))

plt.figure(figsize=(8,5))
plt.imshow(aff_result, cmap='gray')
plt.title("Hasil Affine - Miring ke Lurus")
plt.axis("off")
plt.show()


# ================= 4. PERSPEKTIF =================
print("4. Transformasi Perspektif (4 titik)")

pts_ref4 = np.float32([[100,100],[w-100,100],[w-100,h-100],[100,h-100]])
pts_mov4 = np.float32([[80,120],[w-60,80],[w-40,h-60],[120,h-40]])

M_persp = cv2.getPerspectiveTransform(pts_mov4, pts_ref4)

persp_result = cv2.warpPerspective(moving_img, M_persp, (w,h))

plt.figure(figsize=(8,5))
plt.imshow(persp_result, cmap='gray')
plt.title("Hasil Perspective Registration")
plt.axis("off")
plt.show()


# ================= 5. INTERPOLASI =================
print("5. Perbandingan Interpolasi")

methods = [
    ("Nearest", cv2.INTER_NEAREST),
    ("Bilinear", cv2.INTER_LINEAR),
    ("Bicubic", cv2.INTER_CUBIC)
]

plt.figure(figsize=(15,5))

for i,(name,flag) in enumerate(methods):
    start = time.time()
    result = cv2.warpPerspective(moving_img, M_persp, (w,h), flags=flag)
    comp_time = time.time() - start
    
    mse, psnr = mse_psnr(ref_img, result)
    
    print(f"{name} -> MSE:{mse:.2f}, PSNR:{psnr:.2f}, Time:{comp_time:.5f}s")
    
    plt.subplot(1,3,i+1)
    plt.imshow(result, cmap='gray')
    plt.title(f"{name}\nPSNR:{psnr:.1f}")
    plt.axis("off")

plt.show()


# ================= 6. ANALISIS & KESIMPULAN =================
print("\n6. ANALISIS")

print("""
• Perspective lebih akurat untuk dokumen difoto miring.
• Affine cukup jika hanya rotasi/translasi.
• Bicubic biasanya memberi PSNR tertinggi.
• Nearest paling cepat tapi kualitas kasar.
""")

print("\nRekomendasi Pipeline:")
print("""
1. Deteksi 4 sudut dokumen
2. Hitung matriks perspective
3. Gunakan warpPerspective + Bicubic
4. Evaluasi MSE & PSNR
5. Gunakan hasil untuk OCR
""")