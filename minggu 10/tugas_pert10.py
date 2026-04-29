# Nama: Fajira Zahara
# NIM: 24343033
# Class Code: 202523430039

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def praktikum_morfologi_dua_tampilan():

    print("PIPELINE MORFOLOGI (2 TAMPILAN)")
    print("="*60)

    # ===== 1. LOAD CITRA
    imgA = cv2.imread("gam1.jpeg", 0)
    imgB = cv2.imread("p3.jpg", 0)

    if imgA is None or imgB is None:
        print("ERROR: Pastikan gam1.jpg & gam2.jpg ada")
        return

    # ===== 2. KERNEL =====
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    kernel7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))

    # =================== CITRA A (OCR) =======================
    print("\nPROSES CITRA A (OCR)")
    print("-"*40)

    # Threshold
    th = cv2.adaptiveThreshold(imgA,255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,11,2)

    # Erosi & Dilasi (variasi)
    erosi1 = cv2.erode(th, kernel3, iterations=1)
    erosi3 = cv2.erode(th, kernel3, iterations=3)

    dilasi1 = cv2.dilate(th, kernel3, iterations=1)
    dilasi3 = cv2.dilate(th, kernel3, iterations=3)

    # Operasi majemuk
    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel3)
    closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel5)
    gradient = cv2.morphologyEx(th, cv2.MORPH_GRADIENT, kernel3)
    tophat = cv2.morphologyEx(imgA, cv2.MORPH_TOPHAT, kernel5)
    blackhat = cv2.morphologyEx(imgA, cv2.MORPH_BLACKHAT, kernel5)

    # OCR pipeline
    start_ocr = time.time()
    clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel3)
    connect = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel5)
    final_ocr = cv2.dilate(connect, kernel3)
    time_ocr = time.time() - start_ocr

    # ===== VISUALISASI CITRA A =====
    fig1, ax1 = plt.subplots(3,4, figsize=(15,10))

    ax1[0,0].imshow(imgA, cmap='gray'); ax1[0,0].set_title("Original")
    ax1[0,1].imshow(th, cmap='gray'); ax1[0,1].set_title("Threshold")
    ax1[0,2].imshow(opening, cmap='gray'); ax1[0,2].set_title("Opening")
    ax1[0,3].imshow(closing, cmap='gray'); ax1[0,3].set_title("Closing")

    ax1[1,0].imshow(erosi1, cmap='gray'); ax1[1,0].set_title("Erosi 1x")
    ax1[1,1].imshow(erosi3, cmap='gray'); ax1[1,1].set_title("Erosi 3x")
    ax1[1,2].imshow(dilasi1, cmap='gray'); ax1[1,2].set_title("Dilasi 1x")
    ax1[1,3].imshow(dilasi3, cmap='gray'); ax1[1,3].set_title("Dilasi 3x")

    ax1[2,0].imshow(gradient, cmap='gray'); ax1[2,0].set_title("Gradient")
    ax1[2,1].imshow(tophat, cmap='gray'); ax1[2,1].set_title("TopHat")
    ax1[2,2].imshow(blackhat, cmap='gray'); ax1[2,2].set_title("BlackHat")
    ax1[2,3].imshow(final_ocr, cmap='gray'); ax1[2,3].set_title("Final OCR")

    for ax in ax1.ravel():
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    # =================== CITRA B (COUNTING) ==================

    print("\nPROSES CITRA B (COUNTING)")
    print("-"*40)

    start_count = time.time()

    _, threshB = cv2.threshold(imgB,0,255,cv2.THRESH_OTSU)
    openingB = cv2.morphologyEx(threshB, cv2.MORPH_OPEN, kernel3)

    dist = cv2.distanceTransform(openingB, cv2.DIST_L2,5)
    _, fg = cv2.threshold(dist,0.5*dist.max(),255,0)
    fg = np.uint8(fg)

    _, markers = cv2.connectedComponents(fg)
    markers += 1

    img_color = cv2.cvtColor(imgB, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color, markers)

    count_auto = len(np.unique(markers)) - 2
    time_count = time.time() - start_count

    # ===== VISUALISASI CITRA B =====
    fig2, ax2 = plt.subplots(2,3, figsize=(12,6))

    ax2[0,0].imshow(imgB, cmap='gray'); ax2[0,0].set_title("Original")
    ax2[0,1].imshow(threshB, cmap='gray'); ax2[0,1].set_title("Threshold")
    ax2[0,2].imshow(openingB, cmap='gray'); ax2[0,2].set_title("Opening")

    ax2[1,0].imshow(dist, cmap='gray'); ax2[1,0].set_title("Distance Transform")
    ax2[1,1].imshow(fg, cmap='gray'); ax2[1,1].set_title("Foreground")
    ax2[1,2].imshow(markers, cmap='jet'); ax2[1,2].set_title(f"Watershed ({count_auto})")

    for ax in ax2.ravel():
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    # ===== EVALUASI =====
    def count_components(img):
        num,_ = cv2.connectedComponents(img)
        return max(num-1,1)

    comp_before = count_components(th)
    comp_after = count_components(final_ocr)

    manual = int(input("\nMasukkan jumlah objek manual (Citra B): "))
    acc = (count_auto/manual)*100

    print("\nEVALUASI OCR")
    print("="*40)
    print(f"Sebelum : {comp_before}")
    print(f"Sesudah : {comp_after}")

    print("\nEVALUASI COUNTING")
    print("="*40)
    print(f"Manual  : {manual}")
    print(f"Otomatis: {count_auto}")
    print(f"Akurasi : {acc:.1f}%")

    print("\nWAKTU")
    print("="*40)
    print(f"OCR     : {time_ocr:.4f} detik")
    print(f"Counting: {time_count:.4f} detik")

    print("\nKESIMPULAN:")
    print("✔ Citra A fokus pembersihan teks (OCR)")
    print("✔ Citra B fokus pemisahan objek (Watershed)")
    print("✔ Pemisahan tampilan mempermudah analisis")

# RUN
praktikum_morfologi_dua_tampilan()