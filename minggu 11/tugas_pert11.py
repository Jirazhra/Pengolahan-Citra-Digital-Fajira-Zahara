# Nama: Fajira Zahara
# NIM: 24343033
# Class Code: 202523430039

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def shape_analysis_pipeline():

    print("SHAPE ANALYSIS PIPELINE (FINAL COMPLETE VERSION)")
    print("="*70)

    # ===== 1. PATH =====
    base_path = "MY_DATA/train"

    # ===== 2. AUTO DETECT CLASS =====
    classes = [d for d in os.listdir(base_path)
               if os.path.isdir(os.path.join(base_path, d))]

    print("\nClass terdeteksi:", classes)

    # ===== 3. LOAD DATA =====
    def load_all_data():

        images = []
        labels = []

        print("\nLoading dataset...")

        for label in classes:

            folder = os.path.join(base_path, label)

            files = os.listdir(folder)

            count = 0

            for file in files:

                if not file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    continue

                path = os.path.join(folder, file)

                img = cv2.imread(path, 0)

                if img is None:
                    continue

                img = cv2.resize(img, (256,256))

                images.append(img)
                labels.append(label)

                count += 1

                # max 15 gambar per kelas
                if count >= 15:
                    break

            print(f"{label} : {count} gambar")

        return images, labels

    images, labels = load_all_data()

    if len(images) == 0:
        print("❌ Dataset kosong!")
        return

    # ===== 4. SPLIT DATA =====
    train_images, test_images, train_labels, test_labels = train_test_split(
        images,
        labels,
        test_size=0.3,
        random_state=42,
        stratify=labels
    )

    print("\n=== DISTRIBUSI DATA ===")

    for c in classes:
        print(f"{c} -> Train: {train_labels.count(c)} | Test: {test_labels.count(c)}")

    # ===== 5. PREPROCESS =====
    def preprocess(img):

        blur = cv2.GaussianBlur(img, (5,5), 0)

        _, th = cv2.threshold(
            blur,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        return th

    # ===== 6. CONTOUR =====
    def get_contour(img):

        cnts, _ = cv2.findContours(
            img,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        if len(cnts) > 0:
            return max(cnts, key=cv2.contourArea)

        return None

    # ===== 7. CHAIN CODE =====
    def get_chain_code(cnt):

        directions = [
            (1,0), (1,-1), (0,-1), (-1,-1),
            (-1,0), (-1,1), (0,1), (1,1)
        ]

        code = []

        pts = cnt.squeeze()

        for i in range(len(pts)-1):

            dx = pts[i+1][0] - pts[i][0]
            dy = pts[i+1][1] - pts[i][1]

            step = (
                np.sign(dx).astype(int),
                np.sign(dy).astype(int)
            )

            if step in directions:
                code.append(directions.index(step))

        return code[:40]

    # ===== 8. FOURIER RECONSTRUCTION =====
    def fourier_reconstruction(cnt, n_descriptors=20):

        pts = cnt.squeeze()

        complex_pts = pts[:,0] + 1j*pts[:,1]

        fft = np.fft.fft(complex_pts)

        fft_copy = np.zeros_like(fft)

        fft_copy[:n_descriptors] = fft[:n_descriptors]
        fft_copy[-n_descriptors:] = fft[-n_descriptors:]

        recon = np.fft.ifft(fft_copy)

        return recon.real, recon.imag

    # ===== 9. FEATURE EXTRACTION =====
    def extract_features(img):

        th = preprocess(img)

        cnt = get_contour(th)

        if cnt is None:
            return np.zeros(15)

        # basic shape
        area = cv2.contourArea(cnt)

        perimeter = cv2.arcLength(cnt, True)

        x,y,w,h = cv2.boundingRect(cnt)

        aspect_ratio = w/h if h != 0 else 0

        extent = area/(w*h) if w*h != 0 else 0

        hull = cv2.convexHull(cnt)

        hull_area = cv2.contourArea(hull)

        solidity = area/hull_area if hull_area != 0 else 0

        # moments
        M = cv2.moments(cnt)

        hu = cv2.HuMoments(M).flatten()

        hu = -np.sign(hu) * np.log10(np.abs(hu)+1e-10)

        # fourier descriptor
        pts = cnt.squeeze()

        complex_pts = pts[:,0] + 1j*pts[:,1]

        fft = np.fft.fft(complex_pts)

        fft_desc = np.abs(fft[1:6])

        return np.hstack([
            area,
            perimeter,
            aspect_ratio,
            extent,
            solidity,
            hu[:5],
            fft_desc
        ])

    # ===== 10. EKSTRAK FITUR =====
    print("\nEkstraksi fitur...")

    X_train = np.array([
        extract_features(img)
        for img in train_images
    ])

    X_test = np.array([
        extract_features(img)
        for img in test_images
    ])

    # ===== 11. NORMALISASI =====
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    # ===== 12. TRAIN MODEL =====
    print("\nTraining k-NN...")

    model = KNeighborsClassifier(n_neighbors=3)

    model.fit(X_train, train_labels)

    # ===== 13. TEST =====
    y_pred = model.predict(X_test)

    acc = accuracy_score(test_labels, y_pred)

    print(f"\nAKURASI MODEL: {acc*100:.2f}%")

    # =========================================================
    # TABEL REGION PROPERTIES & MOMENTS
    # =========================================================
    print("\n")
    print("="*70)
    print("TABEL PROPERTI REGION DAN MOMENTS")
    print("="*70)

    region_data = []

    for i in range(min(5, len(test_images))):

        img = test_images[i]

        th = preprocess(img)

        cnt = get_contour(th)

        if cnt is None:
            continue

        area = cv2.contourArea(cnt)

        perimeter = cv2.arcLength(cnt, True)

        x,y,w,h = cv2.boundingRect(cnt)

        aspect_ratio = w/h

        M = cv2.moments(cnt)

        hu = cv2.HuMoments(M).flatten()

        region_data.append([
            test_labels[i],
            round(area,2),
            round(perimeter,2),
            round(aspect_ratio,2),
            round(hu[0],5),
            round(hu[1],5),
            round(hu[2],5)
        ])

    df = pd.DataFrame(
        region_data,
        columns=[
            "Class",
            "Area",
            "Perimeter",
            "AspectRatio",
            "Hu1",
            "Hu2",
            "Hu3"
        ]
    )

    print(df)

    # =========================================================
    # CHAIN CODE DAN FOURIER RECONSTRUCTION
    # =========================================================
    print("\n")
    print("="*70)
    print("CHAIN CODE DAN FOURIER RECONSTRUCTION")
    print("="*70)

    sample_img = test_images[0]

    th = preprocess(sample_img)

    cnt = get_contour(th)

    chain_code = get_chain_code(cnt)

    print("\nChain Code (40 pertama):")
    print(chain_code)

    # Fourier reconstruction
    rx, ry = fourier_reconstruction(cnt)

    fig, ax = plt.subplots(1,2, figsize=(10,5))

    # contour asli
    pts = cnt.squeeze()

    ax[0].plot(pts[:,0], pts[:,1])
    ax[0].invert_yaxis()
    ax[0].set_title("Contour Asli")

    # rekonstruksi
    ax[1].plot(rx, ry)
    ax[1].invert_yaxis()
    ax[1].set_title("Fourier Reconstruction")

    plt.tight_layout()
    plt.show()

    # =========================================================
    # CONFUSION MATRIX
    # =========================================================
    print("\n")
    print("="*70)
    print("MATRIKS AKURASI KLASIFIKASI")
    print("="*70)

    cm = confusion_matrix(test_labels, y_pred, labels=classes)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(test_labels, y_pred))

    # visualisasi confusion matrix
    fig, ax = plt.subplots(figsize=(6,5))

    im = ax.imshow(cm)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))

    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j],
                    ha="center",
                    va="center")

    plt.show()

    # =========================================================
    # VISUALISASI HASIL KLASIFIKASI
    # =========================================================
    print("\nMenampilkan hasil klasifikasi...")

    selected_classes = classes[:3]

    fig, axes = plt.subplots(
        len(selected_classes),
        3,
        figsize=(10,8)
    )

    for row, cls in enumerate(selected_classes):

        idxs = [
            i for i, l in enumerate(test_labels)
            if l == cls
        ]

        if len(idxs) == 0:
            continue

        for col in range(3):

            if col >= len(idxs):
                axes[row, col].axis('off')
                continue

            i = idxs[col]

            img = test_images[i]

            th = preprocess(img)

            cnt = get_contour(th)

            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            if cnt is not None:
                cv2.drawContours(
                    img_color,
                    [cnt],
                    -1,
                    (0,255,0),
                    2
                )

            feat = scaler.transform([
                extract_features(img)
            ])

            pred = model.predict(feat)[0]

            axes[row, col].imshow(img_color)

            axes[row, col].set_title(
                f"True: {cls}\nPred: {pred}"
            )

            axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()

    # =========================================================
    # REKOMENDASI DESKRIPTOR
    # =========================================================
    print("\n")
    print("="*70)
    print("REKOMENDASI DESKRIPTOR UNTUK APLIKASI SPESIFIK")
    print("="*70)

    rekomendasi = {
        "Deteksi Bentuk Buah":
            "Area, Perimeter, Aspect Ratio, Hu Moments",

        "OCR / Tulisan Tangan":
            "Chain Code dan Fourier Descriptor",

        "Pengenalan Logo":
            "Hu Moments + Fourier Descriptor",

        "Medical Imaging":
            "Solidity, Extent, Shape Descriptor",

        "Object Tracking":
            "Centroid dan Region Properties"
    }

    for aplikasi, descriptor in rekomendasi.items():

        print(f"\n{aplikasi}")
        print(f"-> Descriptor yang cocok: {descriptor}")

# RUN
shape_analysis_pipeline()
