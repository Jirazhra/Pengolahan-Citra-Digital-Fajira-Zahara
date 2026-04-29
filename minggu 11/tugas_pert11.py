# Nama: Fajira Zahara
# NIM: 24343033
# Class Code: 202523430039

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def shape_analysis_pipeline():

    print("SHAPE ANALYSIS PIPELINE (FINAL FIXED - SPLIT DATA)")
    print("="*60)

    # ===== 1. PATH =====
    base_path = "MY_DATA/train"  # fokus ke TRAIN saja

    # ===== 2. AUTO DETECT CLASS =====
    classes = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
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
                if not file.lower().endswith(('.jpg','.png','.jpeg')):
                    continue

                path = os.path.join(folder, file)
                img = cv2.imread(path, 0)

                if img is None:
                    continue

                img = cv2.resize(img, (256,256))
                images.append(img)
                labels.append(label)

                count += 1
                if count >= 15:  # ambil max 15 per kelas
                    break

            print(f"{label} : {count} gambar")

        return images, labels

    images, labels = load_all_data()

    if len(images) == 0:
        print("❌ Dataset kosong!")
        return

    # ===== 4. SPLIT DATA =====
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )

    print("\n=== DISTRIBUSI DATA ===")
    for c in classes:
        print(f"{c} -> Train: {train_labels.count(c)} | Test: {test_labels.count(c)}")

    # ===== 5. PREPROCESS =====
    def preprocess(img):
        blur = cv2.GaussianBlur(img,(5,5),0)
        _, th = cv2.threshold(blur,0,255,cv2.THRESH_OTSU)
        return th

    # ===== 6. CONTOUR =====
    def get_contour(img):
        cnts,_ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(cnts) > 0:
            return max(cnts, key=cv2.contourArea)
        return None

    # ===== 7. FEATURE EXTRACTION =====
    def extract_features(img):
        th = preprocess(img)
        cnt = get_contour(th)

        if cnt is None:
            return np.zeros(15)

        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)

        x,y,w,h = cv2.boundingRect(cnt)
        aspect = w/h if h!=0 else 0
        extent = area/(w*h) if w*h!=0 else 0

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area/hull_area if hull_area!=0 else 0

        # Hu Moments
        M = cv2.moments(cnt)
        hu = cv2.HuMoments(M).flatten()
        hu = -np.sign(hu)*np.log10(np.abs(hu)+1e-10)

        # Fourier Descriptor
        pts = cnt.squeeze()
        complex_pts = pts[:,0] + 1j*pts[:,1]
        fft = np.fft.fft(complex_pts)
        fft = np.abs(fft[1:6])

        return np.hstack([
            area, peri, aspect, extent, solidity,
            hu[:3],
            fft
        ])

    # ===== 8. EKSTRAK FITUR =====
    print("\nEkstraksi fitur...")

    X_train = np.array([extract_features(img) for img in train_images])
    X_test  = np.array([extract_features(img) for img in test_images])

    # ===== 9. NORMALISASI =====
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ===== 10. TRAIN MODEL =====
    print("\nTraining k-NN...")

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, train_labels)

    # ===== 11. TEST =====
    y_pred = model.predict(X_test)
    acc = accuracy_score(test_labels, y_pred)

    print(f"\nAKURASI: {acc*100:.2f}%")

    # ===== 12. VISUALISASI PER KELAS =====
    print("\nMenampilkan hasil per kelas...")

    selected_classes = ["Apple", "Banana", "orange"]

    fig, axes = plt.subplots(len(selected_classes), 3, figsize=(10,8))

    for row, cls in enumerate(selected_classes):

        # ambil index data test untuk kelas ini
        idxs = [i for i, l in enumerate(test_labels) if l == cls]

        if len(idxs) == 0:
            print(f"[WARNING] Tidak ada data untuk kelas {cls}")
            continue

        # ambil max 3 contoh
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
                cv2.drawContours(img_color, [cnt], -1, (0,255,0), 2)

            feat = scaler.transform([extract_features(img)])
            pred = model.predict(feat)[0]

            axes[row, col].imshow(img_color)
            axes[row, col].set_title(f"{cls}\nPred: {pred}")
            axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()

# RUN
shape_analysis_pipeline()