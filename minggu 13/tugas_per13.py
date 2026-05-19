# ==============================================================
# PRAKTIKUM PERTEMUAN 13
# KNN vs SVM - Fashion MNIST (IDX Dataset Version)
# ==============================================================

import numpy as np
import matplotlib.pyplot as plt
import idx2numpy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

from sklearn.decomposition import PCA

from skimage.feature import hog
from skimage.measure import moments_hu

import warnings
warnings.filterwarnings('ignore')


# ==============================================================
# LOAD DATASET IDX
# ==============================================================

def load_dataset():

    print("=" * 65)
    print("  LANGKAH 1: MEMUAT DATASET FASHION-MNIST")
    print("=" * 65)

    # ==========================================================
    # BACA FILE IDX
    # ==========================================================

    X_train = idx2numpy.convert_from_file(
        'train-images-idx3-ubyte'
    )

    y_train = idx2numpy.convert_from_file(
        'train-labels-idx1-ubyte'
    )

    X_test = idx2numpy.convert_from_file(
        't10k-images-idx3-ubyte'
    )

    y_test = idx2numpy.convert_from_file(
        't10k-labels-idx1-ubyte'
    )

    # ==========================================================
    # GABUNGKAN TRAIN + TEST
    # ==========================================================

    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    print(f"Jumlah total data : {len(X)}")
    print(f"Ukuran gambar     : {X.shape[1]} x {X.shape[2]}")

    # ==========================================================
    # AMBIL 1000 SAMPEL AGAR CEPAT
    # ==============================================================

    np.random.seed(42)

    idx = np.random.choice(len(X), 1000, replace=False)

    X = X[idx]
    y = y[idx]

    print(f"Jumlah sampel dipakai : {len(X)}")

    return X, y


# ==============================================================
# VISUALISASI DATASET
# ==============================================================

def tampilkan_dataset(X, y):

    print("\nMenampilkan contoh dataset...")

    nama_kelas = [
        'T-shirt',
        'Trouser',
        'Pullover',
        'Dress',
        'Coat',
        'Sandal',
        'Shirt',
        'Sneaker',
        'Bag',
        'Ankle Boot'
    ]

    fig, axes = plt.subplots(2, 5, figsize=(14, 7))

    for i, ax in enumerate(axes.flat):

        # ======================================================
        # TAMBAHAN AGAR GAMBAR LEBIH JELAS
        # ======================================================

        ax.imshow(
            X[i],
            cmap='gray',
            interpolation='lanczos'
        )

        ax.set_title(
            nama_kelas[y[i]],
            fontsize=10
        )

        ax.axis('off')

    plt.tight_layout()
    plt.show()


# ==============================================================
# EKSTRAKSI FITUR
# ==============================================================

def ekstraksi_fitur(X):

    print("\n" + "=" * 65)
    print("  LANGKAH 2: EKSTRAKSI FITUR")
    print("=" * 65)

    fitur = []

    for i, img in enumerate(X):

        # ======================================================
        # HOG FEATURE
        # ======================================================

        hog_feat = hog(
            img,
            orientations=8,
            pixels_per_cell=(7, 7),
            cells_per_block=(2, 2),
            visualize=False,
            block_norm='L2-Hys'
        )

        # ======================================================
        # HU MOMENTS
        # ======================================================

        hu = moments_hu(img)

        hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

        # ======================================================
        # GABUNGKAN FITUR
        # ======================================================

        fitur_gabung = np.concatenate([hog_feat, hu_log])

        fitur.append(fitur_gabung)

        if i % 100 == 0:
            print(f"Proses fitur : {i}/{len(X)}")

    fitur = np.array(fitur)

    print(f"\nShape fitur : {fitur.shape}")

    return fitur


# ==============================================================
# VISUALISASI HOG
# ==============================================================

def visualisasi_hog(X):

    print("\nMenampilkan visualisasi fitur HOG...")

    fig, axes = plt.subplots(2, 5, figsize=(15, 7))

    for i in range(5):

        img = X[i]

        _, hog_image = hog(
            img,
            orientations=8,
            pixels_per_cell=(7, 7),
            cells_per_block=(2, 2),
            visualize=True,
            block_norm='L2-Hys'
        )

        # ======================================================
        # ORIGINAL IMAGE
        # ======================================================

        axes[0, i].imshow(
            img,
            cmap='gray',
            interpolation='lanczos'
        )

        axes[0, i].set_title("Original")
        axes[0, i].axis('off')

        # ======================================================
        # HOG IMAGE
        # ======================================================

        axes[1, i].imshow(
            hog_image,
            cmap='gray',
            interpolation='lanczos'
        )

        axes[1, i].set_title("HOG")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()


# ==============================================================
# TRAINING MODEL
# ==============================================================

def training_model(X_train, X_test, y_train, y_test):

    print("\n" + "=" * 65)
    print("  LANGKAH 3: TRAINING MODEL")
    print("=" * 65)

    hasil = {}

    # ==========================================================
    # KNN
    # ==========================================================

    print("\nTraining KNN...")

    knn = KNeighborsClassifier(
        n_neighbors=5
    )

    knn.fit(X_train, y_train)

    pred_knn = knn.predict(X_test)

    acc_knn = accuracy_score(y_test, pred_knn)

    hasil['KNN'] = {
        'model': knn,
        'prediksi': pred_knn,
        'akurasi': acc_knn
    }

    print(f"Akurasi KNN : {acc_knn:.4f}")

    # ==========================================================
    # SVM
    # ==========================================================

    print("\nTraining SVM...")

    svm = SVC(
        kernel='rbf',
        probability=True,
        max_iter=5000
    )

    svm.fit(X_train, y_train)

    pred_svm = svm.predict(X_test)

    acc_svm = accuracy_score(y_test, pred_svm)

    hasil['SVM'] = {
        'model': svm,
        'prediksi': pred_svm,
        'akurasi': acc_svm
    }

    print(f"Akurasi SVM : {acc_svm:.4f}")

    return hasil


# ==============================================================
# EVALUASI MODEL
# ==============================================================

def evaluasi_model(hasil, y_test):

    print("\n" + "=" * 65)
    print("  LANGKAH 4: EVALUASI MODEL")
    print("=" * 65)

    for nama, data in hasil.items():

        print(f"\n{'=' * 50}")
        print(f"HASIL MODEL : {nama}")
        print(f"{'=' * 50}")

        print(f"\nAkurasi : {data['akurasi']:.4f}")

        print("\nClassification Report")

        print(
            classification_report(
                y_test,
                data['prediksi']
            )
        )

        # ======================================================
        # CONFUSION MATRIX
        # ======================================================

        cm = confusion_matrix(
            y_test,
            data['prediksi']
        )

        plt.figure(figsize=(8, 6))

        plt.imshow(cm, cmap='Blues')

        plt.title(f'Confusion Matrix - {nama}')

        plt.colorbar()

        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        plt.show()


# ==============================================================
# VISUALISASI PCA
# ==============================================================

def visualisasi_pca(X, y):

    print("\nMenampilkan PCA 2D...")

    pca = PCA(n_components=2)

    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(10, 7))

    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=y,
        cmap='tab10'
    )

    plt.title("Visualisasi PCA")

    plt.colorbar(scatter)

    plt.show()


# ==============================================================
# ROC CURVE
# ==============================================================

def tampilkan_roc(hasil, X_test, y_test):

    print("\nMenampilkan ROC Curve...")

    y_bin = label_binarize(y_test, classes=np.arange(10))

    for nama, data in hasil.items():

        model = data['model']

        y_score = model.predict_proba(X_test)

        plt.figure(figsize=(8, 6))

        for i in range(10):

            fpr, tpr, _ = roc_curve(
                y_bin[:, i],
                y_score[:, i]
            )

            roc_auc = auc(fpr, tpr)

            plt.plot(
                fpr,
                tpr,
                label=f'Class {i} (AUC={roc_auc:.2f})'
            )

        plt.plot([0, 1], [0, 1], 'k--')

        plt.title(f'ROC Curve - {nama}')

        plt.xlabel('False Positive Rate')

        plt.ylabel('True Positive Rate')

        plt.legend()

        plt.show()


# ==============================================================
# MAIN PROGRAM
# ==============================================================

def main():

    print("\n")
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  PRAKTIKUM PERTEMUAN 13 – KNN vs SVM (Fashion-MNIST)         ║")
    print("║  Komparasi Klasifikasi untuk Pengenalan Objek Citra          ║")
    print("╚═══════════════════════════════════════════════════════════════╝")

    # ==========================================================
    # LOAD DATASET
    # ==========================================================

    X, y = load_dataset()

    # ==========================================================
    # VISUALISASI DATA
    # ==========================================================

    tampilkan_dataset(X, y)

    # ==========================================================
    # VISUALISASI HOG
    # ==========================================================

    visualisasi_hog(X)

    # ==========================================================
    # EKSTRAKSI FITUR
    # ==========================================================

    fitur = ekstraksi_fitur(X)

    # ==========================================================
    # NORMALISASI
    # ==========================================================

    scaler = StandardScaler()

    fitur_norm = scaler.fit_transform(fitur)

    # ==========================================================
    # SPLIT DATA
    # ==========================================================

    X_train, X_test, y_train, y_test = train_test_split(
        fitur_norm,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nData Training :", X_train.shape)
    print("Data Testing  :", X_test.shape)

    # ==========================================================
    # PCA
    # ==========================================================

    visualisasi_pca(fitur_norm, y)

    # ==========================================================
    # TRAINING
    # ==========================================================

    hasil = training_model(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # ==========================================================
    # EVALUASI
    # ==========================================================

    evaluasi_model(
        hasil,
        y_test
    )

    # ==========================================================
    # ROC CURVE
    # ==========================================================

    tampilkan_roc(
        hasil,
        X_test,
        y_test
    )

    print("\n" + "=" * 65)
    print("PROGRAM SELESAI")
    print("=" * 65)


# ==============================================================
# RUN PROGRAM
# ==============================================================

if __name__ == "__main__":
    main()