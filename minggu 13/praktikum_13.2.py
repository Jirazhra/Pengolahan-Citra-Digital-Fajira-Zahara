import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.multiclass import OneVsRestClassifier

warnings.filterwarnings('ignore')


# =========================================================
# LATIHAN 2 : SVM DENGAN BERBAGAI KERNEL
# =========================================================

def praktikum_svm_fruits():

    print("\nLATIHAN 2 : SVM DENGAN BERBAGAI KERNEL")
    print("=" * 55)

    # =====================================================
    # MEMBUAT DATASET BUAH SINTETIS
    # =====================================================

    def create_fruit_dataset(n_samples=100):

        np.random.seed(42)
        n_features = 20

        # Kelas 0 = Apple
        apples = np.random.randn(n_samples, n_features)
        apples[:, 0] += 2
        apples[:, 1] += 1
        apples_labels = np.zeros(n_samples)

        # Kelas 1 = Banana
        bananas = np.random.randn(n_samples, n_features)
        bananas[:, 0] += 1
        bananas[:, 1] += 3
        bananas_labels = np.ones(n_samples)

        # Kelas 2 = Orange
        oranges = np.random.randn(n_samples, n_features)
        oranges[:, 0] += 1.5
        oranges[:, 1] += 1
        oranges[:, 2] += 2
        oranges_labels = np.ones(n_samples) * 2

        X = np.vstack((apples, bananas, oranges))
        y = np.hstack((apples_labels, bananas_labels, oranges_labels))

        return X, y

    # Membuat dataset
    X, y = create_fruit_dataset()

    fruit_names = ['Apple', 'Banana', 'Orange']

    print(f"Ukuran Dataset : {X.shape}")
    print(f"Distribusi Kelas : {np.bincount(y.astype(int))}")

    # =====================================================
    # SPLIT DATA
    # =====================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    # =====================================================
    # NORMALISASI DATA
    # =====================================================

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # =====================================================
    # PCA UNTUK VISUALISASI
    # =====================================================

    pca = PCA(n_components=2)

    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # =====================================================
    # VISUALISASI DATASET
    # =====================================================

    plt.figure(figsize=(10, 6))

    colors = ['red', 'yellow', 'orange']

    for i, color in enumerate(colors):

        plt.scatter(
            X_train_pca[y_train == i, 0],
            X_train_pca[y_train == i, 1],
            c=color,
            label=fruit_names[i],
            alpha=0.7,
            edgecolors='black'
        )

    plt.title('Visualisasi Dataset Buah (PCA)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.show()

    # =====================================================
    # EKSPERIMEN BERBAGAI KERNEL
    # =====================================================

    kernels = ['linear', 'poly', 'rbf', 'sigmoid']

    results = {}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for idx, kernel in enumerate(kernels):

        print(f"\nTraining SVM Kernel : {kernel}")

        # Membuat model
        if kernel == 'poly':

            svm_model = svm.SVC(
                kernel=kernel,
                degree=3,
                C=1.0,
                probability=True,
                random_state=42
            )

        else:

            svm_model = svm.SVC(
                kernel=kernel,
                C=1.0,
                probability=True,
                random_state=42
            )

        # Cross Validation
        cv_scores = cross_val_score(
            svm_model,
            X_train_scaled,
            y_train,
            cv=5
        )

        # Training
        svm_model.fit(X_train_scaled, y_train)

        # Prediksi
        y_pred = svm_model.predict(X_test_scaled)

        # Accuracy
        accuracy = svm_model.score(X_test_scaled, y_test)

        # Simpan hasil
        results[kernel] = {
            'cv_accuracy': cv_scores.mean(),
            'test_accuracy': accuracy,
            'model': svm_model
        }

        # =================================================
        # VISUALISASI DECISION BOUNDARY
        # =================================================

        x_min, x_max = (
            X_train_pca[:, 0].min() - 1,
            X_train_pca[:, 0].max() + 1
        )

        y_min, y_max = (
            X_train_pca[:, 1].min() - 1,
            X_train_pca[:, 1].max() + 1
        )

        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, 0.1),
            np.arange(y_min, y_max, 0.1)
        )

        mesh_points = np.c_[xx.ravel(), yy.ravel()]

        # Kembalikan ke dimensi asli
        mesh_original = pca.inverse_transform(mesh_points)

        Z = svm_model.predict(mesh_original)
        Z = Z.reshape(xx.shape)

        axes[idx].contourf(
            xx,
            yy,
            Z,
            alpha=0.3,
            cmap=plt.cm.RdYlBu
        )

        # Plot data
        for i, color in enumerate(colors):

            axes[idx].scatter(
                X_train_pca[y_train == i, 0],
                X_train_pca[y_train == i, 1],
                c=color,
                label=fruit_names[i],
                edgecolors='black',
                alpha=0.7
            )

        axes[idx].set_title(
            f'{kernel.upper()} Kernel\n'
            f'CV Acc = {cv_scores.mean():.3f} | '
            f'Test Acc = {accuracy:.3f}'
        )

        axes[idx].set_xlabel('PC 1')
        axes[idx].set_ylabel('PC 2')
        axes[idx].legend()

    plt.tight_layout()
    plt.show()

    # =====================================================
    # PERBANDINGAN HASIL
    # =====================================================

    print("\nHASIL PERBANDINGAN KERNEL")
    print("-" * 50)

    print(f"{'Kernel':<10} {'CV Accuracy':<15} {'Test Accuracy'}")

    print("-" * 50)

    for kernel, result in results.items():

        print(
            f"{kernel:<10} "
            f"{result['cv_accuracy']:<15.4f} "
            f"{result['test_accuracy']:.4f}"
        )

    # =====================================================
    # MEMILIH KERNEL TERBAIK
    # =====================================================

    best_kernel = max(
        results,
        key=lambda k: results[k]['test_accuracy']
    )

    print(f"\nKernel Terbaik : {best_kernel.upper()}")

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    best_model = results[best_kernel]['model']

    y_pred_best = best_model.predict(X_test_scaled)

    cm = confusion_matrix(y_test, y_pred_best)

    plt.figure(figsize=(7, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=fruit_names,
        yticklabels=fruit_names
    )

    plt.title(f'Confusion Matrix - {best_kernel.upper()} Kernel')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.show()

    # =====================================================
    # CLASSIFICATION REPORT
    # =====================================================

    print("\nCLASSIFICATION REPORT")
    print("-" * 50)

    print(
        classification_report(
            y_test,
            y_pred_best,
            target_names=fruit_names
        )
    )

    # =====================================================
    # ROC CURVE
    # =====================================================

    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

    ovr_classifier = OneVsRestClassifier(
        svm.SVC(
            kernel=best_kernel,
            probability=True,
            random_state=42
        )
    )

    ovr_classifier.fit(X_train_scaled, y_train)

    y_score = ovr_classifier.predict_proba(X_test_scaled)

    plt.figure(figsize=(10, 8))

    for i in range(len(fruit_names)):

        fpr, tpr, _ = roc_curve(
            y_test_bin[:, i],
            y_score[:, i]
        )

        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            lw=2,
            label=f'{fruit_names[i]} (AUC = {roc_auc:.3f})'
        )

    plt.plot([0, 1], [0, 1], 'k--')

    plt.title(f'ROC Curve - {best_kernel.upper()} Kernel')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    plt.show()

    return results


# =========================================================
# MENJALANKAN PROGRAM
# =========================================================

svm_results = praktikum_svm_fruits()