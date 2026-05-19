import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings('ignore')


# =========================================================
# PERBANDINGAN KNN vs SVM
# =========================================================

def compare_knn_svm():

    print("\nPERBANDINGAN KNN vs SVM")
    print("=" * 50)

    # =====================================================
    # LOAD DATASET IRIS
    # =====================================================

    iris = load_iris()

    X = iris.data
    y = iris.target

    class_names = iris.target_names

    print(f"Jumlah Data     : {X.shape[0]}")
    print(f"Jumlah Fitur    : {X.shape[1]}")
    print(f"Jumlah Kelas    : {len(class_names)}")

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
    # INISIALISASI MODEL
    # =====================================================

    knn = KNeighborsClassifier(
        n_neighbors=5
    )

    svm_linear = svm.SVC(
        kernel='linear',
        C=1.0,
        probability=True,
        random_state=42
    )

    svm_rbf = svm.SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,
        random_state=42
    )

    models = {
        'KNN (k=5)': knn,
        'SVM Linear': svm_linear,
        'SVM RBF': svm_rbf
    }

    # =====================================================
    # TRAINING DAN EVALUASI
    # =====================================================

    results = {}

    for name, model in models.items():

        print(f"\nTraining Model : {name}")

        # Cross Validation
        cv_scores = cross_val_score(
            model,
            X_train_scaled,
            y_train,
            cv=5
        )

        # Training
        model.fit(X_train_scaled, y_train)

        # Prediksi
        y_pred = model.predict(X_test_scaled)

        # Perhitungan Metrics
        accuracy = model.score(X_test_scaled, y_test)

        precision = precision_score(
            y_test,
            y_pred,
            average='weighted'
        )

        recall = recall_score(
            y_test,
            y_pred,
            average='weighted'
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average='weighted'
        )

        # Simpan hasil
        results[name] = {
            'cv_accuracy': cv_scores.mean(),
            'test_accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'model': model,
            'prediction': y_pred
        }

    # =====================================================
    # TAMPILKAN HASIL PERBANDINGAN
    # =====================================================

    print("\nHASIL PERBANDINGAN MODEL")
    print("-" * 85)

    print(
        f"{'Model':<15}"
        f"{'CV Acc':<12}"
        f"{'Test Acc':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1-Score':<12}"
    )

    print("-" * 85)

    for name, result in results.items():

        print(
            f"{name:<15}"
            f"{result['cv_accuracy']:<12.4f}"
            f"{result['test_accuracy']:<12.4f}"
            f"{result['precision']:<12.4f}"
            f"{result['recall']:<12.4f}"
            f"{result['f1_score']:<12.4f}"
        )

    # =====================================================
    # VISUALISASI PERBANDINGAN METRICS
    # =====================================================

    metrics = [
        'cv_accuracy',
        'test_accuracy',
        'precision',
        'recall',
        'f1_score'
    ]

    metric_titles = [
        'CV Accuracy',
        'Test Accuracy',
        'Precision',
        'Recall',
        'F1-Score'
    ]

    model_names = list(models.keys())

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    axes = axes.ravel()

    for idx, metric in enumerate(metrics):

        values = [
            results[name][metric]
            for name in model_names
        ]

        bars = axes[idx].bar(
            model_names,
            values
        )

        axes[idx].set_title(metric_titles[idx])

        axes[idx].set_ylabel('Score')

        axes[idx].set_ylim(0, 1.1)

        axes[idx].grid(True, alpha=0.3, axis='y')

        # Menampilkan nilai pada bar
        for bar, value in zip(bars, values):

            axes[idx].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f'{value:.3f}',
                ha='center',
                va='bottom'
            )

    # Sembunyikan subplot kosong
    axes[-1].axis('off')

    plt.suptitle(
        'Perbandingan Performa KNN dan SVM',
        fontsize=16
    )

    plt.tight_layout()

    plt.show()

    # =====================================================
    # CONFUSION MATRIX SETIAP MODEL
    # =====================================================

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (name, result) in enumerate(results.items()):

        cm = confusion_matrix(
            y_test,
            result['prediction']
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[idx]
        )

        axes[idx].set_title(name)

        axes[idx].set_xlabel('Predicted Label')

        axes[idx].set_ylabel('True Label')

    plt.suptitle(
        'Confusion Matrix Setiap Model',
        fontsize=16
    )

    plt.tight_layout()

    plt.show()

    # =====================================================
    # CLASSIFICATION REPORT
    # =====================================================

    for name, result in results.items():

        print(f"\nCLASSIFICATION REPORT : {name}")

        print("-" * 60)

        print(
            classification_report(
                y_test,
                result['prediction'],
                target_names=class_names
            )
        )

    return results


# =========================================================
# MENJALANKAN PROGRAM
# =========================================================

comparison_results = compare_knn_svm()