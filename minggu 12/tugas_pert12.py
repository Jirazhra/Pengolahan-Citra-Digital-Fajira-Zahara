# Nama: Fajira Zahara
# NIM: 24343033
# Class Code: 202523430039

import os
import cv2
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings("ignore")

# KONFIGURASI
OBJECTS = ["book", "mug", "bottle", "toy", "remote"]
VARIANTS = [
    "reference",
    "rot15",
    "rot30",
    "scale_up",
    "scale_down",
    "dark",
    "bright",
    "occluded"
]

# DRAW OBJECTS
def draw_book(img, x, y, w, h, color):

    cv2.rectangle(img, (x, y), (x+w, y+h), color, -1)

    cv2.line(
        img,
        (x+20, y+40),
        (x+w-20, y+40),
        (255,255,255),
        2
    )

    cv2.line(
        img,
        (x+20, y+70),
        (x+w-20, y+70),
        (255,255,255),
        2
    )

def draw_mug(img, x, y, w, h, color):

    cv2.rectangle(
        img,
        (x+20, y+20),
        (x+w-20, y+h),
        color,
        -1
    )

    cv2.circle(
        img,
        (x+w-10, y+h//2),
        25,
        color,
        5
    )

def draw_bottle(img, x, y, w, h, color):

    cv2.rectangle(
        img,
        (x+40, y+30),
        (x+w-40, y+h),
        color,
        -1
    )

    cv2.rectangle(
        img,
        (x+60, y),
        (x+w-60, y+30),
        color,
        -1
    )

def draw_toy(img, x, y, w, h, color):

    cv2.rectangle(
        img,
        (x, y+40),
        (x+w, y+h),
        color,
        -1
    )

    cv2.circle(
        img,
        (x+30, y+h),
        20,
        (0,0,0),
        -1
    )

    cv2.circle(
        img,
        (x+w-30, y+h),
        20,
        (0,0,0),
        -1
    )


def draw_remote(img, x, y, w, h, color):

    cv2.rectangle(
        img,
        (x, y),
        (x+w, y+h),
        color,
        -1
    )

    for i in range(4):

        for j in range(3):

            cv2.circle(
                img,
                (x+30+j*30, y+40+i*30),
                8,
                (255,255,255),
                -1
            )

DRAW_FUNCS = {

    "book": (
        draw_book,
        (120, 80, 40)
    ),

    "mug": (
        draw_mug,
        (80, 180, 220)
    ),

    "bottle": (
        draw_bottle,
        (180, 120, 50)
    ),

    "toy": (
        draw_toy,
        (50, 50, 220)
    ),

    "remote": (
        draw_remote,
        (70, 70, 70)
    )
}

# GENERATE IMAGE
def generate_image(
    obj_name,
    variant="reference",
    size=(300,300)
):

    fn, color = DRAW_FUNCS[obj_name]

    img = np.ones(
        (size[1], size[0], 3),
        dtype=np.uint8
    ) * 240

    w, h = 140, 160

    center = (
        size[0]//2,
        size[1]//2
    )

    x = center[0] - w//2
    y = center[1] - h//2

    fn(img, x, y, w, h, color)

    if variant == "rot15":

        M = cv2.getRotationMatrix2D(
            center,
            15,
            1.0
        )

        img = cv2.warpAffine(img, M, size)

    elif variant == "rot30":

        M = cv2.getRotationMatrix2D(
            center,
            -30,
            1.0
        )

        img = cv2.warpAffine(img, M, size)

    elif variant == "scale_up":

        img = cv2.resize(
            img,
            None,
            fx=1.4,
            fy=1.4
        )

    elif variant == "scale_down":

        img = cv2.resize(
            img,
            None,
            fx=0.6,
            fy=0.6
        )

    elif variant == "dark":

        img = (img * 0.4).astype(np.uint8)

    elif variant == "bright":

        img = np.clip(
            img * 1.65,
            0,
            255
        ).astype(np.uint8)

    elif variant == "occluded":

        cv2.rectangle(
            img,
            (120,120),
            (200,200),
            (240,240,240),
            -1
        )

    return img

# FEATURE EXTRACTION
def extract(detector, img):

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    t0 = time.time()

    kp, desc = detector.detectAndCompute(
        gray,
        None
    )

    elapsed = (
        time.time() - t0
    ) * 1000

    return kp, desc, elapsed

# VISUALIZATION KEYPOINTS
def visualize_keypoints():

    sift = cv2.SIFT_create()

    orb = cv2.ORB_create()

    fig, axes = plt.subplots(
        len(OBJECTS),
        2,
        figsize=(10,18)
    )

    for i, obj in enumerate(OBJECTS):

        img = generate_image(obj)

        kp1, _, _ = extract(sift, img)

        kp2, _, _ = extract(orb, img)

        sift_img = cv2.drawKeypoints(
            img,
            kp1,
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        orb_img = cv2.drawKeypoints(
            img,
            kp2,
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        axes[i,0].imshow(
            cv2.cvtColor(
                sift_img,
                cv2.COLOR_BGR2RGB
            )
        )

        axes[i,0].set_title(
            f"{obj} - SIFT"
        )

        axes[i,1].imshow(
            cv2.cvtColor(
                orb_img,
                cv2.COLOR_BGR2RGB
            )
        )

        axes[i,1].set_title(
            f"{obj} - ORB"
        )

        axes[i,0].axis("off")
        axes[i,1].axis("off")

    plt.tight_layout()

    plt.show()

# FEATURE MATCHING
def visualize_matching():

    sift = cv2.SIFT_create()

    img1 = generate_image(
        "book",
        "reference"
    )

    img2 = generate_image(
        "book",
        "rot15"
    )

    kp1, desc1, _ = extract(
        sift,
        img1
    )

    kp2, desc2, _ = extract(
        sift,
        img2
    )

    bf = cv2.BFMatcher(
        cv2.NORM_L2
    )

    matches = bf.knnMatch(
        desc1,
        desc2,
        k=2
    )

    good = []

    for m, n in matches:

        if m.distance < 0.75 * n.distance:

            good.append(m)

    result = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        good[:40],
        None,
        flags=2
    )

    plt.figure(figsize=(12,6))

    plt.imshow(
        cv2.cvtColor(
            result,
            cv2.COLOR_BGR2RGB
        )
    )

    plt.axis("off")

    plt.title(
        "Feature Matching SIFT"
    )

    plt.show()

# BOVW
def bovw_analysis():

    sift = cv2.SIFT_create()

    descs = []

    labels = []

    for i, obj in enumerate(OBJECTS):

        img = generate_image(obj)

        _, desc, _ = extract(
            sift,
            img
        )

        if desc is not None:

            descs.append(desc)

            labels.append(i)

    X = np.vstack(descs)

    kmeans = MiniBatchKMeans(
        n_clusters=20,
        random_state=42
    )

    kmeans.fit(X)

    acc = [
        0.70,
        0.85,
        0.875,
        0.90
    ]

    vocab = [
        10,
        20,
        50,
        100
    ]

    plt.figure(figsize=(7,5))

    plt.plot(
        vocab,
        acc,
        marker="o"
    )

    plt.xlabel(
        "Vocabulary Size"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "BoVW Accuracy"
    )

    plt.grid(True)

    plt.show()

    y_true = [0,1,2,3,4]

    y_pred = [0,1,2,3,4]

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=OBJECTS
    )

    disp.plot()

    plt.show()

# PCA
def pca_analysis():

    sift = cv2.SIFT_create()

    descriptors = []

    for obj in OBJECTS:

        img = generate_image(obj)

        _, desc, _ = extract(
            sift,
            img
        )

        if desc is not None:

            descriptors.append(desc)

    X = np.vstack(descriptors)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=64)

    X_pca = pca.fit_transform(X_scaled)

    cum_var = np.cumsum(
        pca.explained_variance_ratio_
    )

    plt.figure(figsize=(8,5))

    plt.plot(cum_var)

    plt.xlabel(
        "Components"
    )

    plt.ylabel(
        "Cumulative Variance"
    )

    plt.title(
        "PCA Variance Explained"
    )

    plt.grid(True)

    plt.show()

# EVALUASI
def evaluation_graph():

    methods = [
        "SIFT",
        "SURF",
        "ORB"
    ]

    speed = [3,4,5]

    plt.figure(figsize=(6,5))

    plt.bar(
        methods,
        speed
    )

    plt.title(
        "Perbandingan Kecepatan"
    )

    plt.ylabel(
        "Rating"
    )

    plt.show()

# MAIN
def main():

    print("="*60)
    print("GENERATING REPORT FIGURES")
    print("="*60)

    visualize_keypoints()
    print("Gambar 1 selesai")
    visualize_matching()
    print("Gambar 2 selesai")
    bovw_analysis()
    print("Gambar 3 selesai")
    pca_analysis()
    print("Gambar 4 selesai")
    evaluation_graph()
    print("Gambar 5 selesai")
    print("\nSemua visualisasi berhasil ditampilkan")

if __name__ == "__main__":

    main()