# ============================================
# PRAKTIKUM: DIGITALISASI DAN ANALISIS CITRA
# ============================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

print("=== PRAKTIKUM: DIGITALISASI DAN ANALISIS CITRA ===")
print("Materi: Representasi Citra, Resolusi, Bit Depth, Aspect Ratio\n")

# =============== FUNGSI BANTU ===============

def load_local_image(path):
    """Load image from local directory"""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError("Gambar tidak ditemukan!")
    return img


def analyze_image_properties(img, name="Image"):
    """Analyze and display image properties"""
    if len(img.shape) == 2:
        height, width = img.shape
        channels = 1
    else:
        height, width, channels = img.shape

    resolution = width * height
    aspect_ratio = width / height
    bit_depth = img.dtype.itemsize * 8

    print(f"\n{'='*45}")
    print(f"ANALISIS CITRA: {name}")
    print(f"{'='*45}")
    print(f"Dimensi           : {width} x {height}")
    print(f"Jumlah Channel    : {channels}")
    print(f"Resolusi          : {resolution:,} piksel")
    print(f"Aspect Ratio      : {aspect_ratio:.2f}")
    print(f"Bit Depth         : {bit_depth}-bit")

    memory_bytes = img.size * img.dtype.itemsize
    memory_kb = memory_bytes / 1024
    memory_mb = memory_kb / 1024

    print(f"Ukuran Memori     : {memory_bytes:,} bytes")
    print(f"                    {memory_kb:.2f} KB")
    print(f"                    {memory_mb:.2f} MB")

    if channels == 1:
        print(f"Rentang Intensitas: [{img.min()}, {img.max()}]")
        print(f"Rata-rata         : {img.mean():.2f}")
        print(f"Standar Deviasi   : {img.std():.2f}")

    return {
        "width": width,
        "height": height,
        "channels": channels,
        "resolution": resolution,
        "aspect_ratio": aspect_ratio,
        "bit_depth": bit_depth,
        "memory_bytes": memory_bytes
    }


def display_image_grid(images, titles, rows, cols, figsize=(15, 10)):
    """Display images in grid layout"""
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.ravel()

    for i, (img, title) in enumerate(zip(images, titles)):
        if len(img.shape) == 2:
            axes[i].imshow(img, cmap="gray")
        else:
            axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i].set_title(title)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


# =============== MAIN PRAKTIKUM ===============

# 1. LOAD CITRA DIGITAL
print("1. MEMUAT CITRA DIGITAL")
original_img = load_local_image("foto_poster.jpeg")
props_original = analyze_image_properties(original_img, "Citra Asli (RGB)")


# 2. REPRESENTASI SEBAGAI MATRIKS
print("\n\n2. REPRESENTASI CITRA SEBAGAI MATRIKS")
x, y = 100, 100
pixel_value = original_img[x, y]
print(f"Nilai piksel pada posisi ({x},{y}) [BGR]: {pixel_value}")

print("\nMatriks 5x5 piksel pertama (channel Biru):")
print(original_img[:5, :5, 0])


# 3. REPRESENTASI SEBAGAI VEKTOR
print("\n\n3. REPRESENTASI CITRA SEBAGAI VEKTOR")
vector_img = original_img.flatten()
print("25 elemen pertama vektor:")
print(vector_img[:25])
print("Panjang vektor:", len(vector_img))


# 4. KONVERSI KE GRAYSCALE
print("\n\n4. KONVERSI KE GRAYSCALE")
gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
props_gray = analyze_image_properties(gray_img, "Citra Grayscale")


# 5. ANALISIS BIT DEPTH
print("\n\n5. ANALISIS PENGARUH BIT DEPTH")
img_8bit = gray_img
img_4bit = (gray_img // 16) * 16
img_2bit = (gray_img // 64) * 64
img_1bit = (gray_img // 128) * 255

images = [img_8bit, img_4bit, img_2bit, img_1bit]
titles = ["8-bit (256 level)", "4-bit (16 level)", "2-bit (4 level)", "1-bit (2 level)"]
display_image_grid(images, titles, 1, 4, figsize=(16, 4))


# 6. MANIPULASI DASAR CITRA
print("\n\n6. MANIPULASI DASAR CITRA")

rotated = cv2.rotate(original_img, cv2.ROTATE_90_CLOCKWISE)
cropped = rotated[500:2500, 800:3200]
resized = cv2.resize(cropped, (640, 480))

images = [original_img, rotated, cropped, resized]
titles = ["Asli", "Rotasi 90°", "Cropping", "Resizing"]
display_image_grid(images, titles, 2, 2, figsize=(12, 8))


# 7. SEPARASI CHANNEL WARNA
print("\n\n7. SEPARASI CHANNEL WARNA RGB")
b, g, r = cv2.split(original_img)
zeros = np.zeros_like(b)

blue = cv2.merge([b, zeros, zeros])
green = cv2.merge([zeros, g, zeros])
red = cv2.merge([zeros, zeros, r])

images = [original_img, blue, green, red]
titles = ["Original", "Blue Channel", "Green Channel", "Red Channel"]
display_image_grid(images, titles, 2, 2, figsize=(12, 8))


# 8. ANALISIS HISTOGRAM
print("\n\n8. ANALISIS HISTOGRAM INTENSITAS")

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.hist(gray_img.ravel(), 256, [0, 256], color="gray")
plt.title("Histogram Grayscale")

plt.subplot(1, 3, 2)
colors = ("b", "g", "r")
for i, col in enumerate(colors):
    hist = cv2.calcHist([original_img], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
plt.title("Histogram RGB")

plt.subplot(1, 3, 3)
cumulative = np.cumsum(np.histogram(gray_img.ravel(), 256, [0, 256])[0])
plt.plot(cumulative, color="purple")
plt.title("Histogram Kumulatif")

plt.tight_layout()
plt.show()


# 9. ANALISIS UKURAN MEMORI
print("\n\n9. ANALISIS UKURAN MEMORI")

sizes = [(640, 480), (1920, 1080), (3840, 2160)]
formats = [("Grayscale", 1), ("RGB", 3)]

print(f"\n{'Resolusi':<15}{'Format':<15}{'Ukuran (MB)':>15}")
print("-"*45)

for w, h in sizes:
    for fmt, ch in formats:
        mem = w * h * ch
        mem_mb = mem / (1024 * 1024)
        print(f"{w}x{h:<9}{fmt:<15}{mem_mb:>10.2f}")

print("\n=== PRAKTIKUM SELESAI ===")