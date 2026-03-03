import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_my_image(image_path, sample_path=None):
    """Analyze your own image"""
    
    # =============================
    # LOAD IMAGE
    # =============================
    img = cv2.imread(image_path)
    
    if img is None:
        print("Gambar tidak ditemukan. Periksa path!")
        return
    
    print("=== ANALISIS CITRA SENDIRI ===\n")
    
    # =============================
    # 1. DIMENSI DAN RESOLUSI
    # =============================
    height, width, channels = img.shape
    resolution = width * height
    
    print("1. DIMENSI & RESOLUSI")
    print(f"Width x Height : {width} x {height}")
    print(f"Channels       : {channels}")
    print(f"Resolution     : {resolution:,} pixels\n")
    
    # =============================
    # 2. ASPECT RATIO
    # =============================
    aspect_ratio = width / height
    print("2. ASPECT RATIO")
    print(f"Aspect Ratio   : {aspect_ratio:.2f}\n")
    
    # =============================
    # 3. KONVERSI GRAYSCALE
    # =============================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    rgb_memory = img.size * img.dtype.itemsize
    gray_memory = gray.size * gray.dtype.itemsize
    
    print("3. PERBANDINGAN RGB vs GRAYSCALE")
    print(f"Memory RGB      : {rgb_memory:,} bytes")
    print(f"Memory Grayscale: {gray_memory:,} bytes\n")
    
    # =============================
    # 4. STATISTIK
    # =============================
    print("4. STATISTIK CITRA")
    print(f"Mean  : {img.mean():.2f}")
    print(f"Std   : {img.std():.2f}")
    print(f"Min   : {img.min()}")
    print(f"Max   : {img.max()}\n")
    
    # =============================
    # 5. HISTOGRAM
    # =============================
    print("5. MENAMPILKAN HISTOGRAM")
    
    colors = ('b', 'g', 'r')
    plt.figure(figsize=(10,5))
    
    for i, color in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0,256])
        plt.plot(hist, color=color)
    
    plt.title("Histogram RGB")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.show()
    
    # =============================
    # 6. PERBANDINGAN DENGAN SAMPLE
    # =============================
    if sample_path:
        sample = cv2.imread(sample_path)
        
        if sample is not None:
            h2, w2, _ = sample.shape
            res2 = w2 * h2
            
            print("6. PERBANDINGAN DENGAN SAMPLE")
            print(f"Citra Sendiri  : {resolution:,} pixels")
            print(f"Citra Sample   : {res2:,} pixels")
            
            if resolution > res2:
                print("Citra sendiri memiliki resolusi lebih tinggi.")
            else:
                print("Citra sample memiliki resolusi lebih tinggi.")
    
    # =============================
    # HASIL DIKEMBALIKAN
    # =============================
    analysis_results = {
        "width": width,
        "height": height,
        "resolution": resolution,
        "aspect_ratio": aspect_ratio,
        "mean": float(img.mean()),
        "std": float(img.std()),
        "min": int(img.min()),
        "max": int(img.max()),
        "memory_rgb": rgb_memory,
        "memory_gray": gray_memory
    }
    
    return analysis_results


# ===================================
# CARA PAKAI
# ===================================

result = analyze_my_image("IMG1212.jpg")

print("\n=== HASIL ANALISIS ===")
print(result)
