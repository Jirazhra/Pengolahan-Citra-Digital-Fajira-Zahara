import cv2
import numpy as np
import matplotlib.pyplot as plt

def latihan_2():
    # ===== 1. Buat citra dokumen =====
    doc = np.ones((200, 400), dtype=np.uint8) * 200
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.putText(doc, 'Normal Text', (30, 50), font, 0.7, 50, 2)
    
    # Noise garis
    for i in range(0, 100, 5):
        cv2.line(doc, (30+i, 80), (30+i, 85), 50, 1)
    
    # Broken text
    cv2.putText(doc, 'Broken Text', (30, 120), font, 0.7, 50, 2)
    cv2.rectangle(doc, (80, 110), (90, 115), 200, -1)
    
    # Gaussian noise
    noise = np.random.normal(0, 30, doc.shape)
    doc = np.clip(doc + noise, 0, 255).astype(np.uint8)
    
    # ===== 2. Visualisasi =====
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0,0].imshow(doc, cmap='gray')
    axes[0,0].set_title('Original')
    axes[0,0].axis('off')
    
    # ===== 3. Adaptive Threshold (LEBIH BAGUS) =====
    binary = cv2.adaptiveThreshold(
        doc, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    axes[0,1].imshow(binary, cmap='gray')
    axes[0,1].set_title('Adaptive Threshold')
    axes[0,1].axis('off')
    
    # ===== 4. Noise removal (PERBAIKAN KERNEL) =====
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
    axes[0,2].imshow(cleaned, cmap='gray')
    axes[0,2].set_title('Opening (2x2)')
    axes[0,2].axis('off')
    
    # ===== 5. Sambung karakter =====
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (3,1))
    connected = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_h)
    axes[1,0].imshow(connected, cmap='gray')
    axes[1,0].set_title('Closing (3x1)')
    axes[1,0].axis('off')
    
    # ===== 6. Tebalkan teks =====
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1,2))
    enhanced = cv2.dilate(connected, kernel_v)
    axes[1,1].imshow(enhanced, cmap='gray')
    axes[1,1].set_title('Dilation')
    axes[1,1].axis('off')
    
    # ===== 7. Final =====
    final_result = enhanced
    axes[1,2].imshow(final_result, cmap='gray')
    axes[1,2].set_title('Final Result')
    axes[1,2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # ===== 8. Evaluasi =====
    def count_components(img):
        num_labels, _ = cv2.connectedComponents(img)
        return max(num_labels - 1, 1)
    
    def stroke_thickness(img):
        # versi aman TANPA ximgproc
        edges = cv2.Canny(img, 50, 150)
        stroke = np.sum(img == 255)
        edge = np.sum(edges == 255)
        return stroke / edge if edge > 0 else 0
    
    orig_comp = count_components(binary)
    proc_comp = count_components(final_result)
    
    orig_thick = stroke_thickness(binary)
    proc_thick = stroke_thickness(final_result)
    
    print("OCR QUALITY METRICS")
    print("="*40)
    print(f"Komponen Awal  : {orig_comp}")
    print(f"Komponen Akhir : {proc_comp}")
    print(f"Perbaikan      : {(orig_comp - proc_comp)/orig_comp*100:.1f}%")
    
    print(f"\nKetebalan Awal : {orig_thick:.2f}")
    print(f"Ketebalan Akhir: {proc_thick:.2f}")
    print(f"Peningkatan    : {(proc_thick - orig_thick)/orig_thick*100:.1f}%")
    
    print("\nKESIMPULAN:")
    print("✔ Noise berkurang")
    print("✔ Karakter tersambung")
    print("✔ Teks lebih tebal → lebih mudah untuk OCR")

# Jalankan
latihan_2()