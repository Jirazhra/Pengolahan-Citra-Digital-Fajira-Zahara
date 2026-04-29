import cv2
import numpy as np
import matplotlib.pyplot as plt

def latihan_1():
    # Buat citra biner test pattern
    img = np.zeros((200, 300), dtype=np.uint8)
    
    # Tambahkan berbagai bentuk
    cv2.rectangle(img, (30, 30), (80, 80), 255, -1)      # Square
    cv2.circle(img, (150, 50), 20, 255, -1)              # Circle
    cv2.rectangle(img, (200, 30), (220, 70), 255, -1)    # Vertical line
    cv2.rectangle(img, (250, 40), (270, 60), 255, -1)    # Small box
    
    # Tambahkan noise (salt & pepper)
    noise = np.random.rand(*img.shape) < 0.05
    img_noisy = img.copy()
    img_noisy[noise] = 255 - img_noisy[noise]
    
    # Structuring elements
    kernels = {
        '3x3 Rect': cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)),
        '5x5 Rect': cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)),
        '3x3 Ellipse': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)),
        '3x3 Cross': cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    }
    
    operations = ['Erosion', 'Dilation', 'Opening', 'Closing']
    
    fig, axes = plt.subplots(len(operations) + 1, len(kernels) + 1, figsize=(16, 10))
    
    # ===== Baris pertama (Original) =====
    axes[0,0].imshow(img, cmap='gray')
    axes[0,0].set_title('Clean')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(img_noisy, cmap='gray')
    axes[0,1].set_title('Noisy')
    axes[0,1].axis('off')
    
    for j in range(2, len(kernels)+1):
        axes[0,j].axis('off')
    
    # ===== Operasi morfologi =====
    for i, op_name in enumerate(operations, 1):
        axes[i,0].text(0.5, 0.5, op_name, fontsize=10, ha='center')
        axes[i,0].axis('off')
        
        for j, (kernel_name, kernel) in enumerate(kernels.items(), 1):
            
            if op_name == 'Erosion':
                result = cv2.erode(img_noisy, kernel, iterations=1)
            elif op_name == 'Dilation':
                result = cv2.dilate(img_noisy, kernel, iterations=1)
            elif op_name == 'Opening':
                result = cv2.morphologyEx(img_noisy, cv2.MORPH_OPEN, kernel)
            elif op_name == 'Closing':
                result = cv2.morphologyEx(img_noisy, cv2.MORPH_CLOSE, kernel)
            
            axes[i,j].imshow(result, cmap='gray')
            axes[i,j].set_title(kernel_name)
            axes[i,j].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # ===== Analisis =====
    print("ANALISIS OPERASI MORFOLOGI")
    print("=" * 40)
    print("1. Erosion  : Mengikis objek & menghilangkan noise kecil")
    print("2. Dilation : Memperbesar objek & mengisi celah")
    print("3. Opening  : Menghapus noise (erosion → dilation)")
    print("4. Closing  : Menutup lubang kecil (dilation → erosion)")

# Jalankan
latihan_1()