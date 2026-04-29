# Nama: Fajira Zahara
# NIM: 24343033
# Class Code: 202523430039

import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import os
from skimage import measure

def praktikum_segmentasi_lengkap():

    print("PRAKTIKUM SEGMENTASI (INPUT GAMBAR)")
    print("=" * 60)

    # LOAD IMAGE
    def load_images():
        data = {}

        paths = {
            'Bimodal': ('p1.jpeg', 'gt_p1.png'),
            'Uneven': ('p2.png', 'gt_p2.png'),
            'Overlap': ('p3.jpg', 'gt_p3.png')
        }

        for name, (img_path, gt_path) in paths.items():
            img = cv2.imread(img_path, 0)

            if img is None:
                print(f"[ERROR] {img_path} tidak ditemukan!")
                continue

            # Ground truth
            if os.path.exists(gt_path):
                gt = cv2.imread(gt_path, 0)
            else:
                print(f"[INFO] {gt_path} tidak ditemukan → pakai GT sederhana")
                _, gt = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

            data[name] = (img, gt)

        return data

    # THRESHOLDING
    def thresholding(img):
        results = {}

        _, global_th = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        _, otsu = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        mean = cv2.adaptiveThreshold(img,255,
                cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

        gauss = cv2.adaptiveThreshold(img,255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

        results['Global'] = global_th
        results['Otsu'] = otsu
        results['Adaptive Mean'] = mean
        results['Adaptive Gaussian'] = gauss

        return results

    # EDGE DETECTION
    def edge(img):
        results = {}

        sx = cv2.Sobel(img,cv2.CV_64F,1,0)
        sy = cv2.Sobel(img,cv2.CV_64F,0,1)
        sobel = np.sqrt(sx**2 + sy**2).astype(np.uint8)

        prewittx = cv2.filter2D(img,-1,np.array([[1,0,-1],[1,0,-1],[1,0,-1]]))
        prewitty = cv2.filter2D(img,-1,np.array([[1,1,1],[0,0,0],[-1,-1,-1]]))
        prewitt = cv2.magnitude(prewittx.astype(float), prewitty.astype(float))

        results['Canny_1'] = cv2.Canny(img,30,100)
        results['Canny_2'] = cv2.Canny(img,50,150)
        results['Canny_3'] = cv2.Canny(img,100,200)

        results['Sobel'] = sobel
        results['Prewitt'] = prewitt

        return results

    # REGION GROWING
    def region_growing(img, seed):
        h,w = img.shape
        visited = np.zeros_like(img)
        result = np.zeros_like(img)

        stack = [seed]
        seed_val = img[seed]
        threshold = 10

        while stack:
            x,y = stack.pop()
            if visited[x,y]: continue
            visited[x,y] = 1

            if abs(int(img[x,y]) - int(seed_val)) < threshold:
                result[x,y] = 255

                for dx in [-1,0,1]:
                    for dy in [-1,0,1]:
                        nx, ny = x+dx, y+dy
                        if 0<=nx<h and 0<=ny<w:
                            stack.append((nx,ny))

        return result

    # WATERSHED + CC
    def watershed(img):
        _,th = cv2.threshold(img,0,255,cv2.THRESH_OTSU)
        kernel = np.ones((3,3),np.uint8)

        opening = cv2.morphologyEx(th,cv2.MORPH_OPEN,kernel)
        dist = cv2.distanceTransform(opening,cv2.DIST_L2,5)

        _,fg = cv2.threshold(dist,0.7*dist.max(),255,0)
        fg = np.uint8(fg)

        _,markers = cv2.connectedComponents(fg)
        markers += 1

        img_color = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(img_color,markers)

        res = np.zeros_like(img)
        res[markers>1] = 255
        return res

    def connected(img):
        labels = measure.label(img)
        return (labels>0).astype(np.uint8)*255

    # METRIK
    def metrics(pred, gt):
        pred = (pred>0).astype(np.uint8)
        gt = (gt>0).astype(np.uint8)

        tp = np.sum((pred==1)&(gt==1))
        fp = np.sum((pred==1)&(gt==0))
        fn = np.sum((pred==0)&(gt==1))
        tn = np.sum((pred==0)&(gt==0))

        acc = (tp+tn)/(tp+tn+fp+fn+1e-6)
        prec = tp/(tp+fp+1e-6)
        rec = tp/(tp+fn+1e-6)
        iou = tp/(tp+fp+fn+1e-6)
        dice = (2*tp)/(2*tp+fp+fn+1e-6)

        return acc,prec,rec,iou,dice

    # OVERLAY
    def overlay(img, mask):
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cnt,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_color,cnt,-1,(0,255,0),2)
        return img_color

    # MAIN
    data = load_images()

    for name,(img,gt) in data.items():
        print(f"\n=== {name} ===")

        th = thresholding(img)
        ed = edge(img)
        rg = region_growing(img,(img.shape[0]//2, img.shape[1]//2))
        ws = watershed(img)
        cc = connected(th['Otsu'])

        all_methods = {**th, **ed,
                       'Region Growing': rg,
                       'Watershed': ws,
                       'Connected': cc}

        print("Method | Acc | Prec | Rec | IoU | Dice | Time")
        print("-"*60)

        for m,res in all_methods.items():
            start = time.time()
            acc,prec,rec,iou,dice = metrics(res,gt)
            t = time.time()-start

            print(f"{m:15} {acc:.2f} {prec:.2f} {rec:.2f} {iou:.2f} {dice:.2f} {t:.4f}")

        # VISUAL
        fig,axs = plt.subplots(3,4,figsize=(12,8))

        axs[0,0].imshow(img,cmap='gray'); axs[0,0].set_title("Original")
        axs[0,1].imshow(gt,cmap='gray'); axs[0,1].set_title("Ground Truth")

        axs[0,2].imshow(th['Otsu'],cmap='gray')
        axs[0,3].imshow(th['Adaptive Mean'],cmap='gray')

        axs[1,0].imshow(ed['Canny_1'],cmap='gray')
        axs[1,1].imshow(ed['Sobel'],cmap='gray')

        axs[1,2].imshow(ws,cmap='gray')
        axs[1,3].imshow(rg,cmap='gray')

        axs[2,0].imshow(overlay(img,th['Otsu']))
        axs[2,1].imshow(overlay(img,ws))
        axs[2,2].imshow(overlay(img,rg))
        axs[2,3].axis('off')

        for ax in axs.ravel():
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    print("\nKESIMPULAN:")
    print("""
- Otsu cocok untuk bimodal
- Adaptive cocok untuk iluminasi tidak merata
- Watershed terbaik untuk overlap
- Canny stabil untuk edge
- Region Growing tergantung seed
    """)

# RUN
praktikum_segmentasi_lengkap()