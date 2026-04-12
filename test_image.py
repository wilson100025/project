import cv2
import numpy as np

# 1. 讀取影像 (注意：不能用 0，必須讀取彩色資訊才能轉 HSV)
img = cv2.imread(r'D:\project\project\input_images\firefly_2.png', 1)

if img is not None:
    # 1. 放大圖片方便觀察
    scale = 30
    h, w = img.shape[:2]
    img_large = cv2.resize(img, (w*scale, h*scale), interpolation=cv2.INTER_NEAREST)

    # 2. 轉 HSV 看看
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 找出圖片中最亮像素的座標
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    
    print(f"整張圖最亮的像素座標在: {maxLoc}")
    print(f"該點的 BGR 數據為: {img[maxLoc[1], maxLoc[0]]}")
    print(f"該點的 HSV 數據為: {hsv[maxLoc[1], maxLoc[0]]}")

    cv2.imshow('Zoomed Image', img_large)
    cv2.waitKey(0)