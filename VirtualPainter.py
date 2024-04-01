"""
Created by: Naren Sadhwani
Date: 21.03.24
"""

import cv2
import numpy as np
import time
import os
import handTrackingModule as htm

### Parameters
brushThickness = 15
eraserThickness = 100

folderPath = "Images/"

myList = os.listdir(folderPath)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
header = overlayList[0]
drawColor = (255, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 1200)
cap.set(4, 600)
imgCanvas = np.zeros((600, 1200, 3), np.uint8)

cv2.namedWindow("Virtual Painter", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Virtual Painter", imgCanvas)

cv2.destroyAllWindows()
cap.release()
