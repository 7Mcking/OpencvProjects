"""
Created by: Naren Sadhwani
Date: 21.03.24
"""

import cv2
import numpy as np
import handTrackingModule as htm
import pyautogui
import time

width, height = 640, 480
frmeR = 100
smoothening = 1

startTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

detector = htm.HandDetector(maxHands=1)
wScr, hScr = pyautogui.size()

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    imList = detector.findPosition(img, draw=False)

    if len(imList) == 0:
        continue
    if len(imList) != 0:
        x1, y1 = imList[8][1:]
        x2, y2 = imList[12][1:]

        fingers = detector.fingersUp()
        cv2.rectangle(img, (frmeR, frmeR), (width - frmeR, height - frmeR), (255, 0, 255), 2)

        if fingers[1] == 1 and fingers[2] == 0:
            x3 = np.interp(x1, (frmeR, width - frmeR), (0, wScr))
            y3 = np.interp(y1, (frmeR, height - frmeR), (0, hScr))

            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            pyautogui.moveTo(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        if fingers[1] == 1 and fingers[2] == 1:
            length, img, lineInfo = detector.findDistance(8, 12, img)
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                pyautogui.click()

    # FPS
    currentTime = time.time()
    fps = 1 / (currentTime - startTime)
    startTime = currentTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    # Display
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
