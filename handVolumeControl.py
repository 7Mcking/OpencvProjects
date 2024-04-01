"""
Created by: Naren Sadhwani
Date: 21.03.24
"""
from alsaaudio import *
from handTrackingModule import HandDetector
import cv2
import math
import time
import numpy as np

volume = Mixer()

min_vol = 0
max_vol = 100

startTime = 0
endTime = 0


def main():
    global startTime, endTime
    volume_set_allowed = False
    time_start = None

    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = cap.read()
        if not success:
            print("Error in capturing video")
            break

        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:  # If hand is found
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Draw line and circles
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

            # Calculate length
            length = math.hypot(x2 - x1, y2 - y1)

            # Start timer if not already started
            if time_start is None:
                time_start = time.time()  # start the timer

            # If more than one second has passed
            elif time.time() - time_start > 1:
                volume_set_allowed = True

            if volume_set_allowed:
                cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)  # draw a green circle when volume set is allowed
                vol = np.interp(length, [25, 200], [min_vol, max_vol])
                volume.setvolume(int(vol))
                print("Volume set!")
                volume_set_allowed = False

        else:  # If hand is not found
            volume_set_allowed = False
            time_start = None

        # Add a volume bar
        cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 255), 2)
        cv2.rectangle(img, (50, int(400 - (400 - 150) * volume.getvolume()[0] / 100)), (85, 400),
                      (0, 255, 255), cv2.FILLED)

        # Add a volume percentage
        cv2.putText(img, f'{int(volume.getvolume()[0])}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
                    (0, 255, 255), 2)

        # Calculate FPS
        endTime = time.time()
        fps = 1 / (endTime - startTime)
        startTime = endTime
        cv2.putText(img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 255), 2)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
