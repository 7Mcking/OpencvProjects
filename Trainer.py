"""
Created by: Naren Sadhwani
Date: 21.03.24
"""


import cv2
from poseEstimationModule import PoseDetector
import math, time, numpy as np

def main():
    # Video Capture
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('./Resources/bicepCurls3.mp4')

    # Pose Detector
    detector = PoseDetector(min_detection_confidence=0.5)

    # FPS Counter
    startTime = 0
    endTime = None

    # Counter
    count = 0
    dir = 0
    while True:
        success, img = cap.read()
        if success:
            img = detector.findPose(img)
            lmList = detector.findPosition(img, draw=False)
            if len(lmList) != 0:
                p1, p2, p3 = lmList[11][1:], lmList[13][1:], lmList[15][1:]
                # Right Arm
                # angle = detector.findAngle(img, p1, p2, p3, draw=True)
                # print(angle)

                # Left Arm
                angle = detector.findAngle(img, p1, p2, p3, draw=True)

                per = np.interp(angle, (210, 310), (0, 100))
                #print(per)
                bar = np.interp(angle, (210, 310), (650, 100))
                print(bar)


                color = (255, 0, 255)
                if per >85:
                    color = (0, 255, 0)
                    if dir == 0:
                        count += 0.5
                        dir = 1
                if per == 0:
                    color = (0, 255, 0)
                    if dir == 1:
                        count += 0.5
                        dir = 0
                print(count)

                # Draw Bar
                cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
                cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
                cv2.putText(img, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4,
                            color, 4)

                # Draw Counter
                cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15,
                            (255, 0, 0), 25)

            # FPS Counter
        endTime = time.time()
        fps = 1 / (endTime - startTime)
        startTime = endTime

        cv2.putText(img, str(int(fps)), (300, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


