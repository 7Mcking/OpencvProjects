"""
Created by: Naren Sadhwani
Date: 21.03.24
"""

import cv2
import time
from handTrackingModule import HandDetector


def main():
    # Initialize the previous time
    startTime = 0
    endTime = 0

    # Initialize the camera
    cap = cv2.VideoCapture(0)

    # Initialize the hand detector
    detector = HandDetector(detectionCon=0.8)

    # ID of the fingertips
    tipIds = [4, 8, 12, 16, 20]

    # Initialize the variables
    right_hand = False
    left_hand = False

    while True:
        # Read the frame from the camera
        success, img = cap.read()

        # Find the hand landmarks
        img = detector.findHands(img)

        # Get the landmarks
        lmList = detector.findPosition(img, draw=False)

        # If the landmarks are detected
        if len(lmList) != 0:
            fingers = []

            # Figure if the hand is right hand or left hand
            if lmList[tipIds[0]][1] > lmList[tipIds[4]][1]:
                right_hand = True
                left_hand = False
                # print("Right Hand")
            else:
                right_hand = False
                left_hand = True
                # print("Left Hand")

            # Thumb
            if right_hand:
                if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # 4 Fingers
            for id in range(1, 5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            totalFingers =  fingers.count(1)
            print(totalFingers)


            cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), 2)
            cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 255, 255),
                        10)


        # Calculate the FPS
        endTime = time.time()
        fps = 1 / (endTime - startTime)
        startTime = endTime

        # Display the FPS
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

        # Show the image
        cv2.imshow("Image", img)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
