"""
Created by: Naren Sadhwani
Date: 21.03.24
"""




import cv2
import mediapipe as mp
import time


class FaceMesh:
    def __init__(self, static_image_mode=False,
                 max_num_faces=1,
                 refine_landmarks=False,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.min_tracking_confidence = min_tracking_confidence
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_detection_confidence = min_detection_confidence

        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.static_image_mode,
                                                 self.max_num_faces,
                                                 self.refine_landmarks,
                                                 self.min_detection_confidence,
                                                 self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils
        self._color = (200, 200, 255)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1, color=self._color)

    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)

        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_FACE_OVAL,
                                                   self.drawSpec, self.drawSpec)
        return img

    def findPosition(self, img, draw=True):

        self.lmList = []
        if self.results.multi_face_landmarks:
            for id, lm in enumerate(self.results.multi_face_landmarks[0].landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return self.lmList


def main():
    cap = cv2.VideoCapture(0)
    startTime = 0

    faceMesh = FaceMesh()

    while True:
        success, img = cap.read()

        if not success:
            break

        faceMesh.findFaceMesh(img)
        # lmList = faceMesh.findPosition(img)

        # if len(lmList) != 0:
        #     print(lmList[0])

        currentTime = time.time()
        fps = 1 / (currentTime - startTime)
        startTime = currentTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# End of file
