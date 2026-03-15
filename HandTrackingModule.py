'''learning from: https://www.youtube.com/watch?v=01sAkU_NvOY'''

import cv2
import mediapipe as mp
import time



class HandDetector():
    def __init__(self,
        static_image_mode = False,
        max_num_hands = 2,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5):

        self.mode = static_image_mode
        self.maxhands = max_num_hands
        self.detectionCon   = min_detection_confidence
        self.trackingCon    = min_tracking_confidence

        self.mpHands    = mp.solutions.hands
        self.hands      = self.mpHands.Hands(self.mode, self.maxhands,
                                             self.detectionCon, self.trackingCon)
        self.mpDraw     = mp.solutions.drawing_utils

    def findhands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handlms, self.mpHands.HAND_CONNECTIONS)
                    # h, w, c = img.shape
                    # cx, cy = int(landmark.x*w), int(landmark.y*h)
                    # if id == 0: cv2.circle(img, (cx, cy), 25, (255, 0, 0), -1)

        return img

    def findosition(self, img, handno=0, draw=True):
        # myhand = self.results.multi_hand_landmarks[handno]
        landmarklist = []
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handno]
            for id, landmark in enumerate(myhand.landmark):
                h, w, c = img.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                # print(id, cx, cy)
                landmarklist.append([id, cx, cy])
                if draw: cv2.circle(img, (cx, cy), 15, (0, 255, 255), cv2.FILLED)

        return landmarklist


def main():
    cap = cv2.VideoCapture(0)
    previoustime = currenttime = time.time()
    detector = HandDetector()
    while True:
        success, img = cap.read()
        img = detector.findhands(img)
        landmarklist = detector.findosition(img)
        if len(landmarklist) != 0: print(landmarklist[4])

        currenttime = time.time()
        fps = int(1 / (currenttime - previoustime))
        previoustime = currenttime

        cv2.putText(img, str(fps), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.imshow('img', img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()