import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)
previoustime = currenttime = time.time()
detector = htm.HandDetector()
while True:
    success, img = cap.read()
    img = detector.findhands(img)
    landmarklist = detector.findosition(img, draw=False)
    # print(":", landmarklist)
    if landmarklist: print(landmarklist[4])

    currenttime = time.time()
    fps = int(1 / (currenttime - previoustime))
    previoustime = currenttime

    cv2.putText(img, str(fps), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    cv2.imshow('img', img)
    cv2.waitKey(1)