import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture('model/06.mp4')
mppose = mp.solutions.pose()
previoustime = time.time()


while True:
    ret, frame = cap.read()

    currenttime = time.time()
    fps = int(1 / (currenttime - previoustime))
    previoustime = currenttime

    cv2.putText(frame, str(fps), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, \
                (0, 0, 255), 3)

    cv2.imshow("Image", frame)
    cv2.waitKey(1)

