import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import myfunc

cap = cv2.VideoCapture(0)

hands = mp.tasks.vision.HandLandmarksConnections
# print(mpHands)

# hands = mpHands.Hands()

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # result = hands.process(imgRGB)

    cv2.imshow('img', img)
    cv2.waitKey(1)
