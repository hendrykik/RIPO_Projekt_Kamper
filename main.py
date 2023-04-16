import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

#fps
#time
curent = 0
previous = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handsMulti in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handsMulti, mpHands.HAND_CONNECTIONS)

    curent = time.time()
    fps = 1/(curent-previous)
    previous = curent

    cv2.putText(img, str(fps),(10,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
