import cv2
import mediapipe as mp

cap = cv2.VideoCapture("IMG_8456.mov")

mpHands = mp.solutions.hands
hands = mpHands.Hands(False, 2, 1, 0.136, 0.3)
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handsMulti in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handsMulti, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
