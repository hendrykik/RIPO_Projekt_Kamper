import cv2
import mediapipe as mp
import numpy as np

def sprawdz_wspolrzedne(x, y, shp):
    num_vertices = len(shp)
    inside = False

    # sprawdzanie czy punkt jest pod markiza
    # shp: lista zawierająca wierzchołki obszaru w kolejności przeciwnie do ruchu wskazówek zegara
    # dlatego sprawdzamy po krawędziach czy punkt jest na zewnątrz
    for i in range(num_vertices):
        x1, y1 = shp[i]
        x2, y2 = shp[(i + 1) % num_vertices]

        if (y1 < y and y2 >= y) or (y2 < y and y1 >= y):
            if x1 + (y - y1) / (y2 - y1) * (x2 - x1) < x:
                inside = not inside

    return inside

cap = cv2.VideoCapture("IMG_8456.mov")

mpHands = mp.solutions.hands
hands = mpHands.Hands(False, 2, 1, 0.136, 0.3)
mpDraw = mp.solutions.drawing_utils

# Współrzędne pikseli punktów charakterystycznych
punkty = []

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)
    # czyszczenie listy punktów
    punkty.clear()

    if results.multi_hand_landmarks:
        for handsMulti in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handsMulti, mpHands.HAND_CONNECTIONS)
            a, b = int(handsMulti.landmark[mpHands.HandLandmark.RING_FINGER_TIP].x * img.shape[1]), int(handsMulti.landmark[mpHands.HandLandmark.RING_FINGER_TIP].y * img.shape[0])
            punkty.append((a, b))

    shp = [(350, 1080), (1600, 1080), (1600, 540), (350, 540)]

    #sprawdzanie czy czlowiek jest pod markiza
    reka = False
    for x, y in punkty:
        if sprawdz_wspolrzedne(x, y, shp):
            reka = True
    if reka:
        print("reka w zasięgu okna")

    # rysowanie pola markizy
    shp = np.array(shp, np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [shp], True, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

