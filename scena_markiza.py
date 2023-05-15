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


cap = cv2.VideoCapture("IMG_8460.mov")

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# Współrzędne pikseli punktów charakterystycznych
punkty = []

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    # czyszczenie listy punktów
    punkty.clear()

    # dodawanie punktów do listy
    if results.pose_landmarks is not None:
        for landmark in results.pose_landmarks.landmark:
            a, b = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
            punkty.append((a, b))

    #rysowanie lini człowieka
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    shp = [(460, 1080), (1920, 1080), (1920, 820), (1400, 620)]

    #sprawdzanie czy czlowiek jest pod markiza
    czlowiek = False
    for x, y in punkty:
        if sprawdz_wspolrzedne(x, y, shp):
            czlowiek = True
    if czlowiek:
        print("czlowiek pod markiza")

    #rysowanie pola markizy
    shp = np.array(shp, np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [shp], True, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
