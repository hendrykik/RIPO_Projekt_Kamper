import cv2
from numpy import *

cap = cv2.VideoCapture('IMG_8463.mov')


while True:
    success, img = cap.read()
    height, width, channels = img.shape
    mask = zeros((height+2, width+2), uint8)

    #maximum distance to start pixel:
    diff = (2, 2, 2)


    door_top_pixel = (600, 0)

    # wykryj wnętrze drzwi kampera
    retval, rect, _, _ = cv2.floodFill(img, mask, door_top_pixel, (0, 255, 0), diff, diff)

    print(retval)
    # porównaj rozmiar obszaru zalania z wymiarami drzwi kampera
    if (retval > 360000):
        print("camper door close")
    else:
        print("camper door open")

    cv2.imshow("Image", img)
    cv2.waitKey(1)
