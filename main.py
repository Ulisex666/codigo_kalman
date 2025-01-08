import cv2 
import numpy as np
from skin_face_detection import get_face_roi

img = cv2.imread('imgs/frame2.jpg')
height, length = img.shape[0], img.shape[1]

top_left, bottom_right, mid_point = get_face_roi(img)

# Creamos la ROI sobre la imagen y marcamos el centro
cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 4)
cv2.drawMarker(img, mid_point, color=[0,255,0], markerType=cv2.MARKER_CROSS,
               thickness=4, markerSize=50)

# Se muestra el resultado de la clasificacion
cv2.imshow('Seleccion de piel del modelo', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
