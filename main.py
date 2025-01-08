import cv2 
import numpy as np
from skin_face_detection import get_face_roi

imgs_paths = ['imgs/frame1.jpg', 'imgs/frame2.jpg', 'imgs/frame3.jpg', 'imgs/frame4.jpg', 
              'imgs/frame5.jpg', 'imgs/frame6.jpg', 'imgs/frame7.jpg', 'imgs/frame8.jpg',
              'imgs/frame9.jpg', 'imgs/frame10.jpg', 'imgs/frame11.jpg', 'imgs/frame12.jpg', 
              'imgs/frame13.jpg', 'imgs/frame14.jpg']

img = imgs_paths[0]
top_left, bottom_right, mid_point = get_face_roi(img)

# Creamos la ROI sobre la imagen y marcamos el centro
cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 4)
cv2.drawMarker(img, mid_point, color=[0,255,0], markerType=cv2.MARKER_CROSS,
               thickness=4, markerSize=50)

# Se muestra el resultado de la clasificacion
cv2.imshow('Seleccion de piel del modelo', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
