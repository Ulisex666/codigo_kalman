import cv2
import numpy as np
from skin_face_detection import get_face_roi

# Open video
video_path = 'video_prueba.mp4'
cap = cv2.VideoCapture(video_path)

width, length = 480, 624
# Valores para la aproximacion de la profundidad. Son estimaciones
# muy burdas
S_real = 0.2  
F = 1000  
kalman = cv2.KalmanFilter(6, 3)  

# Matriz de transicion. Posicion x_k = x +dx, y_k = y + dy, z_k = z + dz
# La velocidad se asume constante 
full_transition_matrix = np.array(
    [[1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0],
    [0, 0, 1., 0, 0, 1],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1.]], dtype=np.float32) 

kalman.transitionMatrix = np.eye(6, dtype=np.float32)

# Matriz de medida. Solo se nos da informacion de la posicion x, y, z
kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0],
                                     [0, 0, 1., 0, 0, 0]], dtype=np.float32)


# Matriz de ruido en el proceso. Se asume poco ruido independiente en
# cada variable
kalman.processNoiseCov = np.eye(6, dtype=np.float32)*0.3
# Matriz de ruido para la medicion
kalman.measurementNoiseCov = np.eye(3, dtype=np.float32)*0.1
# Matriz de covarianza para el error
kalman.errorCovPost = np.eye(6, dtype=np.float32)

# Estado inicial del filtro de Kalman. Se asume que la cara esta en 
# el centro de la imagen
# El centro de la imagen (480, 624) es (312, 240). Por sencillez,
# tomamos profundidad de 1
initial_state = np.array([320, 240, 0.0, 0, 0, 0], dtype=np.float32)  
# Las medida Pre y Post son iguales dado que no se tienen medidas
kalman.statePost = initial_state
kalman.statePre = initial_state  

def estimate_depth(face_width_pixels):
    Z = (F * S_real) / face_width_pixels  
    return Z

while cap.isOpened():
    ret, img = cap.read()  # Read each frame
    
    if not ret:
        break  # Exit if no frames are left
    
    
    # Apply the ROI function
    top_left, bottom_right, mid_point = get_face_roi(img)
    face_width_pixels = bottom_right[0] - top_left[0]
    Z = estimate_depth(face_width_pixels)


    prediccion = kalman.predict()
    x_predict = int(prediccion[0])
    y_predict = int(prediccion[1])
    z_predict = int(prediccion[2])

    measurement = np.array([mid_point[0], mid_point[1], Z], dtype=np.float32)

    kalman.correct(measurement)
    
    
    
    # Draw the bounding box and center
    cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)  # Green box
    cv2.drawMarker(img, mid_point, color=[0,255,0], markerType=cv2.MARKER_CROSS,
                thickness=4, markerSize=50)
    cv2.circle(img, (x_predict, y_predict), 10, (0, 0, 255), 2)  
    
    cv2.putText(img, f"Prediccion (x, y, z): ({x_predict}, {y_predict}, {z_predict})", 
                (mid_point[0] + 10, mid_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0 , 0), 2)

    # Mostrar las coordenadas medidas
    cv2.putText(img, f"Medicion (x, y, z): ({mid_point[0]}, {mid_point[1]}, {Z:.2f})", 
                (mid_point[0] + 10, mid_point[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255  ), 2)
# Red center point
    
    # Display the frame with ROI
    cv2.imshow('Tracking', img)
    cv2.waitKey(1)
    #cv2.destroyWindow('Frame')
    # cv2.destroyWindow(f'Frame {frame_count}')
    # Wait for a key to continue
    if 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
