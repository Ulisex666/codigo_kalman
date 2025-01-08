import cv2
import numpy as np
from skin_face_detection import get_face_roi

# Open video
video_path = 'video_prueba.mp4'
cap = cv2.VideoCapture(video_path)

kalman_filter = cv2.KalmanFilter(4, 2)

# Matriz de transicion. Posicion x_k = x +dx, y_k = y + dy.
# La velocidad se asume constante 
kalman_filter.transitionMatrix = np.array(
    [[1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1]], dtype=np.float32)

# Matriz de medida. Solo se nos da informacion de la posicion x y y
kalman_filter.measurementMatrix = np.array(
    [[1, 0, 0, 0],
    [0, 1, 0, 0]], dtype=np.float32)

kalman_filter.processNoiseCov = np.eye(4, dtype=np.float32)*0.3
# Matriz de ruido para la medicion
kalman_filter.measurementNoiseCov = np.eye(2, dtype=np.float32)*0.1
# Matriz de covarianza para el error
kalman_filter.errorCovPost = np.eye(4, dtype=np.float32)

kalman_filter.statePost = np.array([[312], [240], [0], [0]], dtype=np.float32)


frame_count = 0  # To count frames
while cap.isOpened():
    ret, img = cap.read()  # Read each frame
    
    if not ret:
        break  # Exit if no frames are left
    frame_count += 1
    
    
    # Apply the ROI function
    top_left, bottom_right, mid_point = get_face_roi(img)
    
    prediccion = kalman_filter.predict()
    x_predict = int(prediccion[0])
    y_predict = int(prediccion[1])
    
    kalman_filter.correct(np.array([mid_point[0], mid_point[1]], dtype=np.float32))
    
    # Draw the bounding box and center
    cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)  # Green box
    cv2.drawMarker(img, mid_point, color=[0,255,0], markerType=cv2.MARKER_CROSS,
                thickness=4, markerSize=50)
    cv2.circle(img, (x_predict, y_predict), 10, (0, 0, 255), 2)  
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
