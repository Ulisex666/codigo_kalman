import cv2
import numpy as np
from skin_face_detection import get_face_roi

# Ubicacion de cada uno de los frames guardados
imgs_paths = ['imgs/frame1.jpg', 'imgs/frame2.jpg', 'imgs/frame3.jpg', 'imgs/frame4.jpg', 
              'imgs/frame5.jpg', 'imgs/frame6.jpg', 'imgs/frame7.jpg', 'imgs/frame8.jpg',
              'imgs/frame9.jpg', 'imgs/frame10.jpg', 'imgs/frame11.jpg', 'imgs/frame12.jpg', 
              'imgs/frame13.jpg', 'imgs/frame14.jpg']

# Establemecos las dimesniones de las imagenes
width, length = 480, 624
# Valores para la aproximacion de la profundidad. Son estimaciones
# muy burdas
S_real = 0.2  
F = 1000  

# Inicializacion del filtro de Kalman. parte de OpenCV
# Buscamos predecir la posicion del centroide en el frame k.
# Se inicializa con 6 variables de estado y 3 de medicion.
# Variables de estado:
# Posicion X, posicion Y, posicion Z, velocidad X, velocidad Y, velocidad Z
# Variables de medicion:
# Posicion X, posicion y, posicion Z

kalman = cv2.KalmanFilter(6, 3)  

# Matriz de transicion. Posicion x_k = x +dx, y_k = y + dy, z_k = z + dz
# La velocidad se asume constante 
kalman.transitionMatrix = np.array(
    [[1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 1],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1]], dtype=np.float32)  

# Matriz de medida. Solo se nos da informacion de la posicion x, y, z
kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0]], dtype=np.float32)


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
initial_state = np.array([320, 240, 1.0, 0, 0, 0], dtype=np.float32)  
# Las medida Pre y Post son iguales dado que no se tienen medidas
kalman.statePost = initial_state
kalman.statePre = initial_state  

# Funcion para estimar el valor de z de acuerdo a los pixeles de cara
def estimate_depth(face_width_pixels):
    Z = (F * S_real) / face_width_pixels  
    return Z

for i, frame in enumerate(imgs_paths):
    # Se elige la imagen y se aplica la funcion para detectar el rostro
    img = cv2.imread(frame)
    top_left, bottom_right, mid_point = get_face_roi(img)

    # Calculamos la longitud del rsotro detectado
    face_width_pixels = bottom_right[0] - top_left[0]

    # Estimacion de la coordenada z
    Z = estimate_depth(face_width_pixels)

    # Se predice el estado siguiente con el filtro de Kalman
    prediccion = kalman.predict()
    x_predict = int(prediccion[0])
    y_predict = int(prediccion[1])
    z_predict = int(prediccion[2])

    # Se corrige la prediccion con la medicion obtenida
    measurement = np.array([mid_point[0], mid_point[1], Z], dtype=np.float32)
    kalman.correct(measurement)

    # Se dibuja el rostro detectado en la imagen
    cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 4)
    cv2.drawMarker(img, mid_point, color=[0, 255, 0], markerType=cv2.MARKER_CROSS, thickness=4, markerSize=20)

    # Mostrar coordenadas predecidas en la imagen y el marcador
    cv2.drawMarker(img, (x_predict, y_predict), color=[0, 0, 255], markerType=cv2.MARKER_STAR, thickness=4, markerSize=20)
    cv2.putText(img, f"Prediccion (x, y, z): ({x_predict}, {y_predict}, {z_predict})", 
                (mid_point[0] + 10, mid_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0 , 0), 2)

    # Mostrar las coordenadas medidas
    cv2.putText(img, f"Medicion (x, y, z): ({mid_point[0]}, {mid_point[1]}, {Z:.2f})", 
                (mid_point[0] + 10, mid_point[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255  ), 2)

    # Guardar los resultados
    output_path = f'imgs/resultados/frame{i}_result.jpg'  # Path to save the image
    #cv2.imwrite(output_path, img)  # Save the image

    cv2.imshow(f'Filtro de Kalman frame {i}',img)
    cv2.waitKey(1000)


cv2.destroyAllWindows()
