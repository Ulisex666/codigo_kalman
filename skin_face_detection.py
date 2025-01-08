import numpy as np
from scipy.stats import multivariate_normal

# Medias y covarianza obtenidas de tareas anteriores
skin_mean = np.array([109.04378154, 130.19204512, 172.47343425])
skin_cov = np.array([[1557.55025745, 1630.09297116, 1814.77250065],
       [1630.09297116, 1731.13069521, 1941.66919639],
       [1814.77250065, 1941.66919639, 2250.5592615 ]])

fondo_mean = np.array([ 95.55063731, 106.95504056, 110.26025492])
fondo_cov = np.array([[4486.90316045, 4649.32282388, 4487.6689383 ],
       [4649.32282388, 5074.34657037, 4956.60725216],
       [4487.6689383 , 4956.60725216, 5081.15951645]])

# Se crea el modelo para el analisis de la imagen
piel_norm = multivariate_normal(skin_mean, skin_cov).pdf
fondo_norm = multivariate_normal(fondo_mean, fondo_cov).pdf

# Probabilidades obtenidas de tareas anteriores
priori_piel = 0.65
priori_fondo = 1 - priori_piel

def get_face_roi(img):
    '''Funcion que recibe una imagen en formato array de numpy,
    detecta la piel en la imagen y devuelve coordenadas indicando el
    contorno y la posicion media de la piel detectada'''

    # Calculamos la probabilidad de ser fondo o piel para cada pixel
    # en la imagen
    img_piel_prob = piel_norm(img) * priori_piel
    img_fondo_prob = fondo_norm(img) * priori_fondo
    
    # Verificamos que probabilidad es mayor para cada pixel
    skin_pixels = np.asarray(img_piel_prob > img_fondo_prob).nonzero()
    
    # Obtenemos las coordenadas minimas y maximas para los pixeles 
    # detectados como piel
    min_row, max_row = min(skin_pixels[0]), max(skin_pixels[0])
    min_col, max_col = min(skin_pixels[1]), max(skin_pixels[1])
    
    # Se obtiene el centroide de estos pixeles
    mid_row = (min_row + max_row) // 2
    mid_col = (min_col + max_col) // 2
    
    # Empaquetamos los datos para regresarlos
    top_left = (min_row, min_col)
    bottom_right = (max_row, max_col)
    
    mid_point = (mid_row, mid_col)
    
    return top_left, bottom_right, mid_point