import cv2

img_path = 'imgs/bowie.jpeg'
img = cv2.imread(img_path)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

face = face_classifier.detectMultiScale(gray_img, scaleFactor=1.1,
                                        minNeighbors=5, minSize=(10,10))

for (x, y, h, w) in face:
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 4)
    
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow('Cara detectada', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
