import cv2

image = cv2.imread('2.jpg')

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = classifier.detectMultiScale(grayscale_image, scaleFactor=1.04, minNeighbors=5)
print(faces)

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

face_count = len(faces)

cv2.putText(image, f'Total Faces: {face_count}', (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

for i, (x, y, w, h) in enumerate(faces):
    cv2.putText(image, f'Face {i+1}', (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

cv2.imwrite('output_faces.jpg', image)

cv2.imshow('Image', image)
cv2.waitKey(0)
