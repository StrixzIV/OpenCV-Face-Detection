import cv2

cap = cv2.VideoCapture(0)
face_cascade = faceCascade = cv2.CascadeClassifier('Machine Learning/OpenCV/OpenCV Face recognition/haarcascade_frontalface_default.xml')

cap.set(3, 1280)
cap.set(4, 720)

while cap.isOpened():
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 3)

    # Display the Image
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()