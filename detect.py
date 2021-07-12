import cv2

#set input as webcam 
cap = cv2.VideoCapture(0)

#select haarcascade datasets
face_cascade = faceCascade = cv2.CascadeClassifier('datasets/haarcascade_frontalface_default.xml')

#set output resolutions
cap.set(3, 1280)
cap.set(4, 720)

while cap.isOpened():
    
    #read the input image
    _, img = cap.read()
    
    #change image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #detect the face
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    #Draw rectangle on face
    for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 3)

    #Exit key
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
