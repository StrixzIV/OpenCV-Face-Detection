import cv2

print('Press \'q\' to stop')

#set input as webcam 
cap = cv2.VideoCapture(0)

#select haarcascade datasets
face_cascade = faceCascade = cv2.CascadeClassifier('datasets/haarcascade_frontalface_default.xml')
glasses_cascade = cv2.CascadeClassifier('datasets/haarcascade_eye_tree_eyeglasses.xml')

#set output resolutions
cap.set(3, 1280)
cap.set(4, 720)

while True:
    
    #read the input image
    _, img = cap.read()
    
    #change image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #detect the face
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5, minSize = (20, 20))
    glasses = glasses_cascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5, minSize = (20, 20))

    #Track face positons
    for (x, y, w, h) in faces:
        #Draw rectangle on faces
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 3)
        
        #Set region of interest
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        
    #Track eyes postions
    for (x, y, w, h) in glasses:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

    #Display the Image
    cv2.imshow('Image', img)
    
    #Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
