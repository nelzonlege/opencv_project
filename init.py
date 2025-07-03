import cv2

capture = cv2.VideoCapture(0)
face_hear_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  

while True:
    ret, image = capture.read()  
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_hear_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)

    cv2.imshow("Video", image)
    k = cv2.waitKey(30) & 0xFF

    if k == 27:  
        break

capture.release()
cv2.destroyAllWindows()
