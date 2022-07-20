import cv2
from cv2 import VideoCapture

face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
video = VideoCapture(0)
while(True):
    check ,frame = video.read()
    img = frame

    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray_img,scaleFactor=1.05,minNeighbors=5)

    for x,y,w,h in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
    cv2.imshow("Detecting...",img)






    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cv2.destroyAllWindows()

    

