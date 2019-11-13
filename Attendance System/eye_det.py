import cv2
import numpy as np

eyes_dataset=cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

capture=cv2.VideoCapture(0)

while True:
    ret,img=capture.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    eyes=eyes_dataset.detectMultiScale(gray,1.3)

    for x,y,w,h in eyes:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
        my_eyes=gray[y:y+h,x:x+w]
        
    
        
    if cv2.waitKey(1) & 0xFF==27:
        break
    
    cv2.imshow('result',img)
    
capture.release()
cv2.destroyAllWindows()
