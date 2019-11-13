import cv2
import os

try:
    cam=cv2.videoCapture(0)
    detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    Id=input('Enter your name with your Enrollment number:')
    path= "C:\Users\wadha\Desktop\Attendance System\Datasets\" + Id 
    new_path=("C:\Users\wadha\Desktop\Attendance System\Datasets\" + Id + '\'
    os.mkdir(path)
    sampleNum=0
    print('We are getting your face images...')

    while True:
        ret,image=cam.read()
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=detector.detectMultiSCale(gray,1.3)

        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
            sampleNum=sampleNum+1
            cv2.imwrite(new_path +str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
            cv2.imshow('result',img)

        if cv2.waitkey(100) & 0xFF == ord('q'):
            break
        elif sampleNum>100:
            print("Thanks!! You have registered")
            break

    cam.release()
    cv2.destroyAllWindows()
    
except FileExistsError as f:
    print(f)
            
    
