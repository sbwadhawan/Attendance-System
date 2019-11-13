from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import cv2

faceData=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

files=['abhishek-7016.npy','abhishek-7016.npy']
face_1=np.load(files[0]).reshape(100,-1)
face_2=np.load(files[1]).reshape(100,-1)

usernames=dict()
for i in range(len(files)):
    usernames[i]=files[i][:-4]
print(usernames)

faces=np.concatenate([face_1,face_2])
labels=np.zeros((faces.shape[0],1))
labels[:100,:]=0.0
labels[100:,:]=1.0
print(faces.shape)

x_train,x_test,y_train,y_test=train_test_split(faces,labels.flatten(),test_size=0.20)

svm=LinearSVC()
svm.fit(x_train,y_train)
y_pred=svm.predict(x_test)

acc=accuracy_score(y_pred,y_test)
print(acc*100)
