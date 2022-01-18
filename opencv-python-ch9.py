# chapter 9 : face detection

import cv2
from cv2 import CascadeClassifier

faceCascade = CascadeClassifier("./image/haarcascade_frontalface_default.xml")    # haar-cascade 학습데이터를 불러옴(사람정면얼굴) / 설명은 아래

img = cv2.imread("./image/lena.png")
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)  # cascadeclassifier의 detectmultiscale 함수에 grayscale 이미지를 입력하여 얼굴 검출
# 얼굴이 검출되면 위치(x,y,w,h)를 리스트로 리턴합니다.
# 1.1은 ScaleFactor  / 4는 minNeighbor를 의미하는 값 (설명은 아래)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)     # img / initial point / final point / color / technic

cv2.imshow("Result",img)
cv2.waitKey(0)


# opencv는 haar-cascade 트레이너와 검출기를 모두 제공한다.
# 예를 들어 haar-cascade 트레이너를 이용하여 자동차에 대한 이미지들을 트레이닝 시킬수 있고, 트레이닝 시킨 학습 데이터를 파일로 저장하여,
# 검출기를 이용하여 특정 이미지에서 자동차를 검출할 수 있다는 얘기다.

# scaleFactor : Parameter specifying how much the image size is reduced at each image scale.
# minNeighbors : Parameter specifying how many neighbors each candidate rectangle should have to retain it.

