# ch2 : basic functions

import cv2
import numpy as np

img = cv2.imread("./image/pig.jpg")
kernal = np.ones((5,5), np.uint8)

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        # convert color / img를 bgr에서 gray로
imgBlur = cv2.GaussianBlur(imgGray, (7,7),0)        # 7x7 사이즈 = 가우시안 커널 크기 / blur 처리 / 표준편차 0
imgCanny = cv2.Canny(img,100,100)              # 윤곽선만 따는 함수 / 하단,상단 임계값
imgDialation = cv2.dilate(imgCanny,kernal,iterations=1 )        # iterations 값이 크면 더굵게 윤곽선땀
imgEroded = cv2.erode(imgDialation, kernal, iterations=1 )


cv2.imshow("Gray image", imgGray)      # bgr2gray 사진
cv2.imshow("Blur image", imgBlur)      # blur 처리한 사진
cv2.imshow("Canny image", imgCanny)      # Canny 처리한 사진
cv2.imshow("Dialation Image", imgDialation)     # dialation 처리한 사진
cv2.imshow("Eroded Image", imgEroded)     # erode 처리한 사진
cv2.waitKey(0)










