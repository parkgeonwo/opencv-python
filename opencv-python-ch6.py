# chpater 6 : joining images
# hstack / vstack

import cv2
import numpy as np

img = cv2.imread("./image/pig.jpg")

imgHor = np.hstack((img,img)) # img 2개를 가로로 정렬 
imgVer = np.vstack((img,img)) # img 2개를 세로로 정렬

cv2.imshow("Horizontal",imgHor)
cv2.imshow("Vertical",imgVer)

cv2.waitKey(0)


