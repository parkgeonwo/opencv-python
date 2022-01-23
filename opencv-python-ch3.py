# chapter 3 : resizing and cropping
# resize / crop

import cv2
import numpy as np

img = cv2.imread("./image/pig.jpg")
print(img.shape)      # img의 shape 확인    / 517,720,3

imgResize = cv2.resize(img,(300,200))   # resize
print(img.shape)   # 200,300,3

imgCropped = img[0:200, 200:500]      # 잘라내기 , height, width

cv2.imshow("Image",img)
cv2.imshow("Image Resize",imgResize)
cv2.imshow("Image Cropped",imgCropped)
cv2.waitKey(0)
