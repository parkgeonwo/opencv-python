# chpater 4 : shpaes and texts

import cv2
import numpy as np

img = np.zeros((512,512,3), np.uint8)
# print(img)

# img[200:300, 100:300] = 255,0,0          # height, width, blue

# cv2.line(img,(0,0),(300,300),(0,255,0),3)  # starting points , ending points, define color, define technic
                                           # green line 0,0 부터 300,300까지

cv2.line(img,(0,0),(img.shape[1],img.shape[0]),(0,255,0),3)  # green line 0,0부터 x,y좌표 끝까지

# cv2.rectangle(img,(0,0),(250,350),(0,0,255),2) # start, end, color, technic
cv2.rectangle(img,(0,0),(250,350),(0,0,255),cv2.FILLED) # start, end, color, technic = 사각형 꽉채우기

cv2.circle(img,(400,50),30,(255,255,0),5)       # center point, radius, color, technic

cv2.putText(img," OPENCV  ",(300,200), cv2.FONT_HERSHEY_COMPLEX,1, (0,150,0),1)   # start point, font, scale, color, technic

cv2.imshow("Image",img)
cv2.waitKey(0)

