# chapter 5 : warp perspective
# getPerspectiveTransform / warpPerspective 

import cv2
import numpy as np

img = cv2.imread("./image/pig.jpg")

width,height = 250, 350
pts1 = np.float32([[111,219],[287,188],[154,482],[352,440]])    # 점 4개의 변환 전(이동 전) 좌표
pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])  # 점 4개의 변환 후(이동 후) 좌표
matrix = cv2.getPerspectiveTransform(pts1,pts2)    # 변환 행렬 계산 / 변환 전후 매핑 좌표를 지정해주면 원근 변환에 필요한 3x3 변환행렬을 계산해주는 함수
imgOutput = cv2.warpPerspective(img,matrix,(width,height))   # 원근 변환 적용 / 보는 사람의 시각에 따라 같은 물체도 먼 것은 작게, 가까운 것은 크게 보이는 형상인 원근감을 주는 변환

cv2.imshow("image",img)
cv2.imshow("Output",imgOutput)

cv2.waitKey(0)
