# chpater 8 : contours / shape detection  (윤곽, 모양 디텍션)
# contour란 같은 값을 가진 곳을 연결한 선이라고 생각하면 된다.
# 예를들면, 같은 높이를 이은 등고선 / 등압선 , 색상에서는 색상강도가 같은 경계를 뜻함

import cv2
from matplotlib.pyplot import contour
import numpy as np

def stackImages(scale,imgArray):         # cv2.imshow한것을 한장에 배열해주는 함수
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def getContours(img):    # contours를 얻기위한 함수 생성
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    # img는 contour찾기를 할 소스 이미지 , 두번째는 contour 추출 모드 --> hierarchy에 영향을 줌, LIST는 계층구조 상관관계 고려X 추출
    # 세번째는 contour 근사 방법, none은 contour를 구성하는 모든 점을 저장 / simple은 수평, 수직, 대각선 방향의 점은 모두 버리고 끝점만 남김
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)


path = "./image/shapes.png"
img = cv2.imread(path)

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # img를 bgr에서 gray로 바꿈
imgBlur = cv2.GaussianBlur(imgGray, (7,7),1)   # blur처리 , 7x7 크기 가우시안 커널, 표준편차 1
imgCanny = cv2.Canny(imgBlur, 50,50)

getContours(imgCanny)

imgBlank = np.zeros_like(img)     # 빈 img 만들기

imgStack = stackImages(0.6,[[img,imgGray,imgBlur], # img, imgGray, imgBlur를 연달아서 0.6크기로 행으로 표현
                            [imgCanny,imgBlank,imgBlank]])    

# cv2.imshow("Original",img)
# cv2.imshow("Gray", imgGray)
# cv2.imshow("Blur", imgBlur)
cv2.imshow("Stack", imgStack)

cv2.waitKey(0)










