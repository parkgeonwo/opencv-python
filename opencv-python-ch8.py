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
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)   # 윤곽선과 계층구조를 반환
    # img는 contour찾기를 할 소스 이미지 , 두번째는 contour 추출 모드 --> hierarchy에 영향을 줌
    # 세번째는 contour 근사 방법, none은 contour를 구성하는 모든 점을 저장
    # 검색방법과 근사화 방법은 코드 맨 아래에 추가 정리하였습니다~

    for cnt in contours:
        area = cv2.contourArea(cnt)  # 각 contours이 감싸는 영역의 넓이를 구함
        # print(area)

        if area>500:  # 너무 작으면 윤곽선에 다 지배당함
            cv2.drawContours(imgContour,cnt,-1,(255,0,0),3)     # 윤곽선 그리는 함수, imgContour에서 파란색으로 인덱스 -1에 해당하는 cnt를 3의 두께로 그린다.
            peri = cv2.arcLength(cnt,True)    # 윤곽선 길이를 출력
            # print(peri)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)     
            # 윤곽선을 근사화(단순화)합니다. 인자로 주어진 곡선 또는 다각형을 epsilon 값에 따라 꼭지점 수를 줄여 새로운 곡선이나 다각형을 생성하여 리턴
            # (cnt) numpy array형식의 곡선 또는 다각형 / 근사 정확도(epsilon) : 오리지널, 근사커브간 거리 최대값 / (TF) 폐곡선 or 열린 곡선
            # print(approx)   

            objCor = len(approx)  # contour을 근사화한 점의 갯수를 의미 
            # print(len(approx))

            x,y,w,h = cv2.boundingRect(approx)  # 주어진 점을 감싸는 최소 크기 사각형(바운딩박스)를 반환합니다.
        
            if objCor == 3: objectType = "Tri"      # objCor이 3이면 = 삼각형이면 , objectType에 Tri
            elif objCor == 4:
                aspRatio = w/float(h)
                if aspRatio > 0.95 and aspRatio < 1.05: objectType = "Square"   # aspect 종횡비가 95~105% 사이라면, square
                else: objectType = "Rectangle"  # 직사각형  
            elif objCor > 4: objectType = "Circle"   
            else: objectType = "None"

            cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)     # 사각형 그리기 , xy좌표, 대각선의 xy좌표, 색깔 , technic
            cv2.putText(imgContour, objectType, (x+(w//2)-10,y+(h//2)-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0),2 )
            # img / text / position / font / scale / color / technic

path = "./image/shapes.png"
img = cv2.imread(path)
imgContour = img.copy()

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # img를 bgr에서 gray로 바꿈
imgBlur = cv2.GaussianBlur(imgGray, (7,7),1)   # blur처리 , 7x7 크기 가우시안 커널, 표준편차 1
imgCanny = cv2.Canny(imgBlur, 50,50)

getContours(imgCanny)

imgBlank = np.zeros_like(img)     # 빈 img 만들기

imgStack = stackImages(0.8,[[img,imgGray,imgBlur], # img, imgGray, imgBlur를 연달아서 0.6크기로 행으로 표현
                            [imgCanny,imgContour,imgBlank]])    

# cv2.imshow("Original",img)
# cv2.imshow("Gray", imgGray)
# cv2.imshow("Blur", imgBlur)
cv2.imshow("Stack", imgStack)

cv2.waitKey(0)


# 계층구조
# 계층 구조는 윤곽선을 포함 관계의 여부를 나타냅니다.
# 즉, 외곽 윤곽선, 내곽 윤곽선, 같은 계층 구조를 구별할 수 있습니다.
# 이 정보는 hierarchy에 담겨있습니다.
# [다음 윤곽선, 이전 윤곽선, 내곽 윤곽선, 외곽 윤곽선]에 대한 인덱스 정보를 포함하고 있습니다.

# 검색 방법
# cv2.RETR_EXTERNAL : 외곽 윤곽선만 검출하며, 계층 구조를 구성하지 않습니다.
# cv2.RETR_LIST : 모든 윤곽선을 검출하며, 계층 구조를 구성하지 않습니다.
# cv2.RETR_CCOMP : 모든 윤곽선을 검출하며, 계층 구조는 2단계로 구성합니다.
# cv2.RETR_TREE : 모든 윤곽선을 검출하며, 계층 구조를 모두 형성합니다. (Tree 구조)

# 근사화 방법
# cv2.CHAIN_APPROX_NONE : 윤곽점들의 모든 점을 반환합니다.
# cv2.CHAIN_APPROX_SIMPLE : 윤곽점들 단순화 수평, 수직 및 대각선 요소를 압축하고 끝점만 남겨 둡니다.
# cv2.CHAIN_APPROX_TC89_L1 : 프리먼 체인 코드에서의 윤곽선으로 적용합니다.
# cv2.CHAIN_APPROX_TC89_KCOS : 프리먼 체인 코드에서의 윤곽선으로 적용합니다.






