# project 1 - virtual_paint

# chapter 1의 webcam copy

import cv2
import numpy as np

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)    # 비디오 불러오는 함수 / 동영상 캡쳐 객체 생성 / 0으로하면 캠 / 1,2,3,, 카메라 인덱스

cap.set(3,frameWidth)    # 가로 사이즈 640
cap.set(4,frameHeight)    # 세로 사이즈 480
cap.set(10,150)   # brightness 130으로 설정

myColors = [[-10,30,30,10,255,255],    # red color hsv
[230,30,30,250,255,255],       # blue color hsv
[50,30,30,70,255,255]]        # yellow color hsv

myColorValues = [[0,0,204],           # BGR / red
[255,0,51],      # blue
[0,255,255]]         # yellow

myPoints = []          # [x, y, colorId]


def findColor(img, myColors,myColorValues):     # color 찾는 함수
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)       # ch 7 참고 / bgr -> hsv로
    count = 0
    newPoints = []

    for color in myColors: 
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHSV, lower, upper)   # img, lower limit, upper limit / 특정 범위 안에 있는 행렬 원소 검출
        x,y = getContours(mask)    # getCountours의 return 값을 x,y에 담는다.
        cv2.circle(imgResult,(x,y),10,myColorValues[count],cv2.FILLED)     # countours 의 특정지점에 circle 그림

        if x!=0 and y!=0:
            newPoints.append([x,y,count])
        count += 1
        # cv2.imshow(str(color[0]),mask)
    return newPoints

def getContours(img):    # ch8. 참고 # contours를 얻고 점을 얻기 위한 함수 생성
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)   # 윤곽선과 계층구조를 반환
    # img는 contour찾기를 할 소스 이미지 , 두번째는 contour 추출 모드 --> hierarchy에 영향을 줌
    # 세번째는 contour 근사 방법, none은 contour를 구성하는 모든 점을 저장
    # 검색방법과 근사화 방법은 코드 맨 아래에 추가 정리하였습니다~

    x,y,w,h = 0,0,0,0

    for cnt in contours:    # 윤곽선 따기 위한 함수 생성 # ch 8 참고
        area = cv2.contourArea(cnt)  # 각 contours이 감싸는 영역의 넓이를 구함
        # print(area)

        if area>500:  # 너무 작으면 윤곽선에 다 지배당함
            # cv2.drawContours(imgResult,cnt,-1,(255,0,0),3)     # 윤곽선 그리는 함수, imgContour에서 파란색으로 인덱스 -1에 해당하는 cnt를 3의 두께로 그린다.
            peri = cv2.arcLength(cnt,True)    # 윤곽선 길이를 출력
            # print(peri)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)     
            # 윤곽선을 근사화(단순화)합니다. 인자로 주어진 곡선 또는 다각형을 epsilon 값에 따라 꼭지점 수를 줄여 새로운 곡선이나 다각형을 생성하여 리턴
            # (cnt) numpy array형식의 곡선 또는 다각형 / 근사 정확도(epsilon) : 오리지널, 근사커브간 거리 최대값 / (TF) 폐곡선 or 열린 곡선
            x,y,w,h = cv2.boundingRect(approx)  # 주어진 점을 감싸는 최소 크기 사각형(바운딩박스)를 반환합니다.
    return x+w//2,y

def drawOnCanvas(myPoints, myColorValues):
    for point in myPoints:
        cv2.circle(imgResult,(point[0],point[1]),10,myColorValues[point[2]],cv2.FILLED)



while True:
    success, img = cap.read()        # 재생되는 비디오의 한 프레임씩 읽습니다. 제대로 읽으면 success에 True 리턴 / 읽은 프레임은 frame에 리턴
    imgResult = img.copy()           # img 복사해서 윤곽선따기 위한 복사본
    
    newPoints = findColor(img, myColors,myColorValues)         # color 찾기
    if len(newPoints) != 0:
        for newP in newPoints:
            myPoints.append(newP)
    if len(myPoints) != 0:
        drawOnCanvas(myPoints,myColorValues)

    cv2.imshow("Result",imgResult)          # 화면에 표시
    if cv2.waitKey(1) & 0xFF == ord('q'):    # 프레임 넘어가는 속도 1ms / q를 누르면 꺼지도록 설정
        break




