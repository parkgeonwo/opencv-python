# chapter 1 : read images - videos - webcam


import cv2
print("package imported")

img = cv2.imread("./image/pig.jpg")      # imread 함수 = 이미지를 불러오는 함수

cv2.imshow("output", img)      # imshow(윈도우에서 출력이름, 이미지or비디오)
cv2.waitKey(0)          # 몇초동안 실행할것이냐 / 단위는 ms / 키 입력을 기다리는 대기 함수 / 0으로하면 무한

#%%

import cv2

cap = cv2.VideoCapture("./image/test_video.mp4")    # 비디오 불러오는 함수 / 동영상 캡쳐 객체 생성 / 0으로하면 캠 / 1,2,3,, 카메라 인덱스

# cap.set(3,640)    # 가로 사이즈 640
# cap.set(4,480)    # 세로 사이즈 480
# cap.set(10,100)   # brightness 100으로 설정

while True:
    success, img = cap.read()        # 재생되는 비디오의 한 프레임씩 읽습니다. 제대로 읽으면 success에 True 리턴 / 읽은 프레임은 frame에 리턴
    cv2.imshow("video",img)          # 화면에 표시
    if cv2.waitKey(10) & 0xFF == ord('q'):    # 프레임 넘어가는 속도 10ms / q를 누르면 꺼지도록 설정
        break

#%%

#%%
# webcam

import cv2

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)    # 비디오 불러오는 함수 / 동영상 캡쳐 객체 생성 / 0으로하면 캠 / 1,2,3,, 카메라 인덱스

cap.set(3,frameWidth)    # 가로 사이즈 640
cap.set(4,frameHeight)    # 세로 사이즈 480
cap.set(10,130)   # brightness 130으로 설정

while True:
    success, img = cap.read()        # 재생되는 비디오의 한 프레임씩 읽습니다. 제대로 읽으면 success에 True 리턴 / 읽은 프레임은 frame에 리턴
    cv2.imshow("Result",img)          # 화면에 표시
    if cv2.waitKey(10) & 0xFF == ord('q'):    # 프레임 넘어가는 속도 10ms / q를 누르면 꺼지도록 설정
        break


