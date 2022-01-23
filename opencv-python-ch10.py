# 이미지 저장 / 동영상 저장
# imwrite

# 이미지 저장
import cv2
img = cv2.imread("./image/pig.jpg",cv2.IMREAD_GRAYSCALE)   # 흑백으로 이미지 불러오기

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

result = cv2.imwrite('./image/pig_save.jpg',img)
print(result)        # True라고 나오면 잘된거

# 간단하게 이렇게 해도 된다. png 형태로도 저장 가능하다.
# import cv2
# img = cv2.imread("./image/pig.jpg",cv2.IMREAD_GRAYSCALE)   # 흑백으로 이미지 불러오기
# cv2.imwrite('./image/pig_save.jpg',img)

# 동영상 저장
import cv2

cap = cv2.VideoCapture('./image/test_video.mp4')

# 코덱 정의 (어떤 형태로 저장할 지 정의)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # *'DIVX' == 'D', 'I', 'V', 'X' 를 나타냄

# 프레임 크기, fps
width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))      # 원래 동영상의 width를 가져옴 (정수로 가져와야함)
height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))     # 원래 동영상의 height를 가져옴 (정수로 가져와야함)
fps = cap.get(cv2.CAP_PROP_FPS)     # 영상의 속도, fps 정보를 가져옴   
# fps = cap.get(cv2.CAP_PROP_FPS)*2      # 영상 재생 속도 2배

out = cv2.VideoWriter('./image/output.mp4',fourcc, fps, (width, height))
# 저장 파일명, 코덱, fps, 크기(width, height)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:     # 더 이상 읽을게 없으면 break
        break

    out.write(frame)    # 영상 데이터만 저장 (소리 x)

    cv2.imshow('video',frame)
    if cv2.waitKey(10) == ord('q'):
        break

out.release()   # 자원 해제 
cap.release()
cv2.destroyAllWindows


