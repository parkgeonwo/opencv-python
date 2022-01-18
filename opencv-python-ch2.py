# ch2 : basic functions

import cv2
import numpy as np

img = cv2.imread("./image/pig.jpg")
kernal = np.ones((5,5), np.uint8)

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        # convert color / img를 bgr에서 gray로
imgBlur = cv2.GaussianBlur(imgGray, (7,7),0)        # 7x7 사이즈 = 가우시안 커널 크기 / blur 처리 / 표준편차 0   # blur 처리하는 이유 아래 설명
imgCanny = cv2.Canny(img,100,100)              # canny edge detection / 하단,상단 임계값(threshold)   # 아래 설명
imgDialation = cv2.dilate(imgCanny,kernal,iterations=1 )        # iterations 값이 크면 더굵게 윤곽선땀 , 필터 내부의 가장 높은(밝은) 값으로 변환(or) - 팽창연산
imgEroded = cv2.erode(imgDialation, kernal, iterations=1 )      # 필터 내부의 가장 낮은(어두운) 값으로 변환(and) - 침식연산


cv2.imshow("Gray image", imgGray)      # bgr2gray 사진
cv2.imshow("Blur image", imgBlur)      # blur 처리한 사진
cv2.imshow("Canny image", imgCanny)      # Canny 처리한 사진
cv2.imshow("Dialation Image", imgDialation)     # dialation 처리한 사진
cv2.imshow("Eroded Image", imgEroded)     # erode 처리한 사진
cv2.waitKey(0)


# blur 처리를 왜할까

# 이미지 블러링은 이미지를 로우 패스 필터 커널로 컨볼루션하는 것입니다.
# 이미지에서 고주파인 노이즈가 흐려지게 됩니다. 이때 같은 고주파인 선도 같이 흐려지게 됩니다. (결국 노이즈를 제거하는데 유용한 방법이란 뜻)
# 이미지를 좀 더 매끈하게 보이도록 만드는 효과 / 4가지 방법의 blur처리가 있다.


# canny 알고리즘

# 경계썬 검출 방식에서 가장 많이 사용하는 알고리즘이다. 일반적으로 경계선 검출기는 잡음에 매우 민감한 특성을 가지고 있다.
# 따라서 잡음으로 인해 잘못된 경계선을 계산하는 것을 방지하기 위해 개발 된 알고리즘이다. 다음 5가지 단계를 거친다.
# 1. Gaussian Filter로 이미지의 잡을 제거
# 2. Sobal Filter를 사용해 Gradient의 크기(intensity)를 구한다.
# 3. Non-maximum suppression을 적용해 경계선 검출기에서 거짓 반응 제거
# 4. 경계선으로써 가능성 있는 픽셀을 골라내기 위해 double threshold 방식 적용
# 5. 앞서 double threshold 방식에서 maxVal을 넘은 부분을 strong edge, minVal과 maxVal 사이의 부분을 weak edge로 설정하여,
#    strong edge 와 연결되어 있는 weak edge를 edge로 판단하고 그렇지 않는 부분은 제거 (Hysteresis thresholding)







