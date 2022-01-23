# chpater 4 : shpaes and texts
# line / rectangle / circle / putText / polylines / fillPoly

import cv2
import numpy as np

img = np.zeros((512,512,3), np.uint8)        # 세로 512, 가로 512, 3 channel (RGB)에 해당하는 스케치북 만들기 / 검정색 스케치북
# print(img)
# img[:] = (255,255,255)  # 전체 공간을 흰 색으로 채우기

# img[200:300, 100:300] = (255,0,0)          # height, width, blue  / 세로영역 200~300 , 가로 영역 100~300, 파란색으로 만들기

# 선그리기
# cv2.line(img,(0,0),(300,300),(0,255,0),3, cv2.LINE_8)  # (선긋기) img, starting points , ending points, define color, thickness ,  define technic
                                           # green line 0,0 부터 300,300까지

cv2.line(img,(0,0),(img.shape[1],img.shape[0]),(0,255,0),3)  # green line 0,0부터 x,y좌표 끝까지

# 사각형 그리기
# cv2.rectangle(img,(0,0),(250,350),(0,0,255),2) # (사각형) img, start, end, color, technic
cv2.rectangle(img,(0,0),(250,350),(0,0,255),cv2.FILLED) 
# img, start(왼쪽 위), end(오른쪽 아래), color, thickness(cv2.FILLED 가능)

# 원그리기
cv2.circle(img,(400,50),30,(255,255,0),5, cv2.LINE_AA)       
# (원) img, center point, radius, color, thickness(cv2.FILLED 사용 가능) , line style

# 다각형 그리기
pts1 = np.array([[100,100],[200,100],[100,200]])   # 대괄호 2개
pts2 = np.array([[200,100],[300,100],[300,200]])

# cv2.polylines(img, [pts1], True, (0,255,0),3,cv2.LINE_AA)
# cv2.polylines(img, [pts2], True, (0,255,0),3,cv2.LINE_AA)
cv2.polylines(img, [pts1,pts2], True, (0,255,0),3,cv2.LINE_AA) # 위의 두줄을 한번에 그리는 코드
# 그릴 위치, 그릴 좌표들, 닫힘 여부, 색깔, 두께, 선 종류

pts3 = np.array([ [[100,300],[200,300],[100,400]], [[200,300],[300,300],[300,400]] ])      # 대괄호 3개
cv2.fillPoly(img, pts3, (0,255,0), cv2.LINE_AA)    # 꽉찬 다각형 / true, thickness 안넣음
# 그릴 위치, 그릴 좌표들, 색깔, 선 종류

# 텍스트 삽입
cv2.putText(img," OPENCV  ",(300,200), cv2.FONT_HERSHEY_COMPLEX,1, (0,150,0),1)   # (텍스트 삽입) img, start point, font, scale, color, technic

# 한글 우회 방법

# PIL(python image library)
from PIL import ImageFont, ImageDraw, Image

def myPutText(src, text, position, font_size, font_color): # 이미지를 받아서 text를 넣고 또 다른 이미지로 return
    img_pil = Image.fromarray(src)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype('fonts/gulim.ttc',font_size)
    draw.text(position, text, font=font, fill = font_color)
    return np.array(img_pil)

img = myPutText(img, "응애코린이", (20,50), 30, (0,255,0) )
# 그릴 위치, 텍스트 내용, 시작 위치, 폰트 사이즈, 색깔

cv2.imshow("Image",img)
cv2.waitKey(0)


# 직선의 종류

# cv2.LINE_4 : 상하좌우 4 방향으로 연결된 선  -- > 한점을 찍고 그다음 점이 상하좌우 4방향으로만 찍기 가능
# cv2.LINE_8 : 대각선을 포함한 8방향으로 연결된 선 (기본값)  --> 상하좌우대각선까지 8방향으로 찍기 가능
# cv2.LINE_AA : 부드러운 선 (anti-aliasing)

# 글꼴 종류

# cv2.FONT_HERSHEY_SIMPLEX  : 보통 크기의 산 세리프 글꼴
# cv2.FONT_HERSHEY_PLAIN : 작은 크기의 산 세리프 글꼴 
# cv2.FONT_HERSHEY_SCRIPT_SIMPLEX : 필기체 스타일 글꼴
# cv2.FONT_HERSHEY_TRIPLEX : 보통 크기의 세리프 글꼴
# cv2.FONT_ITALIC : 기울임 (이탤릭체)        ---> "|" 로 폰트칸에 함께 써주면됨





