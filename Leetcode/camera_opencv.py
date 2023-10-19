import cv2
from cvzone.HandTrackingModule import HandDetector
import math

cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # 帧的宽度和高度
cap.set(4, 720)

def face(img):
    cv2.circle(img, (530, 230), 100, (0, 0, 0), 3)  # 脸
    cv2.circle(img, (530, 230), 100, (0, 255, 255), -1)  # 填充黄脸
    cv2.circle(img, (530-40, 230-40), 30, (0, 0, 0), 1)  # 左眼眭
    cv2.circle(img, (530+40, 230-40), 30, (0, 0, 0), 1)  # 右眼眶
    cv2.ellipse(img, (530, 230+30), (50, 30), 0, 0, 180, (0, 0, 0), 2)  # 霄巴
    cv2.circle(img, (530, 240), 10, (0, 0, 0), 1)  # 鼻子

def eyes(img, x=0, y=0):
    cv2.circle(img, (490+x, 190+y), 20, (0, 0, 0), -1)  # 左眼球
    cv2.circle(img, (570+x, 190+y), 20, (0, 0, 0), -1)  # 右眼球
    cv2.circle(img, (500+x, 190+y), 5, (255, 255, 255), -1)  # 左眼白
    cv2.circle(img, (560+x, 190+y), 5, (255, 255, 255), -1)  # 右眼白

def flow_eyes(eyesx, eyesy, fingerx, fingery, r=10):
    ang = math.atan2(fingery-eyesy, fingerx-eyesx)
    xnew = int(r * math.cos(ang))
    ynew = int(r * math.sin(ang))
    return xnew, ynew

detector = HandDetector(detectionCon=0.8)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    face(img)
    hands, img = detector.findHands(img)
    
    if hands:
        hand1 = hands[0]
        lmList = hand1["lmList"]
        x1, y1 = lmList[8][0:2]
        cv2.circle(img, (x1, y1), 10, (0, 0, 255), -1)
        r_distance = flow_eyes(490, 190, x1, y1)
        eyes(img, r_distance[0], r_distance[1])
    else:
        eyes(img)
    
    cv2.imshow("img", img)
    k = cv2.waitKey(1)
    
    if k == 27:  # 按下 ESC 退出
        cap.release()  # 关闭摄像头
        break

    try:
        if cv2.getWindowProperty('img', cv2.WND_PROP_AUTOSIZE) < 1:
            break
    except:
        pass

cap.release()  # 释放资源
cv2.destroyAllWindows()
