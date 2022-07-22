import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import streamlit as st


import streamlit as st

st.title("OpenCV Demo App")
st.subheader("This app allows you to play with Image filters!")
st.text("We use OpenCV and Streamlit for this demo")
if st.checkbox("Main Checkbox"):
    st.text("Check Box Active")

slider_value = st.slider("Slider", min_value=0.5, max_value=3.5)
st.text(f"Slider value is {slider_value}")

st.sidebar.text("text on side panel")
st.sidebar.checkbox("Side Panel Checkbox")

streamlit run sign.py









# detector를 가지고 손을 인식한다

try:
    cap = cv2.VideoCapture(0)
except:
    cap = cv2.VideoCapture(1) # VideoCapture가 0번으로 안불러와 지는 경우를 대비해 1번으로 가져옴
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # 창 조절하기.
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
# hand detection
detector = HandDetector(maxHands=1) #손 이미지를 마디마다 연결해서 분석해준다.
classifier = Classifier("./keras_model.h5", "./labels.txt") 
 
offset = 20
imgSize = 300
 
labels = [chr(x).upper() for x in range(97, 123)]# 숫자로 결과가 나와서 라벨을 가져와서 알파벳이랑 매칭시켜줌
labels.remove("J")
labels.remove("Z")
 
while True:
    try:
        # 이미지 읽어오기
        ret, img = cap.read()
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        if hands:
            x, y, w, h = hands[0]['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
            imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
            aspectRatio = h/w
            if aspectRatio>1:
                k = imgSize/h
                wCal = math.ceil(k*w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize-wCal)/2)
                imgWhite[:, wGap:wCal+wGap] = imgResize
                # 이미지 읽어와서 인식을 한다 .getPrediction
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                # print(prediction, index)
            else:
                k = imgSize/w
                hCal = math.ceil(k*h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize-hCal)/2)
                imgWhite[hGap:hCal+hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
    
            cv2.rectangle(imgOutput, (x-offset, y-offset-50), (x-offset+90, y-offset-50+50), (255, 0, 139), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y-26), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x-offset, y-offset), (x+w+offset, y+h+offset), (255, 0, 139), 4)
            # cv2.imshow("ImageCrop", imgCrop)
            # cv2.imshow("ImageWhite", imgWhite) -> detector 로 이미지 인식할 때 사각형을 유동적으로 바꿔준다.
        if cv2.waitKey(1)==ord("q"): break
    except:
        print("카메라가 경계선 밖으로 나갔습니다.")
        break
    cv2.imshow("Sign Detectoin", imgOutput)
    
cap.release()
cv2.destroyAllWindows()
