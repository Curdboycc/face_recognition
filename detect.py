#coding=utf-8
import cv2
import face_recognition
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
#path = r'D:\faces'
path = r"D:\work_myself\face_py36\faces"
images = []
classNames = []
mylist = os.listdir(path)
for cl in mylist:
    #curImg = cv2.imread(f'{path}/{cl}')
    curImg = cv2.imdecode(np.fromfile(f'{path}/{cl}',dtype=np.uint8),-1)
    images.append(curImg)
    img_name = cl.split('.')[0]
    classNames.append(img_name)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_face = face_recognition.face_encodings(img)[0]
        encodeList.append(encoded_face)
    return encodeList

encoded_face_train = findEncodings(images)

def markAttendance(name):
    with open('D:\Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            time = now.strftime('%I:%M:%S:%p')
            date = now.strftime('%d-%B-%Y')
            f.writelines(f'n{name}, {time}, {date}')





# imgelon_bgr = face_recognition.load_image_file('1.png')
# imgelon_rgb = cv2.cvtColor(imgelon_bgr,cv2.COLOR_BGR2RGB)
# face = face_recognition.face_locations(imgelon_rgb)[0]
# copy = imgelon_rgb.copy()
# cv2.rectangle(copy, (face[3], face[0]),(face[1], face[2]), (255,0,255), 2)
# cv2.imshow('copy', copy)
# cv2.imshow('elon',imgelon_rgb)
# cv2.waitKey(0)

#train_encodings = face_recognition.face_encodings(imgelon_rgb)[0]

# test = face_recognition.load_image_file('2.jpg')
# test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
# test_encode = face_recognition.face_encodings(test)[0]
# print(face_recognition.compare_faces([train_elon_encodings],test_encode))


# def findEncodings(images):
#     encodeList = []
#     for img in images:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encoded_face = face_recognition.face_encodings(img)[0]
#         encodeList.append(encoded_face)
#     return encodeList
# encoded_face_train = findEncodings(images)
#
#
def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

from datetime import datetime
import pickle
# take pictures from webcam
cap  = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faces_in_frame = face_recognition.face_locations(imgS)
    encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)
    for encode_face, faceloc in zip(encoded_faces,faces_in_frame):
        matches = face_recognition.compare_faces(encoded_face_train, encode_face)
        faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
        matchIndex = np.argmin(faceDist)
        print(matchIndex)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper().lower()

            y1,x2,y2,x1 = faceloc
            # since we scaled down by 4 times
            y1, x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img, (x1,y2-35),(x2,y2), (0,255,0), cv2.FILLED)
            print(name)
            #cv2.putText(img,name, (x1+6,y2-5), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            img = cv2ImgAddText(img,name,x1+6,y2-35,(0,0,0),20)
            markAttendance(name)
    cv2.imshow('webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break