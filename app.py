from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import dlib
from math import hypot
import sys
from time import time
import gdown
import os

app = Flask(__name__)
camera = cv2.VideoCapture(0)


PATH = './shape_predictor_68_face_landmarks.dat'
if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
    print("file exist")
else:
    output = "shape_predictor_68_face_landmarks.dat"
    url = "https://drive.google.com/uc?id=1mQcB-yIrrRq6gShA6br1VYAzqiKGD0Mf"
    gdown.download(url, output)



def main_frames(name):
    if name == 'pig':
        nose_image = cv2.imread("pig_nose.png")
        _, frame1 = camera.read()
        rows, cols, _ = frame1.shape
        nose_mask = np.zeros((rows, cols), np.uint8)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(
            "shape_predictor_68_face_landmarks.dat")
        while True:
            success, frame1 = camera.read()
            try:
                nose_mask.fill(0)
                gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                faces = detector(gray_frame1)
                for face in faces:
                    landmarks = predictor(gray_frame1, face)
                    top_nose = (landmarks.part(29).x, landmarks.part(29).y)
                    center_nose = (landmarks.part(30).x, landmarks.part(30).y)
                    left_nose = (landmarks.part(31).x, landmarks.part(31).y)
                    right_nose = (landmarks.part(35).x, landmarks.part(35).y)
                    nose_width = int(hypot(left_nose[0] - right_nose[0],
                                           left_nose[1] - right_nose[1]) * 1.7)
                    nose_height = int(nose_width * 0.77)
                    top_left = (int(center_nose[0] - nose_width / 2),
                                int(center_nose[1] - nose_height / 2))
                    bottom_right = (int(center_nose[0] + nose_width / 2),
                                    int(center_nose[1] + nose_height / 2))
                    nose_pig = cv2.resize(
                        nose_image, (nose_width, nose_height))
                    nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)
                    _, nose_mask = cv2.threshold(
                        nose_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)
                    nose_area = frame1[top_left[1]: int(top_left[1] + nose_height),
                                       top_left[0]: int(top_left[0] + nose_width)]
                    nose_area_no_nose = cv2.bitwise_and(
                        nose_area, nose_area, mask=nose_mask)
                    final_nose = cv2.add(nose_area_no_nose, nose_pig)
                    frame1[top_left[1]: top_left[1] + nose_height,
                           top_left[0]: top_left[0] + nose_width] = final_nose
            except:
                _, frame2 = camera.read()
                ret, buffer = cv2.imencode('.jpg', frame2)
                frame2 = buffer.tobytes()
                yield (b'--frame2\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')
            else:
                ret, buffer = cv2.imencode('.jpg', frame1)
                frame1 = buffer.tobytes()
                yield (b'--frame1\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame1 + b'\r\n')
          
    if name == 'sunglass':
        sunglass_image = cv2.imread("sunglass.png")
        _, frame3 = camera.read()
        rows, cols, _ = frame3.shape
        sunglass_mask = np.zeros((rows, cols), np.uint8)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(
            "shape_predictor_68_face_landmarks.dat")
        while True:
            success, frame3 = camera.read()
            try:
                sunglass_mask.fill(0)
                gray_frame3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
                faces = detector(gray_frame3)
                for face in faces:
                    landmarks = predictor(gray_frame3, face)
                    center_sunglass = (landmarks.part(28).x,
                                       landmarks.part(28).y)
                    left_sunglass = (landmarks.part(
                        31).x, landmarks.part(31).y)
                    right_sunglass = (landmarks.part(35).x,
                                      landmarks.part(35).y)
                    sunglass_width = int(hypot(left_sunglass[0] - right_sunglass[0],
                                               left_sunglass[1] - right_sunglass[1]) * 5)
                    sunglass_height = int(sunglass_width * 0.77)
                    top_left = (int(center_sunglass[0] - sunglass_width / 2),
                                int(center_sunglass[1] - sunglass_height / 2))
                    sunglass = cv2.resize(
                        sunglass_image, (sunglass_width, sunglass_height))
                    sunglass_gray = cv2.cvtColor(sunglass, cv2.COLOR_BGR2GRAY)
                    _, sunglass_mask = cv2.threshold(
                        sunglass_gray, 25, 255, cv2.THRESH_BINARY_INV)
                    sunglass_area = frame3[top_left[1]: int(top_left[1] + sunglass_height),
                                           top_left[0]: int(top_left[0] + sunglass_width)]
                    sunglass_area_no_sunglass = cv2.bitwise_and(
                        sunglass_area, sunglass_area, mask=sunglass_mask)
                    final_sunglass = cv2.add(
                        sunglass_area_no_sunglass, sunglass)
                    frame3[top_left[1]: top_left[1] + sunglass_height,
                           top_left[0]: top_left[0] + sunglass_width] = final_sunglass
            except:
                _, frame4 = camera.read()
                ret, buffer = cv2.imencode('.jpg', frame4)
                frame2 = buffer.tobytes()
                yield (b'--frame4\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame4 + b'\r\n')
            else:
                ret, buffer = cv2.imencode('.jpg', frame3)
                frame3 = buffer.tobytes()
                yield (b'--frame3\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame3 + b'\r\n')


        
@app.route('/video_feed/1')
def video_feed_pig():
    print(request.base_url, file=sys.stdout)
    return Response(main_frames('pig'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed/2')
def video_feed_sunglass():
    print(request.base_url, file=sys.stdout)
    return Response(main_frames('sunglass'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/pig')
def pig():
    return render_template('pig.html')


@app.route('/sunglass')
def sunglass():
    return render_template('sunglass.html')


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
