from statistics import mode

import cv2
from keras.models import load_model
import numpy as np

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import draw_bounding_box2
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input


def emotion_demo():
    # parameters for loading data and images
    detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
    emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
    emotion_labels = get_labels('fer2013')

    # 目のカスケードファイル追加
    lefteyecc__path = "../trained_models/detection_models/haarcascade_lefteye_2splits.xml"
    righteyecc_path = "../trained_models/detection_models/haarcascade_righteye_2splits.xml"
    nose_path = "../trained_models/detection_models/data_haarcascades_haarcascade_mcs_nose.xml"
    lefteyecc = cv2.CascadeClassifier(lefteyecc__path)
    righteyecc = cv2.CascadeClassifier(righteyecc_path)
    nose = cv2.CascadeClassifier(nose_path)
    lex = 0; ley = 0; lew = 0; leh = 0
    rex = 0; rey = 0; rew = 0; reh = 0
    nox = 0; noy = 0; now = 0; noh = 0

    # hyper-parameters for bounding boxes shape
    frame_window = 10
    emotion_offsets = (20, 40)

    # loading models
    face_detection = load_detection_model(detection_model_path)
    emotion_classifier = load_model(emotion_model_path, compile=False)

    # getting input model shapes for inference
    emotion_target_size = emotion_classifier.input_shape[1:3]

    # starting lists for calculating modes
    emotion_window = []

    global img, flag
    img = cv2.imread('../img/happy.png')
    flag = 0

    # starting video streaming
    cv2.namedWindow('window_frame')
    video_capture = cv2.VideoCapture(1) # 0は内蔵カメラ, 1はUSBカメラ
    while True:
        bgr_image = video_capture.read()[1]
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        faces = detect_faces(face_detection, gray_image)

        for face_coordinates in faces:
            #
            (x,y,w,h) = face_coordinates
            face = img[y:y+h,x:x+w]
            # 目や鼻認識用
            video_face = gray_image[y:y+h,x:x+w]

            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue


            # 左目の検出
            lefteyerects = lefteyecc.detectMultiScale(video_face)
            for lefteyerect in lefteyerects:
                (lex,ley,lew,leh) = lefteyerect
                #color = emotion_probability * np.asarray((255, 0, 0))
                #draw_bounding_box(lefteyerect, rgb_image, color)
                print("左目：", lex,ley,lew,leh)

            # 右目の検出
            righteyerects = righteyecc.detectMultiScale(video_face)
            for righteyerect in righteyerects:
                (rex,rey,rew,reh) = righteyerect
                print("右目：", rex,rey,rew,reh)

            # 鼻の検出
            noserects = nose.detectMultiScale(video_face)
            for noserect in noserects:
                (nox,noy,now,noh) = noserect
                cv2.rectangle(gray_face, (nox, noy), (nox + now, noy + noh), (0, 255, 0), 2)
                print("鼻：", nox,noy,now,noh)


            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = emotion_classifier.predict(gray_face)
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]
            emotion_window.append(emotion_text)

            if len(emotion_window) > frame_window:
                emotion_window.pop(0)
            try:
                emotion_mode = mode(emotion_window)
            except:
                continue

            if emotion_text == 'angry':
                #img = cv2.imread('../img/angry.png', -1)
                img = cv2.imread('../img/tuno2.png', -1)
                color = emotion_probability * np.asarray((255, 0, 0))
            elif emotion_text == 'sad':
                img = cv2.imread('../img/sad.png', -1) # 関数にする
                color = emotion_probability * np.asarray((0, 0, 255))
            elif emotion_text == 'happy':
                img = cv2.imread('../img/happy.png', -1)
                img = cv2.imread('../img/tuno2.png', -1)
                color = emotion_probability * np.asarray((255, 255, 0))
            elif emotion_text == 'surprise':
                img = cv2.imread('../img/odoroki.png', -1)
                color = emotion_probability * np.asarray((0, 255, 255))
            else:
                color = emotion_probability * np.asarray((0, 255, 0))

            color = color.astype(int)
            color = color.tolist()

            if flag == 0: # 別ウィンドウに画像を表示
                    draw_bounding_box(face_coordinates, rgb_image, color)
            elif flag == 1: # リアルタイムで顔に画像を載せる
                    #rgb_image = draw_bounding_box2(face_coordinates, rgb_image, color, img)
                    rgb_image = draw_bounding_box2(face_coordinates, rgb_image, color, img, emotion_text)
            draw_text(face_coordinates, rgb_image, emotion_mode,
                      color, 0, -45, 1, 1)

        if flag == 0:
            cv2.imshow('image', img)
        elif flag == 1:
            cv2.destroyWindow('image')
        cv2.waitKey(10)

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('window_frame', bgr_image)
        if cv2.waitKey(1) & 0xFF == ord('z'):
            flag = 0
        elif cv2.waitKey(1) & 0xFF == ord('x'):
            flag = 1
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break


    video_capture.release()
    cv2.destroyAllWindows()
