from statistics import mode

import cv2
from keras.models import load_model
from imutils import face_utils
import numpy as np
import dlib

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import draw_bounding_box2
from utils.inference import draw_bounding_box3
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from image_processing import image_resize
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

    # dlib
    dlib_ini()

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

    global img, flag, slp_count
    img = cv2.imread('../img/happy.png')
    flag = 0
    slp_count = 0

    # dlib用グローバル変数
    global gray_image, rgb_image, gray_face, mark_list

    # starting video streaming
    cv2.namedWindow('window_frame', cv2.WINDOW_NORMAL)
    video_capture = cv2.VideoCapture(0) # 0は内蔵カメラ, 1はUSBカメラ
    
    while True:
        bgr_image = video_capture.read()[1]
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGRA2RGBA)
        faces = detect_faces(face_detection, gray_image)
          

        for face_coordinates in faces:
            # 目や鼻認識用
            (x,y,w,h) = face_coordinates
            video_face = gray_image[y:y+h,x:x+w]
            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue

            # ランドマーク検出
            marks_list = marks_list_def(bgr_image, x, y, w, h)
            print(marks_list)


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

            if flag == 0 or flag == 1:
                if emotion_text == 'angry':
                    img = cv2.imread('../img/angry.png', -1)
                    color = emotion_probability * np.asarray((255, 0, 0))
                elif emotion_text == 'sad':
                    img = cv2.imread('../img/sad.png', -1) # 関数にする
                    color = emotion_probability * np.asarray((0, 0, 255))
                elif emotion_text == 'happy':
                    img = cv2.imread('../img/happy.png', -1)
                    color = emotion_probability * np.asarray((255, 255, 0))
                elif emotion_text == 'surprise':
                    img = cv2.imread('../img/odoroki.png', -1)
                    color = emotion_probability * np.asarray((0, 255, 255))
                else :
                    img = cv2.imread('../img/neutral.png', -1)
                    color = emotion_probability * np.asarray((0, 255, 0))
            else:    
                if emotion_text == 'angry':
                    img = cv2.imread('../img/ikari.png', -1)
                    color = emotion_probability * np.asarray((255, 0, 0))
                elif emotion_text == 'sad':
                    img = cv2.imread('../img/shock.png', -1) 
                    color = emotion_probability * np.asarray((0, 0, 255))
                elif emotion_text == 'happy':
                    img = cv2.imread('../img/kirakira.png', -1)
                    color = emotion_probability * np.asarray((255, 255, 0))
                elif emotion_text == 'surprise':
                    img = cv2.imread('../img/bikkuri.png', -1)
                    color = emotion_probability * np.asarray((0, 255, 255))
                else :
                    img = cv2.imread('../img/toumei.png', -1)
                    color = emotion_probability * np.asarray((0, 255, 0))
                

            color = color.astype(int)
            color = color.tolist()

            if flag == 0:
                draw_bounding_box(face_coordinates, rgb_image, color)
            elif flag == 1:
                rgb_image = draw_bounding_box2(face_coordinates, rgb_image, color, img, marks_list)              
            elif flag == 2:
                overlay_pic = draw_bounding_box3(face_coordinates, rgb_image, color, img, marks_list)
                rgb_image = overlay_pic       

            draw_text(face_coordinates, rgb_image, emotion_mode, color, 0, -45, 1, 1)
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)  
        
        
        if flag == 0:
            img = image_resize(img)
            cv2.imshow('image', img)
            cv2.destroyWindow('Window_frame')
        elif flag == 1 or flag == 2:
            cv2.destroyWindow('image')
            cv2.imshow('window_frame', bgr_image)
        cv2.waitKey(10)

        
        # cv2.imshow('own_window', bgr_image)

        if cv2.waitKey(1) & 0xFF == ord('z'):
            flag = 0
        elif cv2.waitKey(1) & 0xFF == ord('x'):
            flag = 1
        elif cv2.waitKey(1) & 0xFF == ord('c'):
            flag = 2
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break


    video_capture.release()
    cv2.destroyAllWindows()



## 以下dlib使用 59行目のグローバル変数追加必須
## http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 このURLからdatファイルをダウンロード
def dlib_ini():
    # Dlib
    global face_predictor, face_detector
    face_detector = dlib.get_frontal_face_detector()
    predictor_path = '../trained_models/dlib_model/shape_predictor_68_face_landmarks.dat'
    face_predictor = dlib.shape_predictor(predictor_path)

def marks_list_def(bgr_image, x, y, w, h):
    img_gry = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(img_gry, 1)
    faces_lib = dlib.rectangle(x, y, x+w, y+h) # 顔位置配列をdlib形式に変換

    # 顔のランドマーク検出
    landmark = face_predictor(img_gry, faces_lib)
    # 処理高速化のためランドマーク群をNumPy配列に変換(必須)
    landmark = face_utils.shape_to_np(landmark)

    # 右目, 左目, 鼻のランドマークを取得
    mark_nose = landmark[33]
    mark_left_eye = landmark[36]
    mark_right_eye = landmark[45]
    # マークを表示
    cv2.drawMarker(rgb_image, (mark_nose[0], mark_nose[1]), (21, 255, 12))
    cv2.drawMarker(rgb_image, (mark_left_eye[0], mark_left_eye[1]), (21, 255, 12))
    cv2.drawMarker(rgb_image, (mark_right_eye[0], mark_right_eye[1]), (21, 255, 12))

    mark_list = []
    re = [0, 0]
    le = [0, 0]
    no = [0, 0]
    try:
        re = [mark_right_eye[0], mark_right_eye[1]]
        le = [mark_left_eye[0], mark_left_eye[1]]
        no = [mark_nose[0], mark_nose[1]]
        mark_list.append(re)
        mark_list.append(le)
        mark_list.append(no)

        print('---------------------------------------')
        print(mark_list)
        print('右目：' + str(mark_list[0]))
        print('右目(x)：' + str(mark_list[0][0]))
        print('左目：' + str(mark_list[1]))
        print('左目(x)：' + str(mark_list[1][0]))
        print('鼻：' + str(mark_list[2]))
        print('鼻(x)：' + str(mark_list[2][0]))

        """
        # ランドマーク全描画
        for (x, y) in landmark:
            cv2.circle(rgb_image, (x, y), 1, (0, 0, 255), -1)
        """
    except: # 取得できなかったりエラーの場合は全て0を返す
        mark_list.append(re)
        mark_list.append(le)
        mark_list.append(no)

    return mark_list
    # mark_list[[re(x), re(y)], [le(x), le(y)], [no(x), no(y)]]