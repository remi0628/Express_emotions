import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
from overlay import overlayImage


def load_image(image_path, grayscale=False, target_size=None):
    pil_image = image.load_img(image_path, grayscale, target_size)
    return image.img_to_array(pil_image)

def load_detection_model(model_path):
    detection_model = cv2.CascadeClassifier(model_path)
    return detection_model

def detect_faces(detection_model, gray_image_array):
    return detection_model.detectMultiScale(gray_image_array, 1.3, 5)

def draw_bounding_box(face_coordinates, image_array, color):
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)


'''
def draw_bounding_box2(face_coordinates, image_array, color, img):
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    
    img_face = cv2.resize(img, (int(w), int(h)))  #変更＊　数を一致させる
    img2 = image_array.copy()
    img2[y:y+int(h), x:x+int(w)] = img_face       #変更＊　数を一致させる
    return img2


def draw_bounding_box3(face_coordinates, image_array, color, img, frame):
    x, y, w, h = face_coordinates
    ny = y - 50
    # cv2.rectangle(image_array, (x, ny), (x + w, y + h), color, 2)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        
    img_face = cv2.resize(img, (int(w/5), int(h/5) + 50))  #変更＊　数を一致させる
    img2 = image_array.copy()
    img2[ny:ny+int(h/5)+50, x:x+int(w/5)] = img_face       #変更＊　数を一致させる
    
    img3 = overlayImage(frame, img2, (0, 0))
    return img3
'''

def draw_bounding_box2(face_coordinates, image_array, color, img, marks_list):
    x, y, w, h = face_coordinates
    
    h1 = marks_list[2][0] - marks_list[1][0]
    h2 = marks_list[2][1] - marks_list[1][1]
    h3 = marks_list[0][0] - marks_list[2][0]
    h4 = marks_list[0][1] - marks_list[2][1]
    n1 = marks_list[1][0] - h1
    n2 = marks_list[1][1] - h2
    
    # 角度を変更
    '''
    if h1 > h3:
        angle = h1//10
    else:
        angle = -h3//10
    '''

    # cv2.rectangle(image_array, (n1, n2), (n1 + w, n2 + h), color, 2)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    

    # mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 0.5)
    img_face = cv2.resize(img, (w, h))  #変更＊　数を一致させる
    # img_face = cv2.warpAffine(img_face, mat, (w, h))
    img2 = image_array.copy()
    img2[n2:n2+h, n1:n1+w] = img_face       #変更＊　数を一致させる
    return img2

def draw_bounding_box3(face_coordinates, image_array, color, img, frame, marks_list):
    x, y, w, h = face_coordinates
    ny = y - 50
    m, n = 5, 1
    
    '''
    h1 = marks_list[2][0] - marks_list[1][0]
    h2 = marks_list[2][1] - marks_list[1][1]
    n1 = marks_list[1][0] - h1
    n2 = marks_list[1][1] - h2
    '''
    
    n1 = (-n * marks_list[2][0] + m * marks_list[1][0]) // m - n
    n2 = (-n * marks_list[2][1] + m * marks_list[1][1]) // m - n
    

    # cv2.rectangle(image_array, (n1, n2), (n1 + w, n2 + h), color, 2)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        
    img_face = cv2.resize(img, (int(w/5), int(h/5)))  #変更＊　数を一致させる
    img2 = image_array.copy()
    img2[n2:n2+int(h/5), n1:n1+int(w/5)] = img_face       #変更＊　数を一致させる
    
    img3 = overlayImage(frame, img2, (0, 0))
    return img3

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0,
                                                font_scale=2, thickness=2):
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (x + x_offset, y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)

def get_colors(num_classes):
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()
    colors = np.asarray(colors) * 255
    return colors

