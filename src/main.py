import sys
import cv2
import tkinter as tk
import tkinter.filedialog
from PIL import Image, ImageTk

from statistics import mode

from keras.models import load_model
import numpy as np

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

# face_classificationのファイル
import video_emotion_color_demo
from video_emotion_color_demo import emotion_demo
#from image_processing import image_pro

def main():
    emotion_demo()
    #image_pro()



'''
def image_gui():
    global label, label1, label2
    global img1, img2
    # アプリの作成
    app = tkinter.Tk()

    # アプリの画面設定
    app.geometry("500x600") # アプリ画面のサイズ
    app.title("Express_emotions") # アプリのタイトル

    # 画像設定
    img1 = ImageTk.PhotoImage(image_resize('../img/happy.png'))
    img2 = ImageTk.PhotoImage(image_resize('../img/odoroki.png'))

    # ラベル
    label = tkinter.Label(app, font = ("System", 30), text = "happy")
    label.place(x = 50, y = 500)
    label1 = tkinter.Label(app,image=img1)
    label1.grid(row = 0, column = 0)
    label2 = tkinter.Label(app)
    label2.grid(row = 0, column = 0)

    #keyイベント判定
    app.bind("<KeyPress>", press_key_func)

    # アプリの待機
    app.mainloop()



# 画像切替
def push1():
    label["text"] = "happy"
    label1.configure(image=img1)
    label2.configure(image='')

def push2():
    global emotion_text
    label["text"] = emotion_text
    label1.configure(image='')
    label2.configure(image=img2)

#key取得
def press_key_func(event):
    key = event.keysym # 入力されたキーを取得
    if key == "Left":
        push1()
    elif key == "Right":
        push2()

#画像のリサイズ tk用
def image_resize(image):
    img = Image.open(image)
    w = img.width # 横幅を取得
    h = img.height # 縦幅を取得
    if 500 <= h or 450 <= w:
        img = img.resize(( int(w * (450/h)), int(h * (450/h)) ))
    return img
'''



if __name__ == '__main__':
    main()