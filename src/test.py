from msvcrt import getch
import cv2
import numpy as np

img = cv2.imread('../img/happy.png')
img2 = cv2.imread('../img/odoroki.png')

cv2.imshow('image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()

def select():
    cv2.imshow('image', img)

def moveDown():
    cv2.imshow('image', img2)

def moveUp():
    cv2.imshow('image', img2)

'''
while True:
    key = ord(getch())
    if key == 27: #エスケープ
        break
    elif key == 13: #エンター
        select()

    elif key == 224: #スペシャルキー（矢印、Fキー、ins、del、など）
        key = ord(getch())
        if key == 80: #上矢印
            moveDown()
        elif key == 72: #下矢印
            moveUp()
'''