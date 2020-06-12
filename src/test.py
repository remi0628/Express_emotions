import cv2
import numpy as np
import math

#image = cv2.imread('../img/odoroki.png', -1)

src_img=cv2.imread('../img/odoroki.png', -1)
back_img=cv2.imread('../img/green_back.jpg')

h, w, c = src_img.shape
mat = cv2.getRotationMatrix2D((w / 2, h / 2), 0, 0.5)


img = cv2.warpAffine(src_img, mat, (500, 500), borderValue=(0, 128, 255))
            


cv2.imshow('image', img)


cv2.waitKey(0)
cv2.destroyAllWindows()