import cv2
import numpy as np
import math


def main():
	# 画像の読み込み
	src_img = cv2.imread('../img/happy.png')
	#back_img = cv2.imread('../img/green_back.jpg')
	# 画像のリサイズ
	img = image_resize(src_img)

	

	'''
	### 読み込んだ画像に背景を付ける ###
	# 画像に合わせたNumPy配列を記録
	#height, width, channel = img.shape
	# 変換行列生成
	#mat = cv2.getRotationMatrix2D((width / 2, height / 2), 0, 0.5)
	# アフィン変換
	#img = cv2.warpAffine(img, mat, (500, 500), borderValue=(0,255, 0)) # アフィン変換
	#背景画像合成
	#height, width = img.shape[:2]
	#back_img[0:height, 0:width] = img
	'''            

	cv2.imshow('image', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


# 画像リサイズ
'''
def image_resize(image):
    img = image
    height, width = img.shape[:2]
    
    if 500 <= height or 500 <= width:
    img = cv2.resize(img , (int(width*0.5), int(height*0.5)))
    return img
'''



#パターン2
def image_resize(image):
    img = image
    height, width = img.shape[:2]
    
    img = cv2.resize(img, (450, int(450 * height / width)))
    
    return img



if __name__ == '__main__':
    main()
    