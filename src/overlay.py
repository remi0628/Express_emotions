import cv2
import numpy as np
from PIL import Image
from image_processing import image_resize


# 画像のオーバーレイ
def overlayImage(src, overlay, location):
    overlay_height, overlay_width = overlay.shape[:2]
    
    
    # 背景をPIL形式に変換
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    pil_src = Image.fromarray(src)
    pil_src = pil_src.convert('RGBA')
    
    
    # オーバーレイをPIL形式に変換
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGRA2RGBA)
    pil_overlay = Image.fromarray(overlay)
    pil_overlay = pil_overlay.convert('RGBA')

    # 画像を合成
    pil_tmp = Image.new('RGBA', pil_src.size, (255, 255, 255, 0))
    pil_tmp.paste(pil_overlay, location, pil_overlay)
    result_image = Image.alpha_composite(pil_src, pil_tmp)

    # OpenCV形式に変換
    return cv2.cvtColor(np.asarray(result_image), cv2.COLOR_RGBA2BGRA)





'''    
# 画像の読み込み
img_map = cv2.imread("../img/green_back.jpg")
img_sorami = cv2.imread("../img/odoroki.png", cv2.IMREAD_UNCHANGED)

# 画像のリサイズ
nimage_map = image_resize(img_map)
nimage_sorami = image_resize(img_sorami)

# 画像のオーバーレイ
image = overlayImage(nimage_map, nimage_sorami, (0, 0))

# ウィンドウ表示
cv2.namedWindow("image03", cv2.WINDOW_AUTOSIZE)
cv2.imshow("image03", image)
cv2.waitKey(0)
'''





