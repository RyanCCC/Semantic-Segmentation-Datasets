'''
对图像进行切割
'''
import cv2
import time
import os
from tqdm import tqdm
import numpy as np
import random

angles = [0, 90, 180, 270]

def showImage(image, windowsName = 'result'):
    cv2.imshow(windowsName, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH))
    
save_path = './temp/'
ori = './temp/tokyo1_image.jpg'
lbl = './temp/tokyo1_image.png'
ori_img = cv2.imread(ori)
lbl_img = cv2.imread(lbl)
img_size = 500
step = 200
height, width, channle = ori_img.shape
for i in tqdm(range(0, (width-img_size), step)):
    for j in range(0, (height-img_size), step):
        time.sleep(1)
        angle = random.choice(angles)
        # 原图裁剪
        ori_cropped = ori_img[i:(i+img_size), j:(j+img_size)]
        ori_cropped = rotate_bound(ori_cropped, angle)
        filename = time.strftime('%Y%m%d%H%M%S')
        ori_filename = os.path.join(save_path+'JPEGImages',filename+'.jpg')
        cv2.imwrite(ori_filename, ori_cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        # 标签裁剪
        lbl_cropped = lbl_img[i:(i+img_size), j:(j+img_size)]
        lbl_cropped = rotate_bound(lbl_cropped, angle)
        lbl_filename = os.path.join(save_path+'Labels/',filename+'.png')
        cv2.imwrite(lbl_filename, lbl_cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
print('success')