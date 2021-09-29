import os
import re
import sys
import time
import glob
from tqdm import tqdm
import cv2


import cv2
import os
from datetime import datetime
base_path = r'D:\Datasets\village'
image_path = os.path.join(base_path,'JPEGImages')
annotation = os.path.join(base_path, 'Annotations')



# 查找未标注文件
# for item in tqdm(glob.glob(image_path+'/*')):
#     filename = os.path.basename(item)
#     _filename = os.path.splitext(filename)
#     annotation_file = os.path.join(annotation, _filename[0]+'.xml')
#     if not os.path.exists(annotation_file):
#         print(filename)

# print('finish check file')

# 转换成jpg文件
for item in tqdm(glob.glob(image_path+'/*')):
    # 先改名字
    time.sleep(1)
    new_filename = 'camera_'+datetime.now().strftime("%Y%m%d%H%M%S")+'.jpg'
    save_filename = os.path.join(image_path,new_filename)
    os.rename(item, save_filename)
    # img = cv2.imread(save_filename)
    # cv2.imwrite(save_filename, img)

base_path = r'./TaiPingImage/'
resize_ratio = 2.5
print('finish rename')
# # resize 图像
# base_path = './tmp/'
# for filename in tqdm(glob.glob(os.path.join(base_path+'*.jpg'))):
#     img= cv2.imread(filename)
#     height, width, _ = img.shape
#     # 注意resize是int类型
#     img = cv2.resize(img, (int(width//resize_ratio), int(height//resize_ratio)),interpolation=cv2.INTER_CUBIC)
#     save_filename= os.path.join('./tmp/result/'+time.strftime("%Y%m%d%H%M%S")+'.jpg')
#     cv2.imwrite(save_filename, img)
#     time.sleep(1)

