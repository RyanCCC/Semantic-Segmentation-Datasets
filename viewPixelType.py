import numpy as np
from PIL import Image

image_path = './sidewalk_VOC/labels/1.png'
image = Image.open(image_path)
print(np.unique(image))
