## 使用labelme制作语义分割数据集（以buildings为例）
1. 编写labels.txt，内容包括背景(background)和类别名称，多类别就写上你的类别名称。buildings的labels.txt文档内容如下：

   > background
   >
   > buildings
   
2. 将路径切换到要打标签的图像文件夹所在的目录下，执行命令：labelme buildings --labels labels.txt

3. 将labelme设置成'Saved Automatically'（点击labelme软件左上角的file就有此选项），然后开始标记工作。

4. 标记完成后保存的是json数据，路径在标记图像的目录下，接着需要将json文件转换成VOC格式。假设将转换的图像保存在buildings_VOC下。执行命令：python label2VOC.py buildings buildings_VOC --labels labels.txt。labels.txt就是你一开始编写的txt文件。

   ![image-20210531115110407](./src/image-20210531115110407.png)

   label2VOC.py代码如下：

   ```python
   #!/usr/bin/env python
   # labelme data_annotated --labels labels.txt --nodata --validatelabel exact
   # ./labelme2voc.py data_annotated data_dataset_voc --labels labels.txt
   from __future__ import print_function
   
   import argparse
   import glob
   import os
   import os.path as osp
   import sys
   
   import imgviz
   import numpy as np
   
   import labelme
   
   
   def main():
       parser = argparse.ArgumentParser(
           formatter_class=argparse.ArgumentDefaultsHelpFormatter
       )
       parser.add_argument("input_dir", help="input annotated directory")
       parser.add_argument("output_dir", help="output dataset directory")
       parser.add_argument("--labels", help="labels file", required=True)
       parser.add_argument(
           "--noviz", help="no visualization", action="store_true"
       )
       args = parser.parse_args()
   
       if osp.exists(args.output_dir):
           print("Output directory already exists:", args.output_dir)
           sys.exit(1)
       os.makedirs(args.output_dir)
       os.makedirs(osp.join(args.output_dir, "JPEGImages"))
       os.makedirs(osp.join(args.output_dir, "SegmentationClass"))
       os.makedirs(osp.join(args.output_dir, "SegmentationClassPNG"))
       if not args.noviz:
           os.makedirs(
               osp.join(args.output_dir, "SegmentationClassVisualization")
           )
       print("Creating dataset:", args.output_dir)
   
       class_names = []
       class_name_to_id = {}
       for i, line in enumerate(open(args.labels).readlines()):
           class_id = i - 1  # starts with -1
           class_name = line.strip()
           class_name_to_id[class_name] = class_id
           if class_id == -1:
               assert class_name == "__ignore__"
               continue
           elif class_id == 0:
               assert class_name == "_background_"
           class_names.append(class_name)
       class_names = tuple(class_names)
       print("class_names:", class_names)
       out_class_names_file = osp.join(args.output_dir, "class_names.txt")
       with open(out_class_names_file, "w") as f:
           f.writelines("\n".join(class_names))
       print("Saved class_names:", out_class_names_file)
   
       for filename in glob.glob(osp.join(args.input_dir, "*.json")):
           print("Generating dataset from:", filename)
   
           label_file = labelme.LabelFile(filename=filename)
   
           base = osp.splitext(osp.basename(filename))[0]
           out_img_file = osp.join(args.output_dir, "JPEGImages", base + ".jpg")
           out_lbl_file = osp.join(
               args.output_dir, "SegmentationClass", base + ".npy"
           )
           out_png_file = osp.join(
               args.output_dir, "SegmentationClassPNG", base + ".png"
           )
           if not args.noviz:
               out_viz_file = osp.join(
                   args.output_dir,
                   "SegmentationClassVisualization",
                   base + ".jpg",
               )
   
           with open(out_img_file, "wb") as f:
               f.write(label_file.imageData)
           img = labelme.utils.img_data_to_arr(label_file.imageData)
   
           lbl, _ = labelme.utils.shapes_to_label(
               img_shape=img.shape,
               shapes=label_file.shapes,
               label_name_to_value=class_name_to_id,
           )
           labelme.utils.lblsave(out_png_file, lbl)
   
           np.save(out_lbl_file, lbl)
   
           if not args.noviz:
               viz = imgviz.label2rgb(
                   label=lbl,
                   img=imgviz.rgb2gray(img),
                   font_size=15,
                   label_names=class_names,
                   loc="rb",
               )
               imgviz.io.imsave(out_viz_file, viz)
   
   
   if __name__ == "__main__":
       main()
   
   ```

5. 生成完成后的文件架构如下：JPEGImages保存的是原图。SegmentationClass保存的是npy的文件，是原图的标签，可以用于训练，如何训练.npy文件后面再研究。SegmentationClassPNG保存的是标签，是单通道的图像，以调色板格式保存的。SegmentationClassVisualization保存的是原图+标签，用于可是化打标签的结果。class_names.txt用于保存标签的类别。以JPEGImages下的20210531094423.jpg为例。原图、标签以及可视化图如下图所示。
    
  ![image-20210531115110407](./src/image-20210531115219780.png)
      
  ![image-20210531115110407](./src/image-20210531115128778.png)
         
  ![image-20210531115110407](./src/image-20210531115138232.png)
            
  ![image-20210531115110407](./src/image-20210531115145596.png)
      

6. 到此，可以说数据标签已经完毕了。可以使用JPEGImages下的图像作为原图，SegmentationClassPNG作为标签送入网络进行训练。但是我们一般用单通道，0,1,2,3表示类别，因此我们也可以再做一次转换，将SegmentationClassPNG转换成那种全黑的图。（这里我不太会表达。）修改一下ConvertVOC2Gray.py里面的图像的路径，然后执行文件即可。ConvertVOC2Gray.py主要修改以下代码（路径以及保存路径）：

   ```python
   tf.flags.DEFINE_string('original_gt_folder',#读取voc格式的png图片路径
                                    r'.\buildings_VOC\SegmentationClassPNG',#default
                                    'Original ground truth annotations.')#help
   tf.flags.DEFINE_string('segmentation_format', 'png', 'Segmentation format.')
   tf.flags.DEFINE_string('output_dir',#保存路径
                                    r'.\buildings_VOC\labels',
                                    'folder to save modified ground truth annotations.')
   
   ```

   ```python
   """Removes the color map from segmentation annotations.
   Removes the color map from the ground truth segmentation annotations and save
   the results to output_dir.
   """
   #源程序链接：https://github.com/tensorflow/models/tree/master/research/deeplab/datasets
   #该程序用于将voc格式的索引图片转换为可用于语义分割的灰度图片，因为TensorFlow版本的缘故，在原程序上做了少许改动
   #tf.__version__==1.12
   
   import glob
   import os.path
   import numpy as np
   from PIL import Image
   import tensorflow as tf
   
   #FLAGS = tf.compat.v1.flags.FLAGS
   FLAGS = tf.flags.FLAGS
   tf.flags.DEFINE_string('original_gt_folder',#读取voc格式的png图片路径
                                    r'.\cityBuilding_VOC\SegmentationClassPNG',#default
                                    'Original ground truth annotations.')#help
   tf.flags.DEFINE_string('segmentation_format', 'png', 'Segmentation format.')
   tf.flags.DEFINE_string('output_dir',#保存路径
                                    r'.\cityBuilding_VOC\labels',
                                    'folder to save modified ground truth annotations.')
   
   
   def _remove_colormap(filename):
     """Removes the color map from the annotation.
     Args:
       filename: Ground truth annotation filename.
     Returns:
       Annotation without color map.
     """
     return np.array(Image.open(filename))
   
   
   def _save_annotation(annotation, filename):
     """Saves the annotation as png file.
     Args:
       annotation: Segmentation annotation.
       filename: Output filename.
     """
     
     pil_image = Image.fromarray(annotation.astype(dtype=np.uint8))
     '''
     with tf.io.gfile.GFile(filename, mode='w') as f:
     #with open(filename, mode='w') as f:
       print(f)
       pil_image.save(f, 'PNG')
       '''
     pil_image.save(filename)
   
   
   def main(unused_argv):
     # Create the output directory if not exists.
     if not os.path.exists(FLAGS.output_dir):
       os.makedirs(FLAGS.output_dir)
     #if not tf.io.gfile.isdir(FLAGS.output_dir):
       #tf.io.gfile.makedirs(FLAGS.output_dir)
   
     annotations = glob.glob(os.path.join(FLAGS.original_gt_folder,
                                          '*.' + FLAGS.segmentation_format))
     for annotation in annotations:
       raw_annotation = _remove_colormap(annotation)
       filename = os.path.basename(annotation)[:-4]
       _save_annotation(raw_annotation,
                        os.path.join(
                            FLAGS.output_dir,
                            filename + '.' + FLAGS.segmentation_format))
   
   
   if __name__ == '__main__':
     #tf.compat.v1.app.run()
     tf.app.run()
   ```

   然后可以看到building_VOC下的labels文件夹有的标签图。虽然是全黑，但是通过np.unique可以看到有0,1两种类别的像素。到此我们的数据集已经制作完成了。

   ![image-20210531115748487](./src/image-20210531115748487.png)

7. 参考

   https://github.com/wkentaro/labelme/tree/master/examples/semantic_segmentation

   https://blog.csdn.net/Winters____/article/details/105659632

## 制作实例分割数据集

1. 类似语义分割数据集一样，对图像进行标注，生成json文件。要注意一点的是：不同的实例之间的标签尽可能不一样，如person的实例分为person.1，person.2，当然你也可以按照你的风格用不同的分割符，如“，”，“。”等。这样方便后续生成yaml文件。为什么要这样做？看一下第二步json_to_dataset就知道了。

```json
"shapes": [
    {
      "label": "sign",
      "points": [
        [
          1065.8518518518517,
          535.5925925925926
        ],
        [
          954.7407407407406,
          744.8518518518518
        ],
        [
          791.7777777777776,
          1133.7407407407406
        ],
        [
          791.7777777777776,
          1174.4814814814813
        ],
        [
          1049.185185185185,
          1170.7777777777776
        ],
        [
          1389.9259259259259,
          1135.5925925925926
        ],
        [
          1389.9259259259259,
          1102.2592592592591
        ],
        [
          1119.5555555555554,
          561.5185185185185
        ],
        [
          1099.185185185185,
          531.8888888888888
        ]
      ],
      "group_id": 1,
      "shape_type": "polygon",
      "flags": {}
    },
    {
      "label": "sign.1",
      "points": [
        [
          443.62962962962956,
          1259.6666666666665
        ],
        [
          778.8148148148148,
          1567.074074074074
        ],
        [
          754.7407407407406,
          1352.2592592592591
        ],
        [
          423.25925925925924,
          1074.4814814814815
        ]
      ],
      "group_id": null,
      "shape_type": "polygon",
      "flags": {}
    },
    {
      "label": "sign",
      "points": [
        [
          2136.222222222222,
          559.6666666666666
        ],
        [
          2264.0,
          807.8148148148148
        ],
        [
          2591.7777777777774,
          793.0
        ],
        [
          2539.9259259259256,
          596.7037037037037
        ],
        [
          2338.074074074074,
          441.1481481481481
        ],
        [
          2260.296296296296,
          437.4444444444444
        ]
      ],
      "group_id": 3,
      "shape_type": "polygon",
      "flags": {}
    }
  ]
```

2. 生成训练集，运行json_to_dataset文件，注意自己的路径以及实例标签的分隔符。

```python

import json
import os
import os.path as osp
import warnings
import PIL.Image
import yaml
from labelme import utils
import base64
from glob import glob
import shutil

'''
labelme:4.5.7
args:
    dataset: 使用labelme标注完成的数据集，其中包括标注图以及json文件存放的位置
    train_dir：保存的待训练的数据集，imgs--原图   mask--掩膜  yaml--yaml存放
注意：标注的时候后面以.区分实例，如person.1， person.2，以此类推
'''
def main(dataset = "./dataset/JPEGImages", train_dir = './dataset', env_command = 'conda activate labelme', split_flag = '.'):
    # 批量处理json文件
    if env_command is not None:
        os.system(env_command)
    json_file = os.path.join(dataset, '*.json')
    for item in glob(json_file):
        os.system(f"labelme_json_to_dataset.exe {item}")

    save_path = os.path.join(dataset, '*.json')
    for json_path in glob(save_path):
        # 保存yaml文件
        json_basename = os.path.basename(json_path).split('.')[0]
        data = json.load(open(json_path))
        if data['imageData']:
            imageData = data['imageData']
        else:
            imagePath = os.path.join(os.path.dirname(json_path), data['imagePath'])
            with open(imagePath, 'rb') as f:
                imageData = f.read()
                imageData = base64.b64encode(imageData).decode('utf-8')

        label_name_to_value = {'_background_': 0}
        for shape in data['shapes']:
            label_name = shape['label']
            if label_name in label_name_to_value:
                label_value = label_name_to_value[label_name]
            else:
                label_value = len(label_name_to_value)
                label_name_to_value[label_name] = label_value
            
            # label_values must be dense
            label_values, label_names = [], []
            for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
                label_values.append(lv)
                label_names.append(ln.split(split_flag)[0])
            
            assert label_values == list(range(len(label_values)))            
            yaml_path = os.path.join(train_dir, "yaml")
            if not os.path.exists(yaml_path):
                os.mkdir(yaml_path)
            warnings.warn('info.yaml is being replaced by label_names.txt')
            info = dict(label_names=label_names)
            with open(osp.join(yaml_path, str(json_basename)+'.yaml'), 'w') as f:
                yaml.safe_dump(info, f, default_flow_style=False)
        print('Saved : %s' % str(json_basename+'.yaml'))
        # copy ori image to imgs and rename 
        src_path = os.path.join(dataset, json_basename+'_json', 'img.png')
        dst_path = os.path.join(train_dir, 'images')
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)
        shutil.copy(src_path, os.path.join(dst_path, json_basename+'.png'))
        # copy mask img to mask
        src_path = os.path.join(dataset, json_basename+'_json', 'label.png')
        dst_path = os.path.join(train_dir, 'mask')
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)
        shutil.copy(src_path, os.path.join(dst_path, json_basename+'.png'))
if __name__ == '__main__':
    main()

```

3. 最终设定的目录下生成images、mask、yaml文件夹，分别存放原图，掩膜图，yaml文件。具体的Demo可以参照Instance_dataset_demo