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