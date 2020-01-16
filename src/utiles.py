import json
import os
import cv2
from PIL import Image
import numpy as np


def creat_id2info_dic(json_path, save_path=None):
    """
    #根据images生成 图片id：图片信息字典
    :param json_path:
    :param save_path:
    :return:
    """
    id2info_dic = {}
    fb = open(json_path)
    json_data = json.load(fb)
    images_info = json_data['images']

    for img in images_info:
        id2info_dic[img['id']] = img


    print(id2info_dic)
    if save_path is not None:
        json_str = json.dumps(id2info_dic, indent = 4)
        fb = open(os.path.join(save_path, 'id2info_dic.json'), 'w')
        fb.write(json_str)

def get_id2info_dic(id2info_file):
    fb = open(id2info_file)
    json_data = json.load(fb)
    return json_data


def get_annotations(annotations_file):
    fb = open(annotations_file)
    json_data = json.load(fb)
    return json_data['annotations']

def get_crop_image(image, segmentation):
    pass
    # segmentation = np.array(segmentation, np.int32).reshape([-1, 2])
    # cv2.polylines(image, segmentation)


if __name__ == "__main__":
    json_path = '/home/hewaele/Desktop/coco数据集/annotations_trainval2014/annotations/instances_val2014.json'
    save_path = '../source_data/'
    creat_id2info_dic(json_path, save_path)