#数据及的制作
"""
1: 遍历 annotations keys
2: 获取图片id
3: 根据图片id获取改张图片信息
4: 获取annotations的area数据
5: 根据面积占比对mask进行筛选，跳过面积占比大于10的mask
6： 获取segmentation信息
7： 打开原始图片并裁剪得到mask部分
    裁剪目标方式，性生成一个只有目标的图片m1
    对图片以及mask图片进行旋转，缩放，确定上下左右移动位置
        上下左右移动位置方案：

8： 将原始图片和trans——image图片重叠，得到copy move图片
9： 生成copy move标签
10： 将训练图片和标签进行同名分文件及保存
"""
import json
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
from crop_test import get_cm_image_mask, get_new_iamge
from utiles import get_annotations, get_id2info_dic


def main():
    source_image_path = '/home/hewaele/Desktop/coco数据集/val2014'
    annotations_path = '/home/hewaele/Desktop/coco数据集/annotations_trainval2014/annotations/instances_val2014.json'
    images_path = '../cmfd_data/images_v2'
    mask_path = '../cmfd_data/mask_v2'
    id2info_path = '../source_data/id2info_dic.json'

    count = 0
    id2info_data = get_id2info_dic(id2info_path)
    annotations = get_annotations(annotations_path)
    for pos, annotation in enumerate(annotations[:]):
        image_id = annotation['image_id']
        image_info = id2info_data[str(image_id)]
        annotation_area = annotation['area']
        # print(image_info['file_name'])
        h, w = image_info['height'], image_info['width']
        if 0.0005 <= annotation_area/(h*w) <= 0.1:
            try:
                #获取mask标注信息
                #[x1 y1, x2, y2, x3, y3 ......]
                segmentation = annotation['segmentation'][0]
                image = Image.open(os.path.join(source_image_path, image_info['file_name']))
                image = np.array(image)
                #获得copy move
                trans_image, trans_mask = get_cm_image_mask(image, np.array(segmentation, np.int32).reshape([-1, 2]))

                #将原始图片和trans image合并
                new_image = get_new_iamge(image, trans_image, trans_mask)
                # plt.subplot(221)
                # plt.imshow(image)
                # plt.subplot(222)
                # plt.imshow(trans_image)
                # plt.subplot(223)
                # plt.imshow(trans_mask)
                # plt.subplot(224)
                # plt.imshow(new_image)
                # # plt.scatter(segmentation[::2], segmentation[1::2])
                # plt.show()
                # break
                #
                # #将生成的结果图片保存
                save_image = Image.fromarray(new_image)
                save_mask = Image.fromarray(trans_mask)
                save_image.save(os.path.join(images_path, 'image_'+str(count)+'.png'))
                save_mask.save(os.path.join(mask_path, 'mask_'+str(count)+'.png'))
                print(count)
                count += 1

            except:
                print('error')

            if count >= 150000:
                break

    print(count)
    print('done')
if __name__ == '__main__':
    main()

