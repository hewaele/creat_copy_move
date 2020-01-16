import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import copy
import random

#需要的参数： 图片的尺寸w, h, mask信息
def get_movexy(w, h, segmentation):
    top, buttom = np.min(segmentation[:, 1]), np.max(segmentation[:, 1])
    left, right = np.min(segmentation[:, 0]), np.max(segmentation[:, 0])

    #根据上下左右预留的空间确定移动方向
    space_top = top
    space_buttom = h - buttom

    space_left = left
    space_right = w - right

    min_space_bounder = min(space_top, space_buttom, space_left, space_right)
    #确保不重叠
    #先确定水平和垂直方向其中一个不重叠
    #如果目标高度占比小于宽度占比，则确保上下不重叠
    if (buttom - top)/h <= (right - left)/w:
        #先移动上下，保证上下不重叠
        #上移
        if space_top >= space_buttom:
            move_h = -1*random.randint(int((buttom-top)*1.2), max(int((buttom-top)*1.21), int(space_top*0.8)))
        #下移
        else:
            move_h = random.randint(int((buttom-top)*1.2), max(int((buttom-top)*1.21), int(space_buttom*0.8)))
        #左右的移动随机移动
        move_w = random.randint(int(-space_left * 0.8), int(space_right * 0.8))
    else:
        #先移动左右，保证左右不重叠
        #左移
        if space_left >= space_right:
            move_w = -1*random.randint(int((right-left)*1.2), max(int((right-left)*1.21), int(space_left*0.8)))
        #右移
        else:
            move_w = random.randint(int((right - left)*1.2), max(int((right - left)*1.21), int(space_right*0.8)))
        #上下随机移动
        move_h = random.randint(int(-space_top * 0.8), int(space_buttom*0.8))

    return move_w, move_h, min_space_bounder


def trans_image_mask(source_image, only_object_image, source_mask, segmentation):
    (h, w, _) = only_object_image.shape
    center_x, center_y = h//2, w//2
    angle = random.randint(-30, 30)
    rotate_M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    rotated_image = cv2.warpAffine(only_object_image, rotate_M, (w, h))
    rotated_mask = cv2.warpAffine(source_mask, rotate_M, (w, h))

    #对图片进行平移 移动w， 后面移动h
    move_w, move_h, min_space_bounder = get_movexy(w, h, segmentation)

    # 对旋转的图片进行放大缩小
    # 如果出现图片在边界，只能进行缩小
    if min_space_bounder/w <= 0.1 or min_space_bounder/h <= 0.1:
        rescale = random.randint(80, 100) * 0.01
    else:
        rescale = random.randint(80, 120) * 0.01

    rescaled_image = cv2.resize(rotated_image, (int(w * rescale), int(h * rescale)))
    rescaled_mask = cv2.resize(rotated_mask, (int(w * rescale), int(h * rescale)))

    move_M = np.float32([[1, 0, move_w], [0, 1, move_h]])
    move_image = cv2.warpAffine(rescaled_image, move_M, (w, h))
    move_mask = cv2.warpAffine(rescaled_mask, move_M, (w, h))

    trans_image = only_object_image + move_image
    trans_mask = source_mask + move_mask
    # #显示测试
    # plt.subplot(321)
    # plt.imshow(rotated_image)
    # plt.subplot(322)
    # plt.imshow(rotated_mask)
    # plt.subplot(323)
    # plt.imshow(move_image)
    # plt.subplot(324)
    # plt.imshow(move_mask)
    # #叠加结果
    # plt.subplot(325)
    # plt.imshow(trans_image)
    # plt.subplot(326)
    # plt.imshow(trans_mask)
    # plt.show()

    #根据生成的图片得到
    return trans_image, trans_mask

def get_cm_image_mask(source_image, segmentation):
    only_object_image = copy.deepcopy(source_image)
    h, w, _ = only_object_image.shape
    source_mask = np.zeros([h, w], np.uint8)
    # cv2.polylines(img, [pts2], True, (0,255,255))
    cv2.fillPoly(source_mask, [segmentation], 255)

    # 根据mask生成一个自由目标本身的图片
    for i, r in enumerate(source_mask):
        for j, c in enumerate(r):
            if c != 255:
                only_object_image[i][j] = [0, 0, 0]

    return trans_image_mask(source_image, only_object_image, source_mask, segmentation)

def get_new_iamge(source_image, trans_image, trans_mask):
    row, col = trans_mask.shape
    for i in range(row):
        for j in range(col):
            if trans_mask[i][j] == 255:
                source_image[i][j] = trans_image[i][j]

    return source_image

if __name__ == '__main__':
    source_image = Image.open('./dog.jpg')
    segmentation = np.array([[321.02, 321.0, 314.25, 307.99, 307.49, 293.94, 300.2, 286.14,
                              290.84, 277.81, 285.11, 276.25, 267.94, 277.81, 256.49, 279.89,
                              244.52, 281.97, 227.35, 287.18, 192.49, 290.3, 168.55, 289.26,
                              142.53, 287.18, 121.72, 293.42, 105.07, 303.83, 94.14, 313.2,
                              86.33, 326.73, 84.25, 339.22, 76.97, 343.9, 67.6, 345.46, 61.87,
                              350.66, 69.16, 360.03, 77.49, 360.03, 93.62, 358.99, 105.07, 356.91,
                              110.27, 351.7, 117.55, 353.79, 121.2, 352.74, 132.64, 361.07, 139.41,
                              367.32, 145.89, 373.77, 156.05, 374.5, 160.41, 370.14, 167.67, 367.96,
                              168.39, 370.87, 169.84, 362.88, 166.94, 356.35, 177.83, 353.45, 190.89,
                              353.45, 209.54, 358.32, 224.96, 360.09, 240.82, 361.85, 258.45, 364.49,
                              267.43, 374.29, 275.14, 377.71, 293.14, 379.43, 300.86, 370.86, 303.43,
                              358.86, 312.0, 356.29, 326.57, 361.43, 341.14, 365.71, 344.57, 369.14,
                              358.29, 370.86, 358.29, 364.0, 355.71, 360.57, 342.86, 348.57, 334.29,
                              340.0, 320.57, 322.86]], np.int32).reshape([-1, 2])

    source_image = np.array(source_image)
    trans_image, trans_mask = get_cm_image_mask(source_image, segmentation)

    plt.subplot(131)
    plt.imshow(source_image)
    plt.subplot(132)
    plt.imshow(trans_image)
    plt.subplot(133)
    plt.imshow(trans_mask)
    plt.show()
