import numpy as np
import cv2
import os
from EStag.code16 import num2code


def polygon_iou(poly1, poly2):
    min_xy = np.min(np.vstack((poly1, poly2)), axis=0)
    max_xy = np.max(np.vstack((poly1, poly2)), axis=0)
    w, h = np.squeeze(max_xy - min_xy + 1)
    poly1 = poly1 - min_xy
    poly2 = poly2 - min_xy
    im1 = np.zeros((h, w), dtype=np.uint8)
    im2 = np.zeros((h, w), dtype=np.uint8)
    filled_im1 = cv2.fillPoly(im1, [poly1], 255)
    filled_im2 = cv2.fillPoly(im2, [poly2], 255)
    or_area = cv2.bitwise_or(filled_im1, filled_im2)
    and_area = cv2.bitwise_and(filled_im1, filled_im2)
    iou = np.sum(np.float32(np.greater(and_area, 0))) / np.sum(np.float32(np.greater(or_area, 0)))
    return iou


def code_match1(raw_code, number):
    real_code = num2code(number)
    match = np.zeros(4)
    for i in range(4):
        variant_code = np.rot90(raw_code, i + 1)
        match[i] = np.sum(variant_code == real_code) / np.prod(raw_code.shape)
    return np.max(match), np.argmax(match)


def code_match2(raw_code, code):
    match = np.zeros(4)
    for i in range(4):
        variant_code = np.rot90(raw_code, i + 1)
        match[i] = np.sum(variant_code == code)
    return np.max(match), np.argmax(match)


def mean_filter(im, k_size, border_type):
    """
    均值滤波的一种实现方法，利用图像的积分，减少计算量，速度更快。可惜，cv2.boxFilter 速度比这快的多。
    :param im:灰度图像
    :param k_size:核大小
    :param border_type:边界填充的方式
    :return: im_I:均值滤波后的图像
    """
    h, w = im.shape
    im = cv2.copyMakeBorder(im, *[int((k_size + 1) / 2), int(k_size / 2)] * 2, border_type)
    im_I = np.cumsum(np.cumsum(im, axis=0), axis=1)  # 累加
    im_I = im_I[k_size:, k_size:] + im_I[:h, :w] - im_I[k_size:, :w] - im_I[:h, k_size:]
    im_I = im_I / k_size ** 2
    return np.round(im_I).astype(np.uint8)


def bradley(im, k_size, t, border_type=None):
    """
    一种图像二值化方法，如果像素点值低于领域平局值的T%,为黑0，否者为白1
    :param im: 单通道灰度图像
    :param k_size: 窗口大小，元组类型
    :param t: 百分比
     :param border_type: 边界扩展方式
    :return:
    """
    mean = cv2.boxFilter(im, cv2.CV_64F, k_size, borderType=border_type)
    output = np.zeros(im.shape, dtype=np.uint8)
    output[im <= mean * (1 - t / 100)] = 0
    return


def frameFromVideo(videoName, imageCount=10, save_folder=None):
    video = cv2.VideoCapture(videoName)
    frameCount = video.get(cv2.CAP_PROP_FRAME_COUNT)
    position = np.linspace(0, frameCount - 1, num=imageCount, dtype=int)
    if save_folder is None:
        save_folder = os.path.join('./bumblebeeImage', os.path.splitext(os.path.basename(videoName))[0])
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for p in position:
        video.set(cv2.CAP_PROP_POS_FRAMES, p)
        _, frame = video.read()
        cv2.imwrite(os.path.join(save_folder, f"{p:06d}.jpg"), frame)


if __name__ == '__main__':
    filename = '../data//videos//00000000_00000000001A0011.mp4'
    if os.path.exists(filename):
        frameFromVideo(filename, save_folder="../data//images")
    else:
        print("file not found")
