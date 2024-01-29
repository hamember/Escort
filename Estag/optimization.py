import cv2
import os
import yaml
import numpy as np
import math
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
from detect import locate

# function to plot heatmap
def show_stats(stats, size_range, thresh_range):
    sns.heatmap(stats, annot=True, fmt='.0f', yticklabels=[str(num) for num in range(*size_range)],
                xticklabels=[str(num) for num in range(*thresh_range)])
    plt.xlabel('c')
    plt.ylabel('kernel')
    plt.savefig("optimize.svg")
    plt.show()


def optimize_track_parameters(video_file, method=1, size_range=(11, 91, 2), thresh_range=(-11, 5),
                              frames_count=1, valid_tags=None):
    video = cv2.VideoCapture(video_file)
    n_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    frames = np.linspace(0, n_frames - 1, frames_count, dtype=np.int32)
    stats = np.zeros((frames_count, len(range(*size_range)), len(range(*thresh_range))))
    all_boxes = []
    for i, ind in enumerate(tqdm(frames, desc='optimize process...')):
        video.set(cv2.CAP_PROP_POS_FRAMES, ind)
        _, frame = video.read()
        for j, size in enumerate(range(*size_range)):
            for k, thresh in enumerate(range(*thresh_range)):
                tags, orientations, boxes = locate(frame, method, size, thresh, valid_tags=valid_tags)
                all_boxes.extend(boxes)
                stats[i, j, k] = len(tags)

    stats = np.sum(stats, axis=0)
    # 可视化优化结果
    show_stats(stats, size_range, thresh_range)

    loc = cv2.minMaxLoc(stats)[3]  # 最好的结果所在位置
    best_param = loc[0] + thresh_range[0], loc[1] * size_range[2] + size_range[0]  # 最好结果位置对应的参数值

    areas = []
    mean_area = None
    for box in all_boxes:
        box = np.asarray(box)
        areas.append(cv2.contourArea(box))
    if areas:  # 可能检测结果为空
        areas = np.sort(areas)  # 对所有检测的标记面积进行排序
        n = len(areas)
        mean_area = np.mean(areas)  # 求取标记的平均面积

    # 保存结果到文件中
    np.savez(os.path.basename(video_file).replace('.mp4', '.npz'), stats=stats, areas=areas)
    optimizer = {'block_size': best_param[1], 'C': best_param[0], 'area': int(mean_area)}
    save_filename = os.path.basename(video_file).replace('.bmp', '.yaml')
    with open(save_filename, 'w') as f:
        yaml.dump(optimizer, f)
    print(optimizer)
    return best_param, mean_area, optimizer


def optimize_thresh(video_file, method=1, thresh_range=(-9, 9), frames_count=50, valid_tags=None,
                    save_filename="config.yaml"):
    '''
    block_size设置为二维码平均边长的一半，枚举thresh调优
    :param save_filename:
    :param video_file:
    :param method:
    :param thresh_range:
    :param frames_count:
    :param valid_tags:
    :param save_file:
    :return:
    '''
    video = cv2.VideoCapture(video_file)
    n_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    frames = np.linspace(0, n_frames - 1, frames_count, dtype=int)
    all_boxes = []
    for i, ind in enumerate(tqdm(frames, desc='optimize block_size process...')):
        video.set(cv2.CAP_PROP_POS_FRAMES, ind)
        _, frame = video.read()
        tags, orientations, boxes = locate(frame, method, 15, -3, valid_tags=valid_tags)
        all_boxes.extend(boxes)

    areas = []
    mean_area = None
    for box in all_boxes:
        areas.append(cv2.contourArea(np.asarray(box)))
    if areas:  # 可能检测结果为空
        areas = np.sort(areas)  # 对所有检测的标记面积进行排序
        n = len(areas)
        mean_area = np.mean(areas[int(0.1*n):int(0.9*n)])  # 求取标记的平均面, 去掉端点可能存在的异常值
    else:
        print("初始的参数大小不合适，无法检测出二维码")
        return
    block_size = int(math.sqrt(mean_area)*2/3)
    block_size = block_size if block_size % 2 == 1 else block_size+1

    stats = np.ones(len(range(*thresh_range)), dtype=int)
    for i, ind in enumerate(tqdm(frames, desc='optimize c process...')):
        video.set(cv2.CAP_PROP_POS_FRAMES, ind)
        _, frame = video.read()
        for j, value in enumerate(range(*thresh_range)):
            tags, orientations, boxes = locate(frame, method, block_size, value, valid_tags=valid_tags)
            stats[j] += len(tags)
    c = list(range(*thresh_range))[np.argmax(stats)]

    optimizer = {'thresh_method': method, 'block_size': block_size, 'c': c, 'mean_area': int(mean_area),
                 'valid_tags': valid_tags}
    with open(save_filename, 'w') as f:
        yaml.dump(optimizer, f)
    return block_size, c


if __name__ == '__main__':
    filename = r'L:\0220_WT\pm\WT4_0220_2030\00000251.tiff'
    optimize_track_parameters(filename, size_range=(23, 43, 2), thresh_range=(1, 15))

