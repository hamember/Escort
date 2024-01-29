import time

import cv2
import numpy as np
from collections import Counter

import yaml
from glob import glob

from EStag.code16 import check


def get_coordinates(dst_points):
    src_points = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float32)
    grid = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32) / 6
    x, y = np.meshgrid(grid, grid)
    points = np.vstack((x.ravel(), y.ravel())).transpose()
    points = points[:, None]
    transform = cv2.getPerspectiveTransform(src_points, dst_points.astype(np.float32))
    transform_points = np.int32(np.round(np.squeeze(cv2.perspectiveTransform(points, transform))))
    return transform_points


def order_points(points):
    points = points.copy()
    centroid = np.mean(points, axis=0)
    directions = points - centroid
    direction = np.arctan2(directions[:, 1], directions[:, 0])
    ind_order = np.argsort(direction)
    points = points[ind_order]
    return points[:, None]


def permissive_code_tracking(im, pts):
    transform_points = get_coordinates(pts)
    p_color = [im[p[1], p[0]] for p in transform_points]  # 获取坐标坐标的颜色
    p_color = np.array(p_color, dtype=np.float).reshape(4, 4)
    p_color = (p_color - np.min(p_color)) / (np.max(p_color) - np.min(p_color))
    threshes = [0.5]
    code_data = []
    for thresh in threshes:
        code = p_color > thresh
        code = code.astype(np.int32)
        num, ori = check(code)
        if num > 0:
            code_data.append([num, ori])
    if code_data:
        nums = [num for num, _ in code_data]
        stat = Counter(nums)
        common_num = stat.most_common(1)[0][0]
        ind = nums.index(common_num)
        ori = code_data[ind]
        return num, ori
    return -1, -1


def code_tracking(dst_points, im):
    transform_points = get_coordinates(dst_points)
    code = [im[p[1], p[0]] for p in transform_points]  # 对应点的编码
    # code = [np.mean(im[p[1]-1:p[1] + 2, p[0]-1:p[0]+2]) > 0.5 for p in transform_points]  # 对应点的局部平均编码
    code = np.array(code).reshape(4, 4)
    tag, ori = check(code)
    return tag, ori, transform_points, code


def locate(im, thresh_method=1, block_size=15, c=3, thresh=None, mean_area=500, valid_tags=None, visual=0,
           config_file=None, detect_area=None, second=1):
    if config_file is not None:
        with open(config_file, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            thresh_method, block_size, c = params['thresh_method'], params['block_size'], params['c']
            mean_area, valid_tags = params['mean_area'], params['valid_tags']

    area_thresh = [0.2 * mean_area, 5 * mean_area]
    if detect_area:
        original_im = im
        im = im[detect_area[1]:detect_area[3], detect_area[0]:detect_area[2]]

    if im.ndim == 3:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    else:
        gray = im
    if thresh is not None:
        _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    else:
        if thresh_method == 1:  # 23, -3
            bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c)
        elif thresh_method == 2:  # 13, -4
            bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c)
        elif thresh_method == 3:
            _, bw = cv2.threshold(gray, -1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.namedWindow("bumblebee", cv2.WINDOW_NORMAL)
    # cv2.imshow('bumblebee', bw)
    # cv2.imwrite("binary.jpg", bw)
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area_thresh[0] < area < area_thresh[1]:
            rect = cv2.minAreaRect(contour)  # 获得最小外包矩形（中心（x,y），(宽，高），旋转角度）
            if 0.75 < rect[1][0] / rect[1][1] < 1.25 and area / np.prod(rect[1]) > 0.8:
                valid_contours.append(contour)
    new_bw = np.zeros_like(bw)
    boxes = []
    tags = []
    not_pass_boxes = []
    orientations = []
    # 形态学操作， 去除毛刺平滑边缘
    # cv2.fillPoly(new_bw, valid_contours, color=1)  # 对每个连通域进行填充
    cv2.drawContours(new_bw, valid_contours, -1, 1, thickness=-1)
    new_bw = cv2.morphologyEx(new_bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(new_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        # Douglas-Peucker algorithm 多边形近似
        epsilon = np.sqrt(area) / 5
        box = cv2.approxPolyDP(contour, epsilon, True)

        if len(box) != 4:
            continue
        tag, orientation, points_16, code = code_tracking(box, bw // 255)
        # tag, orientation = permissive_code_tracking(gray, box)

        if tag < 0:
            not_pass_boxes.append(box)
            continue
        if valid_tags is not None:
            if tag in valid_tags:
                tags.append(tag)
                orientations.append(orientation)
                boxes.append(box)
        else:
            tags.append(tag)
            orientations.append(orientation)
            boxes.append(box)
            # portion_visual(bw, box, points_16)
    if detect_area:
        if len(boxes) > 0:
            boxes += np.array(detect_area[:2])
        if len(not_pass_boxes) > 0:
            not_pass_boxes += np.array(detect_area[:2])
        im = original_im
    if visual == 1:
        visualization(im, tags, orientations, boxes, not_pass_boxes, second)

    for i, box in enumerate(boxes):
        boxes[i] = box.tolist()

    return tags, orientations, boxes


def locate_from_box(im, thresh_method=1, block_size=15, c=3, thresh=None, mean_area=500, valid_tags=None, visual=0,
                    second=1):
    global bw
    area_thresh = [0.75 * mean_area, 1.25 * mean_area]

    if im.ndim == 3:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    else:
        gray = im
    if thresh is not None:
        _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    else:
        if thresh_method == 1:  # 23, -3
            bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c)
        elif thresh_method == 2:  # 13, -4
            bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c)
        elif thresh_method == 3:
            _, bw = cv2.threshold(gray, -1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area_thresh[0] < area < area_thresh[1]:
            rect = cv2.minAreaRect(contour)  # 获得最小外包矩形（中心（x,y），(宽，高），旋转角度）
            if 0.75 < rect[1][0] / rect[1][1] < 1.25 and area / np.prod(rect[1]) > 0.8:
                valid_contours.append(contour)
    new_bw = np.zeros_like(bw)
    boxes = []
    tags = []
    not_pass_boxes = []
    not_pass_codes = []
    orientations = []
    # 形态学操作， 去除毛刺平滑边缘
    # cv2.fillPoly(new_bw, valid_contours, color=1)  # 对每个连通域进行填充
    cv2.drawContours(new_bw, valid_contours, -1, 1, thickness=-1)
    new_bw = cv2.morphologyEx(new_bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(new_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        # Douglas-Peucker algorithm 多边形近似
        epsilon = np.sqrt(area) / 5
        box = cv2.approxPolyDP(contour, epsilon, True)

        if len(box) != 4:
            continue
        tag, orientation, points_16, code = code_tracking(box, bw // 255)
        # tag, orientation = permissive_code_tracking(gray, box)

        if tag < 0:
            not_pass_boxes.append(box)
            not_pass_codes.append(code)
            continue
        if valid_tags is not None:
            if tag in valid_tags:
                tags.append(tag)
                orientations.append(orientation)
                boxes.append(box)
            else:
                not_pass_boxes.append(box)
                not_pass_codes.append(code)
        else:
            tags.append(tag)
            orientations.append(orientation)
            boxes.append(box)
            # portion_visual(bw, box, points_16)

    if visual == 1:
        visualization(im, tags, orientations, boxes, not_pass_boxes, second)

    if len(tags) > 0:
        return tags, orientations, boxes
    else:
        if len(not_pass_boxes) > 0:
            return not_pass_codes, not_pass_boxes

    return tags, orientations, boxes


def portion_visual(im, box, points_16_all):
    box = np.asarray(box)
    box = np.squeeze(box)
    x1, y1 = np.min(box, axis=0)
    x2, y2 = np.max(box, axis=0)
    portion_im = im[int(y1) - 5:int(y2) + 5, int(x1) - 5:int(x2) + 5]
    portion_im = cv2.resize(portion_im, dsize=(0, 0), fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
    portion_im = cv2.morphologyEx(portion_im, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    if portion_im.ndim == 2:
        portion_im = cv2.cvtColor(portion_im, cv2.COLOR_GRAY2BGR)
    for i in range(len(box)):
        color = (0, 255, 0)
        cv2.circle(portion_im, tuple(np.int32((box[i] - np.array([x1 - 5, y1 - 5])) * 5)), 8, color, thickness=3)
    for i in range(len(points_16_all)):
        color = (0, 0, 255)
        cv2.circle(portion_im, tuple(np.int32((points_16_all[i] - np.array([x1 - 5, y1 - 5])) * 5)), 5, color,
                   thickness=-1)
    cv2.imshow('portion_bin', portion_im)
    cv2.waitKey(0)
    cv2.imwrite('portion.jpg', portion_im)


def visualization(im, tags, orientations, boxes, not_pass_boxes, second=1):
    for i, box in enumerate(boxes):
        for j, point in enumerate(box):
            color = (0, 255, 0)
            cv2.circle(im, tuple(np.int32(point[0])), 5, color, thickness=2)
            cv2.imwrite('points.jpg', im)
        cv2.putText(im, str(tags[i]), tuple(np.int32(point[0])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    for box in not_pass_boxes:
        for point in box:
            cv2.circle(im, tuple(np.int32(point[0])), 5, (255, 0, 0), thickness=-1)
    cv2.imshow('bumblebeeImage', im)
    cv2.waitKey(second*1000)


if __name__ == '__main__':
    folder = '../data//images//*.jpg'
    for file in glob(folder):
        img = cv2.imread(file)
        t0 = time.time()
        tags, _, _ = locate(img, block_size=65, c=-7, visual=0)
        print(tags)
        # print(time.time() - t0)
