import argparse
import time
from glob import glob

import numpy as np
import os
import cv2
import yaml
from numpy import random
import torch
from filterpy.kalman import KalmanFilter

from EStag.detect import locate_from_box
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

import matplotlib
from EStag.code16 import num2code
from EStag.tool import code_match2

matplotlib.use('TkAgg')
num_to_code = {}


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return o


def tag_similarity(EStag_infos, trks_id):
    similarity = np.zeros((len(EStag_infos), len(trks_id)))
    for j in range(len(trks_id)):
        trk_id = trks_id[j]
        if trk_id < 0:
            continue
        for i in range(len(EStag_infos)):
            EStag_info = EStag_infos[i]
            if trk_id not in num_to_code.keys():
                num_to_code[trk_id] = num2code(trk_id)
            if len(EStag_info) == 3:
                if len(EStag_info[0]) > 1:
                    continue
                if EStag_info[0] and EStag_info[0][0] == trk_id:
                    similarity[i, j] = 1
            else:
                if len(EStag_info[0]) > 1:
                    continue
                s, ind = code_match2(num_to_code[trk_id], EStag_info[0][0])
                if s == 15:
                    similarity[i, j] = 0.5
                elif s == 14:
                    similarity[i, j] = 0.1

    return similarity


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox, id_):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        KalmanBoxTracker.count += 1
        if id_ is None:
            self.id = - KalmanBoxTracker.count
        else:
            self.id = id_
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox, id_):
        """
        Updates the state vector with observed bbox.
        """
        if id_ is not None:
            self.id = id_
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, EStag_infos, trks_id, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    w1, w2 = 0.4, 0.6

    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = iou_batch(detections, trackers)
    tag_matrix = tag_similarity(EStag_infos, trks_id)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32) * iou_matrix
        matrix = w1 * a + w2 * tag_matrix
        print(matrix)
        matched_indices = linear_assignment(-matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    print(matches, unmatched_detections, unmatched_trackers)
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.trackers_id = []
        self.convert_id = {}

    def update(self, image, dets=np.empty((0, 5))):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        trks_id = []
        for i, t in enumerate(self.trackers):
            if i not in to_del:
                trks_id.append(self.trackers[i].id)
        EStag_infos = []
        tags = []
        for d in det:
            EStag_info = locate_from_box(img[int(d[1]):int(d[3]), int(d[0]):int(d[2])], block_size=65, c=-7,
                                         valid_tags=[75, 78, 80, 86, 89, 90, 95, 97, 100, 103])
            if len(EStag_info[0]) != 1:
                EStag_info = [], [], []
            if len(EStag_info) == 3 and EStag_info[0]:
                tags.append(EStag_info[0][0])
            else:
                tags.append(0)
            EStag_infos.append(EStag_info)
        print(tags)
        print(trks_id)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, EStag_infos, trks_id,
                                                                                   self.iou_threshold)

        # update matched trackers with assigned detections
        m_det = [m[0] for m in matched]
        m_trk = [m[1] for m in matched]
        for m in matched:
            if tags[m[0]] > 0 and sum([tags[m[0]] == tags[i] for i in m_det]) == 1:
                self.convert_id[self.trackers[m[1]].id] = tags[m[0]]
                self.trackers[m[1]].update(dets[m[0], :], tags[m[0]])
            else:
                self.trackers[m[1]].update(dets[m[0], :], None)

        # create and initialise new trackers for unmatched detections
        trks_id = []
        for i in range(len(self.trackers)):
            trks_id.append(self.trackers[i].id)
        for i in unmatched_dets:
            if tags[i] > 0 and tags[i] not in trks_id and sum([tags[i] == tag for tag in tags]) == 1:
                trk = KalmanBoxTracker(dets[i, :], tags[i])
            else:
                trk = KalmanBoxTracker(dets[i, :], None)
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))


class Detector:
    def __init__(self, weights, device, conf_thres, iou_thres, agnostic_nms, img_size):
        # Initialize
        set_logging()
        self.device = select_device(device)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.agnostic_nms = agnostic_nms
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        if self.half:
            self.model.half()  # to FP16

        self.img_size = check_img_size(img_size, s=self.model.stride.max())  # check img_size

    def step(self, img0):
        img = letterbox(img0, new_shape=self.img_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0 - 1
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = self.model(img)[0]
        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, agnostic=self.agnostic_nms)
        t2 = time_synchronized()
        assert len(pred) == 1, 'batch size should be 1'
        det = pred[0]
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
        return det.cpu().numpy()


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',
                        action='store_true')
    parser.add_argument("--max_age",
                        help="Maximum number of frames to keep alive a track without associated detections.",
                        type=int, default=3)
    parser.add_argument("--min_hits",
                        help="Minimum number of associated detections before track is initialised.",
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    parser.add_argument('--weights', nargs='+', type=str, default='weights/bumblebee/last.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/videos', help='source')  # file/folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.75, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument("--start_frame", type=int, default=0, help="frame position of starting tracking")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    display = args.display
    # args.display = True
    total_time = 0.0
    total_frames = 0
    colours = np.random.rand(32, 3)  # used only for display
    if not os.path.exists('output'):
        os.makedirs('output')

    detector = Detector(args.weights, args.device, args.conf_thres, args.iou_thres, args.agnostic_nms, args.img_size)

    for video in glob(os.path.join(args.source, '00*.*')):
        print('Processing %s:' % video, end='')
        frame = args.start_frame
        video_name = os.path.basename(video).split('.')[0]
        vid_cap = cv2.VideoCapture(video)
        n_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        detections = np.empty((0, 10))
        results = np.empty((0, 10))
        mot_tracker = Sort(max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.iou_threshold)
        suc, img = vid_cap.read()
        while suc:
            det = detector.step(img)
            total_frames += 1
            frame += 1
            print('processing frame %s' % str(frame))
            start_time = time.time()
            trackers = mot_tracker.update(img, det)
            print(det)
            print(np.flipud(trackers))
            cycle_time = time.time() - start_time
            total_time += cycle_time

            suc, img = vid_cap.read()
            if frame > args.start_frame + 20:
                break
