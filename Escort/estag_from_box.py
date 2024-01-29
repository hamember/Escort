import time
import cv2
import os
import numpy as np

from EStag.detect import locate_from_box

if __name__ == '__main__':
    video_file = "./data/videos/00000000_00000000001A0011.mp4"
    det_file = os.path.join("output", os.path.basename(video_file).replace(".mp4", ".txt_det"))
    video = cv2.VideoCapture(video_file)
    dets = np.loadtxt(det_file, delimiter=",")
    pos = 2580
    video.set(cv2.CAP_PROP_POS_FRAMES, pos)
    _, img = video.read()
    dets = dets[dets[:, 0] == pos + 1]
    dets = dets[:, 2:6].astype(int)
    t0 = time.time()
    for det in dets[8:]:
        info = locate_from_box(img[int(det[1]):int(det[1] + det[3]), int(det[0]):int(det[0] + det[2])],
                               block_size=51, c=-1, visual=1, second=5)
        print(det, info[0])
    print(time.time() - t0)
