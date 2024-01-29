import cv2
from os import path
import time

if __name__ == '__main__':
    video_file = "output/00000000_00000000001A0011.mp4"
    assert path.exists(video_file), "video folder not find"
    pos = 2579
    video = cv2.VideoCapture(video_file)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.set(cv2.CAP_PROP_POS_FRAMES, pos)
    ret, frame = video.read()
    cv2.namedWindow('bumblebee', 0)
    t0 = time.time()
    while ret:
        print(pos)
        cv2.imshow('bumblebee', frame)
        a = cv2.waitKey(0)
        if a == 81:
            if pos != 0:
                pos -= 1
                video.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, frame = video.read()
        elif a == 83:
            pos += 1
            ret, frame = video.read()