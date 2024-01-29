from collections import defaultdict
import cv2
import yaml
import numpy as np
import os

from EStag.detect import locate


class VideoProcess(object):

    def __init__(self, videoName):
        self.unique_labels = []  # 视频中检测到的二维码
        self.__positions = []  # 检测的视频帧
        self.__labels = defaultdict(list)  # 视频检测帧每个二维码id
        self.__coordinates = defaultdict(list)  # 视频检测帧每个二维码中心
        self.__orientations = defaultdict(list)  # 视频检测每个二维码的方向
        self.videoName = videoName
        self.video = cv2.VideoCapture(videoName)
        self.config = None
        self.user_setting = None
        self.video_info = {}
        self.detectAllFlag = False

    def addFrameInfo(self, position, label, coordinate, orientation):
        self.__positions.append(position)
        self.__labels[position] = label
        self.__coordinates[position] = coordinate
        self.__orientations[position] = orientation

    def detect(self, config_file='config.yaml', user_setting_file='userSetting.yaml'):
        with open("config.yaml", 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.config = config
        with open("userSetting.yaml", 'r') as f:
            user_setting = yaml.load(f, Loader=yaml.FullLoader)
            self.user_setting = user_setting
        print(config, user_setting)
        start = user_setting['start']
        end = user_setting['end']
        step = user_setting['step']
        fps = self.video.get(cv2.CAP_PROP_FPS)
        self.video_info['fps'] = fps
        frames_count = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
        self.video_info['frame_count'] = frames_count
        if end is None:
            end = int(frames_count / fps)
        if start == 0 and end == int(frames_count / fps) and step == 1:
            self.detectAllFlag = True
            count = 0
            success, frame = self.video.read()
            while success:
                print(count)
                tags, orientations, boxes = locate(frame, **config, detect_area=user_setting['detect_area'])
                self.addFrameInfo(count, tags, [np.mean(box, axis=0) for box in boxes], orientations)
                success, frame = self.video.read()
                count += 1
                if count > 900:
                    break
        else:
            start_frame = int(start * fps)
            end_frame = int(end * fps)
            for pos in range(start_frame, end_frame, step):
                print(pos)
                self.video.set(cv2.CAP_PROP_POS_FRAMES, pos)
                _, frame = self.video.read()
                tags, orientations, boxes = locate(frame, **config, detect_area=user_setting['detect_area'])
                self.addFrameInfo(pos, tags, [np.mean(box, axis=0) for box in boxes], orientations)
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def mask_video(self, fps=None):
        if fps is None:
            fps = self.video_info['fps'] / self.user_setting['step']
        width, height = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        write_video = cv2.VideoWriter('masked_' + os.path.basename(self.videoName), fourcc, fps, (width, height))
        if self.detectAllFlag:
            success, frame = self.video.read()
            count = 0
            while success:
                for i in range(len(self.__labels[count])):
                    cv2.putText(frame, str(self.__labels[count][i]),
                                org=tuple(*np.int32(self.__coordinates[count][i])), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=2, color=(255, 0, 0), thickness=3)
                write_video.write(frame)
                count += 1
                if count > 900:
                    break
                success, frame = self.video.read()
            write_video.release()
        else:
            for pos in self.__positions:
                self.video.set(cv2.CAP_PROP_POS_FRAMES, pos)
                _, frame = self.video.read()
                for i in range(len(self.__labels[pos])):
                    cv2.putText(frame, str(self.__labels[pos][i]),
                                org=tuple(*np.int32(self.__coordinates[pos][i])), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=2, color=(255, 0, 0), thickness=3)
                write_video.write(frame)
            write_video.release()


if __name__ == '__main__':
    videoProcess = VideoProcess('../data/videos/1.mp4')
    videoProcess.detect()
    videoProcess.mask_video()
    
