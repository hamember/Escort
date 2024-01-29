from collections import defaultdict
import yaml
import cv2
import numpy as np
import os
from detect import locate
import time
import pandas as pd
import sqlite3
import sys


class VideoProcess(object):

    def __init__(self, videoName):
        self.unique_labels = []  # 视频中检测到的二维码
        self.positions = []  # 检测的视频帧
        self.labels = defaultdict(list)  # 视频检测帧每个二维码id
        self.coordinates = defaultdict(list)  # 视频检测帧每个二维码中心
        self.orientations = defaultdict(list)  # 视频检测每个二维码的方向
        self.videoName = videoName
        self.video = cv2.VideoCapture(videoName)
        self.config = None
        self.user_setting = None
        self.video_info = {}
        self.detectAllFlag = False

    def addFrameInfo(self, position, label, coordinate, orientation):
        self.positions.append(position)
        self.labels[position] = label
        self.coordinates[position] = coordinate
        self.orientations[position] = orientation

    def getFrameInfo(self, position):
        return self.labels[position], self.coordinates[position], self.orientations[position]

    def detect(self, save, config_file='config.yaml', user_setting_file='userSetting.yaml'):
        with open("config.yaml", 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.config = config
        with open("userSetting.yaml", 'r') as f:
            user_setting = yaml.load(f, Loader=yaml.FullLoader)
            self.user_setting = user_setting
        start = user_setting['start']
        end = user_setting['end']
        step = user_setting['step']
        fps = self.video.get(cv2.CAP_PROP_FPS)
        self.video_info['fps'] = fps
        frames_count = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
        self.video_info['frame_count'] = frames_count
        print(self.video_info)
        process_time = []
        if end is None:
            end = int(frames_count / fps)

        if save:
            if fps is None:
                fps = self.video_info['fps'] / self.user_setting['step']
            width, height = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # track_file = os.path.join('output', '%s.txt' % video_name + '_track')
            # vid_writer_path = os.path.join('output', os.path.basename(video_path))
            fourcc = 'mp4v'  # output video codec
            write_video = cv2.VideoWriter(save, cv2.VideoWriter_fourcc(*fourcc), fps/step, (width, height))

        if start == 0 and end == int(frames_count / fps) and step == 1:
            self.detectAllFlag = True
            count = 0
            success, frame = self.video.read()
            while success:
                print(count)
                start_time = time.time()
                tags, orientations, boxes = locate(frame, **config, detect_area=user_setting['detect_area'])
                cycle_time = time.time() - start_time
                print(cycle_time)
                process_time.append(cycle_time)
                self.addFrameInfo(count, tags, [np.mean(box, axis=0) for box in boxes], orientations)
                for i in range(len(self.labels[count])):
                    cv2.putText(frame, str(self.labels[count][i]),
                                org=tuple(*np.int32(self.coordinates[count][i])), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=2, color=(255, 0, 0), thickness=3)
                if save:
                    write_video.write(frame)
                success, frame = self.video.read()
                count += 1
            if save:
                write_video.release()
        else:
            start_frame = int(start * fps)
            end_frame = int(end * fps)
            count = 0
            for pos in range(start_frame, end_frame, step):
                print(pos)
                print(count)
                self.video.set(cv2.CAP_PROP_POS_FRAMES, pos)
                _, frame = self.video.read()
                start_time = time.time()
                tags, orientations, boxes = locate(frame, **config, detect_area=user_setting['detect_area'])
                cycle_time = time.time() - start_time
                print(cycle_time)
                process_time.append(cycle_time)
                self.addFrameInfo(count, tags, [np.mean(box, axis=0) for box in boxes], orientations)
                for i in range(len(self.labels[count])):
                    cv2.putText(frame, str(self.labels[count][i]),
                                org=tuple(*np.int32(self.coordinates[count][i])), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=2, color=(255, 0, 0), thickness=3)
                if save:
                    write_video.write(frame)
                count += 1
            if save:
                write_video.release()
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return process_time

    def mask_video(self, fps=None):
        if fps is None:
            fps = self.video_info['fps'] / self.user_setting['step']
        width, height = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vid_writer_path = os.path.join('data', os.path.basename(self.videoName))
        write_video = cv2.VideoWriter(vid_writer_path, fourcc, fps, (width, height))
        if self.detectAllFlag:
            success, frame = self.video.read()
            count = 0
            while success:
                print(count)
                for i in range(len(self.labels[count])):
                    cv2.putText(frame, str(self.labels[count][i]),
                                org=tuple(*np.int32(self.coordinates[count][i])), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=2, color=(255, 0, 0), thickness=3)
                write_video.write(frame)
                count += 1
                success, frame = self.video.read()
            write_video.release()
        else:
            for pos in self.positions:
                print(pos)
                self.video.set(cv2.CAP_PROP_POS_FRAMES, pos)
                _, frame = self.video.read()
                for i in range(len(self.labels[pos])):
                    cv2.putText(frame, str(self.labels[pos][i]),
                                org=tuple(*np.int32(self.coordinates[pos][i])), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=2, color=(255, 0, 0), thickness=3)
                write_video.write(frame)
            write_video.release()


    def save_to_database(self,video):
        detect_info = self.toDataFrameFormat()
        save_path = video + '._db'
        if os.path.exists(save_path):
            os.remove(save_path)
        con = sqlite3.connect(save_path)
        for number, data in detect_info.items():
            pd.io.sql.to_sql(data, 'tag' + str(number), con=con, index=False, if_exists='replace')
        con.close()
        return save_path


    def toDataFrameFormat(self):
        id_time = []
        fps = self.video_info['fps'] / self.user_setting['step'] 
        pos_frames = list(range(len(self.labels)))
        for pos in pos_frames:
            id_time.append(pd.Timestamp(int(pos) / fps, unit='s').time())
            # id_time.append(pd.Timestamp(pos/fps, unit='s').time().strftime('%H:M:%S.%f'))
        base_id = pd.DataFrame({'time': id_time, 'frame': pos_frames})
        tags_pos = defaultdict(list)
        tags = defaultdict(list)
        detect_info = {}
        for pos in pos_frames:
            for i, label in enumerate(self.labels[pos]):
                tags_pos[label].append(pos)
                tags[label].append(self.coordinates[pos][i])

        print(tags.keys())
        for label in tags.keys():
            tag = np.squeeze(np.asarray(tags[label]))
            if tag.ndim != 2:
                continue
            info_df = pd.DataFrame(tag, columns=['x', 'y'])
            keys = pd.Series(tags_pos[label], name='frame', index=info_df.index)
            info_df = pd.concat([keys, info_df], axis=1)
            df = pd.merge(base_id, info_df, how='left', )
            # df = df.fillna(-1)
            # print(df)
            detect_info[label] = df

        return detect_info

if __name__ == '__main__':
    #
    # file = r'E:\bee\Escort\data\test.mp4'
    # videoProcess = VideoProcess(file)
    # videoProcess.detect(save='test0124_2.mp4')
    # 默认参数
    detect=None
    otname=None
    # 参数读取
    for i in range(len(sys.argv)):
        if '--video' == sys.argv[i]:
            file = sys.argv[i+1]
        elif '-V' == sys.argv[i]:
            file = sys.argv[i+1]
        elif '--detect' == sys.argv[i]:
            detect = sys.argv[i+1]
        elif '-D' == sys.argv[i]:
            detect = sys.argv[i+1]
        elif '--otname' == sys.argv[i]:
            otname = sys.argv[i+1]
        elif '-O' == sys.argv[i]:
            otname = sys.argv[i+1]

    videoProcess = VideoProcess(file)

    if detect:
        videoProcess.detect(save=detect)
    else:
        videoProcess.detect(False)
    if otname:
        videoProcess.save_to_database(otname)
    else:
        pass


    # videoProcess.mask_video()
    # frame_num = list(range(len(videoProcess.labels)))
    # save_to_database('test1_estag_long_30',videoProcess.video_info["fps"]/30, frame_num, videoProcess.labels, videoProcess.coordinates)