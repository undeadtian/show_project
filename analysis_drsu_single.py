# -*- coding: utf-8 -*-
"""
@Project : test_code
@File    : analysis_drsu_single.py
@Author  : 王白熊
@Data    ： 2020/11/10 16:45
单个drsutrack——id分析
"""
import pandas as pd
import numpy as np
from pandas import Series
from scipy import optimize
from Log import Logger
from constant import const
import math
import random
import glob
import os
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

logger = Logger('TrackDrsu').getlog()

FRAME_LOSS = 10
TRACK_ID_MIN = 11126
DISTANCE_MAX = 220

def target_func(x, A, B):
    return A * x + B


class TrackDrsu(object):
    def __init__(self, file_drsu, ort=True):
        self.df = round(pd.read_csv(file_drsu), 1)
        self.track_id = int(os.path.basename(file_drsu).split('_')[1].strip('.csv'))
        self.ort = ort
        self.frame_num = self.df.shape[0]
        # 障碍物类型可能出现跳变,所以取众数
        self.obj_type = file_drsu.split('_')[1]
        self.dict_track_info = pd.Series()
        self._file_name = file_drsu
        # 判断轨迹类型，采用方差法判断轨迹是否是静止, 返回值0：代表是静止状态，1代表是直线运动状态

    # 单个track_id画轨迹图
    def draw_track_straight(self, ax):
        ax.plot(self.df['center_x'], self.df['center_y'], random.choice(const.LIST_COLOR),
                linewidth=0.4, label='track_id:{}'.format(self.track_id))

    # 基础的参数用这个获取，其他的调用函数获取
    def calc_track_info(self):
        track_info = self.dict_track_info
        track_info['track_id'] = self.track_id
        track_info['obj_type'] = self.obj_type
        track_info['volume'] = self.df['width'].mean() * self.df['height'].mean()
        start_time = int(round(self.df.timestamp[0] * 10))
        end_time = int(round(self.df.timestamp[self.df.shape[0] - 1] * 10))
        track_info['time_stamp'] = end_time - start_time
        track_info['time_start'] = start_time
        track_info['time_end'] = end_time
        # 坐标参数一般是用在静态场景下，所以取总数会比较
        track_info['center_x_start'] = self.df.center_x[0]
        track_info['center_y_start'] = self.df.center_y[0]
        track_info['center_x_end'] = self.df.center_x[self.df.shape[0] - 1]
        track_info['center_y_end'] = self.df.center_y[self.df.shape[0] - 1]
        track_info['add_x'] = self.df.center_x[self.df.shape[0] - 1] - self.df.center_x[0]
        track_info['add_y'] = self.df.center_y[self.df.shape[0] - 1] - self.df.center_y[0]
        track_info['speed_x'] = track_info['add_x']/track_info['time_stamp']
        track_info['speed_y'] = track_info['add_y']/track_info['time_stamp']
        track_info['frame_num'] = self.df.shape[0]
        logger.debug('单个处理结果：%s' % track_info)
        return track_info

    @property
    def track_info(self):
        if not self.dict_track_info.empty:
            return self.dict_track_info
        else:
            return self.calc_track_info()

    def check_type(self,ord=False, type=True):
        if self.df.shape[0] < FRAME_LOSS:
            logger.info('track_id:{}帧数过小，丢弃'.format(self.track_id))
            return 0
        elif self.track_id < TRACK_ID_MIN:
            logger.info('track_id:{}小于最早trackid{}，丢弃'.format(self.track_id, TRACK_ID_MIN))
            return 1
        elif type and self.obj_type not in [6,7,8]:
            logger.info('track_id:{}障碍物类型{}，丢弃'.format(self.track_id,self.obj_type))
            return 2
        elif ord and self.track_info['add_y'] > 0:
            logger.info('track_id:{}是对向车道车辆，丢弃'.format(self.track_id))
            return 3
        elif abs(self.track_info['add_y']) + abs(self.track_info['add_x']) > DISTANCE_MAX:
            logger.info('track_id{} 轨迹移动距离大于{}米，判断为完整轨迹'.format(self.track_id,DISTANCE_MAX))
            return 4
        elif self.track_info['center_y_start']>3344690 or self.track_info['center_y_start'] < 235240:
            return 5
        elif self.track_info['center_y_end'] < 3344460:
            return 6
        else:
            return 7


def wth_test1():
    file_dir = r'D:\data\drsu_staright\group1\speed20_uniform_03\obs_data_trackid\98.csv'
    track = TrackDrsu(file_dir)
    print(track.track_info)


if __name__ == '__main__':
    wth_test1()
