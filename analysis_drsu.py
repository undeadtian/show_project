# -*- coding: utf-8 -*-
"""
@Project : test_code
@File    : analysis_drsu.py
@Author  : 王白熊
@Data    ： 2020/11/10 16:36
"""
import os
import time
import math
import pandas as pd
from pandas import DataFrame
import numpy as np
from glob import glob
from Log import Logger
from constant import const
from analysis_drsu_single import TrackDrsu


logger = Logger('DrsuScene').getlog()


# drsu 场景，由多个track_id组成
class DrsuScene(object):
    def __init__(self, file_path, ort=True, use_time=False):
        """
        :param file_path: drsu路径，到obs_data_trackid这一层
        :param ort:摄像机朝向是否为x方向
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError('drsu数据文件夹:%s不存在' % file_path)
        self.data_path = file_path
        self.df = DataFrame()
        self.bk_df = DataFrame()
        # self.match_type = const.MATCH_TYPE_NOT
        # 分别记录不同类别的匹配程度 0为完全匹配。下标index分别为0：障碍物类型
        self.match_list = [0] * 10
        self.track_num = 0
        self.track_num_loss = 0
        self.rq = time.strftime('%Y%m%d', time.localtime(time.time()))
        logger.info('对文件夹{}进行分析'.format(file_path))

    # 获取场景下所有trackid的特征DataFrame
    def get_drsu_data(self, ):
        files = glob(os.path.join(self.data_path, '*.csv'))
        if not files:
            logger.error('文件夹：%s中不存在csv文件' % self.data_path)
            exit(0)
        for i in files:
            track = TrackDrsu(i)
            # 遍历获取track_id的特征信息,并全部放入一个DataFrame中
            track_info = track.track_info
            # track_info['file_name'] = i
            self.df = self.df.append(track_info, ignore_index=True)
            self.track_num += 1
        self.bk_df = self.df.sort_index(axis=1)
        self.bk_df.to_csv('parsed.csv', index=False)

# trackid１前数据
def fuse(trackid1, trackid2):
    dir_name = 'data/drsu_data/traffic_report_obstacle_2d_drsu_split'
    df1 = pd.read_csv(os.path.join(dir_name, trackid1))
    df2 = pd.read_csv(os.path.join(dir_name, trackid2))
    len_1 = df1.shape[0]
    len_2 = df2.shape[0]
    add_x1 = df1.loc[len_1 - 1, 'center_x'] - df1.loc[0,'center_x']
    add_y1 = df1.loc[len_1 - 1, 'center_y'] - df1.loc[0, 'center_y']
    frame1 = df1.loc[len_1 - 1, 'frame_number', ] - df1.loc[0, 'frame_number']
    v_x1 = add_x1/frame1
    v_y1 = add_y1/frame1
    start_time = df1.loc[0, 'timestamp', ]

    add_x2 = df2.loc[len_2 - 1, 'center_x', ] - df2.loc[0, 'center_x']
    add_y2 = df2.loc[len_2 - 1, 'center_y', ] - df2.loc[0, 'center_y']
    frame2 = df2.loc[len_2 - 1, 'frame_number', ] - df2.loc[0, 'frame_number']
    v_x2 = add_x2/frame2
    v_y2 = add_y2/frame2
    join_df = df1.loc[df1.center_y > df2.center_y[0]]
    if join_df.empty:
        frame = df2.loc[0, 'frame_number'] - df1.loc[df1.shape[0]-1, 'frame_number']
        if frame < 0:
            logger.warning('没有重合轨迹且后半段起始时间比前半段结束时间早，请检查两个轨迹是否可以融合')
            return
        x_dis = df2.loc[0, 'center_x'] - df1.loc[len_1-1, 'center_x', ]
        y_dis = df2.loc[0, 'center_y'] - df1.loc[len_1-1, 'center_y', ]
        frame_num = int(round(y_dis/v_y1, 0))-1
        tmp_serise = df1.iloc[len_1-1]
        for i in range(frame_num):
            tmp_serise['center_x'] += v_x1
            tmp_serise['center_y'] += v_y1
            df1 = df1.append(tmp_serise)
        df1 = df1.append(df2)

    else:
        # 用后一段轨迹的的第一帧数据 找到前一段轨迹可以拼接的位置
        first_series = df2.iloc[0]
        v2 = df2.loc[1, 'center_y'] - df2.loc[0, 'center_y']
        # 寻找第一段拼接处数据的d
        index_df1 = abs(df1['center_y'] - (df2.loc[0, 'center_y'] - v2)).sort_values(ascending=True).index[0]
        df1 = df1.loc[0:index_df1, ].append(df2,)

    df1.reset_index(inplace=True, drop=True)
    df1['timestamp'] = [start_time + i*0.1 for i in range(df1.shape[0])]
    df1['frame_number'] = [start_time*10 + i for i in range(df1.shape[0])]
    save_name = ''.join([trackid1.split('_')[1].strip('.csv'), '_', trackid2.split('_')[1].strip('.csv'), '.csv'])
    df1.to_csv(os.path.join('data/merge_data/', save_name), index=False)
    df1.to_excel(os.path.join('data/merge_data/', save_name.replace('csv', 'xlsx')), index=False)


if __name__ == '__main__':
    # a = DrsuScene(os.path.join(os.path.dirname(__file__), 'data', 'drsu_data', 'traffic_report_obstacle_2d_drsu_split'))
    # a.get_drsu_data()
    fuse('6_11140.csv', '6_11157.csv')