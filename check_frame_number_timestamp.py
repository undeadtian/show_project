# -*- coding: utf-8 -*-
"""
@Project : show_project
@File    : check_frame_number_timestamp.py
@Author  : 王白熊
@Data    ： 2021/3/25 14:45
"""
import pandas as pd
from Log import Logger
import os

logger = Logger('check_frame_number_timstamp').getlog()

def check_path(file_path):
    if not os.path.exists(str(file_path)):
        os.makedirs(str(file_path))
        logger.info('创建目录：{}'.format(file_path))
    return str(file_path)

def check_frame_number_timstamp(file_name, mode=0):
    if file_name.endswith('csv'):
        df = pd.read_csv(file_name)
    elif file_name.endswith('xlsx'):
        df = pd.read_excel(file_name)
    else:
        logger.warning('不支持的格式')
        return
    track_ids = df.groupby('id')
    df.sort_values(by='id')
    first_flag = True
    new_df = pd.DataFrame()
    for track_id, group in track_ids:
        group_single = track_ids.get_group(track_id)
        dir_path = 'data/drsu_data/{}'.format(os.path.basename(file_name).split('.')[0])
        check_path(dir_path)
        group_single.to_excel(os.path.join(dir_path, '{}.xlsx'.format(track_id)))
        if first_flag:
            tmp_series = group_single.iloc[0]
            start_frame_number = tmp_series.frame_number
            start_times_stamp = tmp_series.timestamp
            first_flag = False
        group_single['timestamp'] = (group_single['frame_number'] - start_frame_number)*0.1 + start_times_stamp
        group_single.to_excel(os.path.join(dir_path, '{}parsed.xlsx'.format(track_id)))
        new_df = new_df.append(group_single)

    new_df.to_excel(file_name.replace('.', 'parsed.'))


if __name__ == '__main__':
    check_frame_number_timstamp(r'data\origin_data\traffic_report_obstacle_2d.xlsx')