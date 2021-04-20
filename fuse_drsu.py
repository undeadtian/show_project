# -*- coding: utf-8 -*-
"""
@Project : show_project
@File    : fuse_drsu.py
@Author  : 王白熊
@Data    ： 2021/4/15 15:25
"""
import os
import pandas as pd
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import time
from analysis_drsu_single import TrackDrsu
from Log import Logger

logger = Logger('DrsuScene').getlog()
FRANE_NUM = 10
FRANE_NUM_DEL_END = 5
FRANE_NUM_DEL_HEAD = 5
r_square_threshold = 0.9
distance_threshold = 0.5
frame_incr = 2
timestamp_incr = 0.1
distance_exc = 50
rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))

def draw(data, file=None, mode=0, show=False):
    if isinstance(data, pd.core.frame.DataFrame):
        df = data
    elif isinstance(data, str):
        if data.endswith('csv'):
            df = pd.read_csv(data)
        elif data.endswith('xlsx'):
            df = pd.read_excel(data)
        else:
            logger.warning('不支持的格式')
            return
    else:
        logger.warning('不支持的格式')
        return
    if mode == 0:
        plt.scatter(df.center_x, df.center_y, s=1)
        plt.savefig(file if file else r'x_y.jpg')
    elif mode == 1:
        plt.scatter(df.frame_number, df.center_x, s=1, )
        plt.savefig(file if file else r'x.jpg')
    elif mode == 2:
        plt.scatter(df.frame_number, df.center_y, s=1)
        plt.savefig(file if file else r'y.jpg')
    if show:
        plt.show()
    plt.cla()


def simple_merge(file_dir):
    list_file = glob(os.path.join(file_dir, '*.xlsx'))
    df_merge = pd.DataFrame()
    for i in list_file:
        df = pd.read_excel(i)
        df['timestamp'] = df['frame_number'] / 10
        df_merge = df_merge.append(df)
    df_merge.to_excel('data/merge_data_add/simple_merge_{}.xlsx'.format(rq), index=False)


def target_func(x, A, B):
    return A * x + B


def check_path(file_path):
    if not os.path.exists(str(file_path)):
        os.makedirs(str(file_path))
        logger.info('创建目录：{}'.format(file_path))
    return str(file_path)


class ParseDataFrame(object):
    """
    对单个trackid的数据进行分析，并在数据和头部补全数据
    """

    def __init__(self, data_file, x_flag=True):
        self.file_path = data_file
        if data_file.endswith('csv'):
            self.df = pd.read_csv(data_file)
        elif data_file.endswith('xlsx'):
            self.df = pd.read_excel(data_file)
        else:
            logger.warning('只支持.xlsx和.csv格式')
            return
        # 对df按帧号进行排序，并重置index值
        self.df = self.df.sort_values(by='frame_number').reset_index(drop=True)
        self.id = self.df.loc[0, 'id']
        # self.id = self.df.loc[0, 'track_id']
        obs_main_type = self.df.obj_type.mode()
        self.df.obj_type = obs_main_type[0] if obs_main_type[0] or len(obs_main_type) < 2 else obs_main_type[1]
        logger.info('analysis track id:{}'.format(self.id))
        self.fit_list = []
        self.utm_list = []
        # 数据长度必须大于5，小于5的数据直接丢去
        self.length = self.df.shape[0]
        self.add_df = pd.DataFrame()
        self.x_flag = x_flag

    # 计算拟合优度
    def check_fit_we(self, popt, series_x, series_y):
        series_x = series_x.reset_index(drop=True)
        series_y = series_y.reset_index(drop=True)
        y_prd = pd.Series(list(map(lambda x: popt[0] * x + popt[1], series_x)))
        egression = sum((y_prd - series_x.mean()) ** 2)  # r回归平方和
        residual = sum((series_y - y_prd) ** 2)  # 残差平方和
        total = sum((series_y - series_y.mean()) ** 2)  # 总体平方和
        if total == 0:
            r_square = 1
        else:
            r_square = 1 - residual / total  # 相关性系数R^2
        logger.debug('对track id:%s 轨迹进行拟合，拟合参数:%s,拟合优度：%s' % (self.id, popt, r_square))
        return r_square

    # 利用curve_fit 函数获取拟合参数 以及判断拟合优度,返回值为参数及标准方差
    def check_stright_fit(self, series_x, series_y):
        popt, pcov = optimize.curve_fit(target_func, series_x, series_y)
        perr = np.sqrt(np.diag(pcov))
        r_square = self.check_fit_we(popt, series_x, series_y)
        return popt, r_square

    def get_stright_fit(self):
        """
        分别计算出行车轨迹全程的拟合直线 以及前 FRANE_NUM 帧和后FRANE_NUM帧数据轨迹的拟合直线
        """
        if self.x_flag:
            self.fit_list.append(self.check_stright_fit(self.df['center_x'], self.df['center_y']))
            # self.check_stright_fit(self.df['center_x'][0:max(length//4, 5)], self.df['center_y'][0:max(length//4, 5)])
            # self.check_stright_fit(self.df['center_x'][min((length*3) // 4, self.df.shape[0]-5)::], self.df['center_y'][min((length*3) // 4, self.df.shape[0]-5)::])
            self.fit_list.append(self.check_stright_fit(self.df['center_x'][0:FRANE_NUM],
                                                        self.df['center_y'][0:FRANE_NUM]))
            self.fit_list.append(self.check_stright_fit(self.df['center_x'][self.df.shape[0] - FRANE_NUM::],
                                                        self.df['center_y'][self.df.shape[0] - FRANE_NUM::]))
        else:
            self.fit_list.append(self.check_stright_fit(self.df['center_y'], self.df['center_x']))
            # self.check_stright_fit(self.df['center_x'][0:max(length//4, 5)], self.df['center_y'][0:max(length//4, 5)])
            # self.check_stright_fit(self.df['center_x'][min((length*3) // 4, self.df.shape[0]-5)::], self.df['center_y'][min((length*3) // 4, self.df.shape[0]-5)::])
            self.fit_list.append(self.check_stright_fit(self.df['center_y'][0:FRANE_NUM],
                                                        self.df['center_x'][0:FRANE_NUM]))
            self.fit_list.append(self.check_stright_fit(self.df['center_y'][self.df.shape[0] - FRANE_NUM::],
                                                        self.df['center_x'][self.df.shape[0] - FRANE_NUM::]))

    def get_utm_offset(self):
        """
        分别计算出行车轨迹全程的x,y方向差值，以及前 FRANE_NUM 帧和后FRANE_NUM帧数据轨迹的坐标差值
        """
        series_x = self.df['center_x']
        series_y = self.df['center_y']
        length = self.df.shape[0]
        self.utm_list.append(
            [series_x.iloc[length - 1] - series_x.iloc[0], series_y.iloc[length - 1] - series_y.iloc[0]])
        self.utm_list.append(
            [series_x.iloc[FRANE_NUM] - series_x.iloc[0], series_y.iloc[FRANE_NUM] - series_y.iloc[0]])
        self.utm_list.append(
            [series_x.iloc[length - 1] - series_x.iloc[length - FRANE_NUM - 1],
             series_y.iloc[length - 1] - series_y.iloc[length - FRANE_NUM - 1]])

    # 表的前几帧和后几帧数据不稳定删除掉
    def del_end(self):
        self.df = self.df.drop([i for i in range(self.df.shape[0] - FRANE_NUM_DEL_END, self.df.shape[0])])
        self.df = self.df.reset_index(drop=True)
        self.get_utm_offset()

    def repair_end(self):
        logger.info('补充尾部数据')
        series_data = self.df.iloc[self.df.shape[0] - 1]
        series_tmp = self.df.iloc[self.df.shape[0] - 1]
        # 选择以哪条拟合直线为补全的基准  fit_list[0][0] 为全局拟合直线，fit_list[1][0]为头FRAME_NUM
        # 帧拟合直线
        popt = self.fit_list[2][0]
        # x_incr = (self.df.center_x.iloc[self.df.shape[0] - 1] - self.df.center_x.iloc[0]) / self.df.shape[0]
        # y_incr = (self.df.center_y.iloc[self.df.shape[0] - 1] - self.df.center_y.iloc[0]) / self.df.shape[0]
        x_incr = (self.df.center_x.iloc[self.df.shape[0] - 1] - self.df.center_x.iloc[
            self.df.shape[0] - 1 - FRANE_NUM]) / FRANE_NUM
        y_incr = (self.df.center_y.iloc[self.df.shape[0] - 1] - self.df.center_y.iloc[
            self.df.shape[0] - 1 - FRANE_NUM]) / FRANE_NUM
        if abs(x_incr) < 0.1 and abs(y_incr) < 0.1:
            logger.info('速度较小，丢弃')
            return
        if self.x_flag:
            # 最多延长距离为distance_exc
            for i in range(1, abs(int(distance_exc / x_incr))):
                series_tmp.center_x = series_data.center_x + x_incr * i
                series_tmp.center_y = popt[0] * series_tmp.center_x + popt[1]
                series_tmp.frame_number = series_data.frame_number + frame_incr * i
                series_tmp.timestamp = series_data.timestamp + timestamp_incr * i
                self.df = self.df.append(series_tmp)
                self.add_df = self.add_df.append(series_tmp)
        else:
            for i in range(1, abs(int(distance_exc / y_incr))):
                series_tmp.center_y = series_data.center_y + y_incr * i
                series_tmp.center_x = popt[0] * series_tmp.center_y + popt[1]
                series_tmp.frame_number = series_data.frame_number + frame_incr * i
                series_tmp.timestamp = series_data.timestamp + timestamp_incr * i
                self.df = self.df.append(series_tmp)
                self.add_df = self.add_df.append(series_tmp)

    def parse_data(self):
        """
        数据处理入口
        """
        if self.df.shape[0] < FRANE_NUM + FRANE_NUM_DEL_HEAD + FRANE_NUM_DEL_END:
            logger.info('障碍物数据帧数小于{},数据不处理'.format(FRANE_NUM + FRANE_NUM_DEL_HEAD + FRANE_NUM_DEL_END))
            return

        self.del_end()
        if not self.utm_list:
            self.get_utm_offset()
        if self.utm_list[0][1] > 0:
            logger.info('逆向车道车辆不处理')
            return
        if abs(self.utm_list[0][0]) + abs(self.utm_list[0][1]) < distance_threshold:
            logger.info('trackid:{}, 运动距离小于{} 判定为静止，暂时不处理'.format(self.id, distance_threshold))
            self.data_save()
            return
        if not self.fit_list:
            self.get_stright_fit()
        self.repair_end()
        self.draw(show=False)
        self.draw(mode=1, show=False)
        self.draw(mode=2, show=False)
        self.data_save()

    def parse_data_end(self):
        if not self.utm_list:
            self.get_utm_offset()
        if not self.fit_list:
            self.get_stright_fit()
        self.repair_end()

    def data_save(self):
        self.df = self.df.sort_values(by='frame_number')
        self.df = self.df.reset_index(drop=True)
        save_path = '{}_repaired'.format(os.path.dirname(self.file_path))
        check_path(save_path)
        self.df.to_excel(os.path.join(save_path, '{}.xlsx'.format(self.id)), index=False)
        if not self.add_df.empty:
            save_add_path = '{}_add_repaired'.format(os.path.dirname(self.file_path))
            check_path(save_add_path)
            self.add_df = self.add_df.sort_values(by='frame_number')
            self.add_df = self.add_df.reset_index(drop=True)
            self.add_df.to_excel(os.path.join(save_add_path, '{}.xlsx'.format(self.id)), index=False)

    def draw(self, mode=0, show=True):
        save_path = '{}_repaired'.format(os.path.dirname(self.file_path))
        check_path(save_path)
        if mode == 0:
            plt.scatter(self.df.center_x, self.df.center_y, s=1)
            plt.savefig(os.path.join(save_path, '{}.jpg'.format(self.id)))
        elif mode == 1:
            plt.scatter(self.df.frame_number, self.df.center_x, s=1, )
            plt.savefig(os.path.join(save_path, '{}_x.jpg'.format(self.id)))
        elif mode == 2:
            plt.scatter(self.df.frame_number, self.df.center_y, s=1)
            plt.savefig(os.path.join(save_path, '{}_y.jpg'.format(self.id)))
        # plt.scatter(self.df.center_x, self.df.center_y, s=1)
        # plt.savefig(os.path.join(save_path, '{}.jpg'.format(self.id)))
        if show:
            plt.show()
        plt.cla()


# trackid１前数据
def fuse(trackid1, trackid2, object_type=None, use_start=False):
    dir_name = 'data/drsu_data/new1748_drsu_split'
    df1 = pd.read_csv(os.path.join(dir_name, trackid1))
    df2 = pd.read_csv(os.path.join(dir_name, trackid2))

    df1 = df1.dropna(axis=0, how='all')
    df2 = df2.dropna(axis=0, how='all')
    len_1 = df1.shape[0]
    len_2 = df2.shape[0]
    add_x1 = df1.loc[len_1 - 6, 'center_x'] - df1.loc[len_1 - 26, 'center_x']
    add_y1 = df1.loc[len_1 - 6, 'center_y'] - df1.loc[len_1 - 26, 'center_y']
    frame1 = df1.loc[len_1 - 6, 'frame_number',] - df1.loc[len_1 - 26, 'frame_number']
    v_x1 = add_x1 / frame1
    v_y1 = add_y1 / frame1
    start_time = df1.loc[0, 'timestamp',]

    add_x2 = df2.loc[len_2 - 6, 'center_x',] - df2.loc[0, 'center_x']
    add_y2 = df2.loc[len_2 - 6, 'center_y',] - df2.loc[0, 'center_y']
    frame2 = df2.loc[len_2 - 6, 'frame_number',] - df2.loc[0, 'frame_number']
    v_x2 = add_x2 / frame2
    v_y2 = add_y2 / frame2
    join_df = df1.loc[df1.center_y < df2.center_y[0]]
    if join_df.empty:
        frame = df2.loc[0, 'frame_number'] - df1.loc[df1.shape[0] - 1, 'frame_number']
        if frame < 0:
            logger.warning('没有重合轨迹且后半段起始时间比前半段结束时间早，请检查两个轨迹是否可以融合')
            return
        x_dis = df2.loc[0, 'center_x'] - df1.loc[len_1 - 1, 'center_x',]
        y_dis = df2.loc[0, 'center_y'] - df1.loc[len_1 - 1, 'center_y',]
        frame_num = int(round(y_dis / v_y1, 0)) - 1
        v_x = x_dis / frame_num
        tmp_serise = df1.iloc[len_1 - 1]
        for i in range(frame_num):
            tmp_serise['center_x'] += v_x
            tmp_serise['center_y'] += v_y1
            df1 = df1.append(tmp_serise)
        df1 = df1.append(df2)

    else:
        # 用后一段轨迹的的第一帧数据 找到前一段轨迹可以拼接的位置
        first_series = df2.iloc[0]
        v2 = df2.loc[1, 'center_y'] - df2.loc[0, 'center_y']
        # 寻找第一段拼接处数据的d
        if use_start:
            index_df2 = abs(df2['center_y'] - (df1.loc[df1.shape[0]-1, 'center_y'] + v2)).sort_values(ascending=True).index[0]
            df1 = df1.append(df2.iloc[index_df2:, ])
        else:
            index_df1 = abs(df1['center_y'] - (df2.loc[0, 'center_y'] - v2)).sort_values(ascending=True).index[0]
            df1 = df1.loc[0:index_df1, ].append(df2, )
    track_id = trackid1.split('_')[1].strip('.csv')
    df1['id'] = track_id
    if object_type:
        df1['obj_type'] = object_type
    df1.reset_index(inplace=True, drop=True)
    df1['timestamp'] = [start_time + i * 0.1 for i in range(df1.shape[0])]
    df1['frame_number'] = [start_time * 10 + i for i in range(df1.shape[0])]
    save_name = ''.join([track_id, '_', trackid2.split('_')[1].strip('.csv'), '.csv'])
    draw(df1, file=os.path.join('data/merge_data_0420/', save_name.replace('csv', 'jpg')), show=True)
    df1.to_csv(os.path.join('data/merge_data_0420/', save_name), index=False)
    df1.to_excel(os.path.join('data/merge_data_0420/', save_name.replace('csv', 'xlsx')), index=False)


def repair_drsu(drsu_data_dir, x_flag=True):
    csv_list = glob(os.path.join(drsu_data_dir, '*.csv'))
    for i in csv_list:
        if int(os.path.basename(i).split('_')[1].strip('.csv')) in [11232, 11278, 11263, 11293, 11304, 11309, 11254,
                                                                    11265, 11282, 11291, 11290, 11312, 11264]:
            continue
        if int(os.path.basename(i).split('_')[0]) not in [6, 7]:
            continue
        Track_data = ParseDataFrame(i, x_flag)
        Track_data.parse_data()

def repair_end_static():
    dict_track = {
                  11290: '3344446.1',
                  11293: '3344446.2',
                  11291: '3344437.1',
                  11282: '3344431.3',
                  11263: '3344439.1',
                  11265: '3344424.0',
                  11278: '3344430.1',
                  11254: '3344418.1',
                  11232: '3344422.1',
                  }
    source = 11304
    file_path = 'data/drsu_data/new1748_drsu_split'
    tmp_df = pd.read_csv(os.path.join(file_path, '6_11304.csv'))
    last_y = tmp_df.loc[tmp_df.shape[0]-1, 'center_y']
    for key, value in dict_track.items():
        df1 = pd.read_csv(os.path.join(file_path, '6_{}.csv'.format(str(key))))
        df1 = df1.drop([i for i in range(df1.shape[0] - FRANE_NUM_DEL_END, df1.shape[0])])
        df1.reset_index(inplace=True, drop=True)

        start_time = df1.loc[0, 'timestamp']
        df1_y_end = float(df1.loc[df1.shape[0]-1, 'center_y'])
        repair_dis_y = df1_y_end - float(value)
        repair_df = tmp_df.loc[tmp_df.center_y < last_y + repair_dis_y]
        repair_df.loc[:,'center_y'] = repair_df['center_y'] - (last_y + repair_dis_y - df1_y_end)
        add_x = (df1.loc[df1.shape[0]-1, 'center_x'] - df1.loc[df1.shape[0]-6, 'center_x'])/6
        # repair_df.loc[:, 'center_x'] = [df1.loc[df1.shape[0]-1, 'center_x'] - i*add_x if i < repair_df.shape[0] - 60 else df1.loc[df1.shape[0]-1, 'center_x'] - (repair_df.shape[0] - 60)*add_x for i in range(repair_df.shape[0])]
        repair_df.loc[:, 'center_x'] = df1.loc[df1.shape[0]-1, 'center_x']
        df1 = df1.append(repair_df, ignore_index=True)
        df1.loc[:, 'id'] = key
        df1.reset_index(inplace=True, drop=True)
        df1.loc[:, 'timestamp'] = [start_time + i * 0.1 for i in range(df1.shape[0])]
        df1.loc[:, 'frame_number'] = [start_time * 10 + i for i in range(df1.shape[0])]
        draw(df1, file=os.path.join('data/drsu_data/new1748_drsu_spilt_static', '{}.jpg'.format(key)), show=False)
        df1.to_csv(os.path.join('data/drsu_data/new1748_drsu_spilt_static', '{}.csv'.format(key)), index=False)
        df1.to_excel(os.path.join('data/drsu_data/new1748_drsu_spilt_static', '{}.xlsx'.format(key)), index=False)


def del_ob(file_dir):
    files_path = glob(os.path.join(file_dir, '*.xlsx'))
    for i in files_path:
        df1 = pd.read_excel(i)
        if df1.loc[df1.shape[0]-1, 'center_y'] > df1.loc[0, 'center_y']:
            save_name = i.replace('split', 'split_repaired').replace('6_', '')
            df1.to_excel(save_name, index=False)

if __name__ == '__main__':
    list_ = [
        # ('6_11140.csv', '6_11157.csv'),
        # ('6_11151.csv', '6_11168.csv'),
        # ('6_11180.csv', '6_11216.csv'),
        # ('6_11197.csv', '6_11237.csv'),
        # ('6_11226.csv', '6_11244.csv'),
        # ('6_11248.csv', '6_11270.csv'),
        # ('6_11309.csv', '6_11325.csv'),
        # ('6_11335.csv', '6_11349.csv'),
        # ('6_11323.csv', '6_11351.csv'),
        # ('6_11329.csv', '6_11339.csv'),

        # ('6_11164.csv', '6_11241.csv'),

        # ('6_11278.csv', '6_11311.csv'),
        # ('6_11304.csv', '6_11344.csv'),
        # ('6_11290.csv', '6_11278.csv'),

        # ('6_11140.csv', '6_11157.csv'),
        # ('6_11140.csv', '6_11157.csv'),
        # ('6_11140.csv', '6_11157.csv'),
        # ('6_11140.csv', '6_11157.csv'),
        # ('6_11140.csv', '6_11157.csv'),
        # ('6_11140.csv', '6_11157.csv'),
        # ('6_11140.csv', '6_11157.csv'),
        # ('6_11140.csv', '6_11157.csv'),
        # ('4_11149.csv', '4_11185.csv'),
        # ('4_11159.csv', '4_11192.csv'),
        # ('4_11252.csv', '4_11326.csv'),
        # ('4_11300.csv', '11252_11326.csv'),
        # ('4_11327.csv', '11159_11192.csv'),
        # ('4_11331.csv', '11149_11185.csv'),

        # ('6_11236.csv', '6_11257.csv'),

        ('6_11266.csv', '6_11284.csv'),
        # ('6_11297.csv', '6_11308.csv'),
        # ('6_11301.csv', '6_11316.csv'),
        # ('6_11138.csv', '6_11142.csv'),
        # ('6_11127.csv', '6_11153.csv'),


    ]
    for i in list_:
        fuse(*i, object_type=6)

    # for i in list_:
    #     fuse(*i, object_type=4, use_start=True)

    # tmp_list = []
    # for i in list_:
    #     for j in i:
    #         tmp_list.append(int(j.split('_')[1].strip('.csv')))
    # print(tmp_list)
    # simple_merge('data/drsu_data/new1748_drsu_split_repaired')
    # repair_drsu('data/drsu_data/new1748_drsu_split', x_flag=False)
    # repair_end_static()
    # del_ob('data/drsu_data/new1748_drsu_split')