# -*- coding: utf-8 -*-
"""
@Project : show_project
@File    : vehicle_simulation.py
@Author  : 王白熊
@Data    ： 2021/3/22 9:33
"""

import pandas as pd
from scipy import optimize
import numpy as np
import math, os
import random
import time
from glob import glob
from pyecharts import Line
from Log import Logger
import matplotlib.pyplot as plt

main_type = [6, 7, 8]  # 过滤出类型为678（car truck bus）的障碍物
ANGEL_VER_VALUE = 57.3
FRANE_NUM = 20
FRANE_NUM_DEL_END = 5
FRANE_NUM_DEL_HEAD = 5
r_square_threshold = 0.9
distance_threshold = 0.5
frame_incr = 2
timestamp_incr = 0.1
# start_time = 1615864730.3  # 临时数据
# start_time = 1616384226.7  # drsu3
# start_time = 16163847999.4  # drsu2
start_time = 1616384781.8  # drsu2
ratio_v = 0.7
distance_exc = 150
FRANE_NUM_MIN = 20  # 真实drsu数据如果障碍物帧数小于改值，就对障碍物进行过滤
show_time = 800
frame_period = 1  # 帧数间隔，确定两个连续数据的frame_number差值
utm_list = []

logger = Logger('').getlog()


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
            [series_x.iloc[length - 1] - series_x.iloc[FRANE_NUM],
             series_y.iloc[length - 1] - series_y.iloc[FRANE_NUM]])

    def repair_head(self):
        """
        # 直接用全局的拟合直线补全，比起始数据补全更好
        用拟合出的直线函数对数据进行补全头部信息
        """
        logger.info('补充头部数据')
        series_data = self.df.iloc[0]
        series_tmp = self.df.iloc[0]
        # 选择以哪条拟合直线为补全的基准  fit_list[0][0] 为全局拟合直线，fit_list[1][0]为头FRAME_NUM
        # 帧拟合直线
        popt = self.fit_list[1][0]
        if self.x_flag:
            x_incr = (self.df.center_x.iloc[FRANE_NUM] - self.df.center_x.iloc[0]) / FRANE_NUM
            if abs(x_incr) < 0.1:
                return
            for i in range(1, abs(int(distance_exc / x_incr))):
                series_tmp.center_x = series_data.center_x - x_incr * i
                series_tmp.center_y = popt[0] * series_tmp.center_x + popt[1]
                series_tmp.timestamp = series_data.timestamp - timestamp_incr * i
                series_tmp.frame_number = series_data.frame_number - frame_incr * i
                # if series_tmp.timestamp < start_time:
                #     #  不能比整体最早时间还早
                #     break
                self.df = self.df.append(series_tmp)
                self.add_df = self.add_df.append(series_tmp)
        else:
            y_incr = (self.df.center_y.iloc[FRANE_NUM] - self.df.center_y.iloc[0]) / FRANE_NUM
            if abs(y_incr) < 0.1:
                return
            for i in range(1, abs(int(distance_exc / y_incr))):
                series_tmp.center_y = series_data.center_y - y_incr * i
                series_tmp.center_x = popt[0] * series_tmp.center_y + popt[1]
                series_tmp.timestamp = series_data.timestamp - timestamp_incr * i
                series_tmp.frame_number = series_data.frame_number - frame_incr * i
                # if series_tmp.timestamp < start_time:
                #     #  不能比整体最早时间还早
                #     break
                self.df = self.df.append(series_tmp)
                self.add_df = self.add_df.append(series_tmp)
        # 最多延长距离为distance_exc

        self.df = self.df.sort_values(by='frame_number')
        self.df = self.df.reset_index(drop=True)

    # 表的前几帧和后几帧数据不稳定删除掉
    def del_head_end(self):
        self.df = self.df.drop([i for i in range(self.df.shape[0] - FRANE_NUM_DEL_END, self.df.shape[0])])
        self.df = self.df.drop([i for i in range(FRANE_NUM_DEL_HEAD)])
        self.df = self.df.reset_index(drop=True)
        self.get_utm_offset()

    def replace_head(self):
        vx_head = (self.df.center_x.iloc[FRANE_NUM] - self.df.center_x.iloc[0]) // FRANE_NUM
        vx = (self.df.center_x.iloc[self.length - 1] - self.df.center_x.iloc[0]) // self.length
        if vx > 0:
            return
        if vx_head / vx > ratio_v:
            logger.info('初始{}帧的平均速度达到平均数据的70%，不进行修改')
            return
        tmp_series = self.df.iloc[FRANE_NUM]
        popt = self.fit_list[0][0]
        for i in range(FRANE_NUM):
            self.df.loc[i, 'center_x'] = tmp_series.center_x - (FRANE_NUM - i) * vx
            self.df.loc[i, 'center_y'] = popt[0] * self.df.loc[i, 'center_x'] + popt[1]

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
        self.del_head_end()
        if not self.utm_list:
            self.get_utm_offset()
        if abs(self.utm_list[0][0]) + abs(self.utm_list[0][1]) < distance_threshold:
            logger.info('trackid:{}, 运动距离小于{} 判定为静止，暂时不处理'.format(self.id, distance_threshold))
            self.data_save()
            return
        if not self.fit_list:
            self.get_stright_fit()
        self.repair_head()
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


class ACUTrackData():
    def __init__(self, file_path, copy_num=20):
        """
        :param file_path: 带处理数据文件名
        :param copy_num: 拷贝数量
        """
        if isinstance(file_path, str):
            logger.info('读入文件：{}'.format(file_path))
            if file_path.endswith('csv'):
                self.df = pd.read_csv(file_path)
            elif file_path.endswith('xlsx'):
                self.df = pd.read_excel(file_path)
            else:
                logger.warning('只支持.xlsx和.csv格式')
                return
            self.file_path = file_path
        else:
            logger.warning('参数只支持DataFrame以及文件目录')
            return
        self.rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        self.index_list = []
        self.df = self.df.sort_values(by='frame_number')
        self.df = self.df.reset_index(drop=True)
        self.start_time = self.df.loc[0, 'frame_number']
        self.start_timestamp = self.df.loc[0, 'timestamp']
        self.id = self.df.iloc[0].track_id
        self.copy_num = copy_num

    def get_time_list(self):
        """
        选取多个时间点，将这些时间点上车辆对应的坐标作为虚拟车辆的起始坐标。需满足车辆都在行驶状态
        """
        df = self.df.loc[abs(self.df['velocity_x']) > 3]
        data_length = df.shape[0]
        per_frame = data_length // self.copy_num
        for i in range(1, self.copy_num):
            # frame_offset = random.randint(i * per_frame, (i + 1) * per_frame)
            frame_offset = i * per_frame
            index = df.iloc[frame_offset].name
            self.index_list.append(index)

    def copy_data(self):
        for i in self.index_list:
            if i + show_time > self.df.shape[0]:
                continue
            df_copy = self.df.iloc[i:i + show_time, :]
            df_copy = df_copy.drop(columns=['frame_number'])
            df_copy = df_copy.reset_index(drop=True)
            df_copy['frame_number'] = [j * frame_period + self.start_time for j in range(df_copy.shape[0])]
            df_copy['timestamp'] = [j * 0.1 + self.start_timestamp for j in range(df_copy.shape[0])]
            df_copy['id'] = self.id
            df_copy['track_id'] = self.id
            dir_path = r'data\track_data\{}_{}'.format(os.path.basename(self.file_path).split('.')[0], self.copy_num)
            check_path(dir_path)
            draw(df_copy, os.path.join(dir_path, '{}.jpg'.format(self.id)))
            df_copy.to_excel(os.path.join(dir_path, '{}.xlsx'.format(self.id)), index=False)
            logger.info('生成track_id：{}的车辆行驶数据'.format(self.id))
            self.id += 1

    def copy_static_data(self):
        file_path = r'data_by_hand\{}.xlsx'.format(self.id - 1)
        tmp_df = pd.read_excel(file_path)
        for i in utm_list:
            df = tmp_df
            df.center_x = i[0]
            df.center_y = i[1]
            df.track_id = self.id
            self.id += 1

            logger.info('生成track_id：{}的车辆静态数据，坐标({},{})'.format(self.id, i[0], i[1]))
            df.to_excel(r'track_data_{}\{}.xlsx'.format(self.copy_num, self.id), index=False)

    def parse_data(self):
        self.get_time_list()
        self.copy_data()


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


# 把acu的数据转换为能存入数据库的障碍物信息，主要是进行格式的转换
def drc2obs(acu_file, first_flag=True):
    # acu_file = r'data\acu_status_info_history_6_319.xlsx'  # 原始数据目录
    # obs_file = r'data\acu_status_info_history_6_parsed.xlsx'  # 初步处理后数据目录
    # round_file = r'data\acu_status_info_history_6_round.xlsx'
    obs_file = acu_file.replace('.', 'parsed.')
    round_file = acu_file.replace('.', 'round.')
    obs_df = pd.DataFrame(columns=['id', 'center_x', 'center_y', 'center_z', 'length', 'width',
                                   'height', 'obj_type', 'velocity_x', 'velocity_y',
                                   'velocity_z', 'acceleration_x', 'acceleration_y',
                                   'acceleration_z', 'theta', 'local_timestamp', 'timestamp'])
    # 固定值直接赋值

    acu_df = pd.read_excel(acu_file)
    logger.info('读取acu数据文件：{}'.format(acu_file))
    acu_df = acu_df.loc[acu_df['ulglobalacuid'] == 810110007]
    acu_df = acu_df.sort_values(by='local_timestamp')
    acu_df.reset_index(drop=True)
    # plt.scatter(acu_df.stcoordinate_dx, acu_df.stcoordinate_dy, s=1)
    # plt.savefig(r'data\acu.jpg')
    start_timestamp = acu_df.loc[0, 'acu_time_stamp']
    if first_flag:
        for index, row in acu_df.iterrows():
            obs_df.loc[index, ['center_x', 'center_y', 'velocity_x', 'velocity_y', 'acceleration_x',
                               'acceleration_y', 'local_timestamp']] = \
                acu_df.loc[index, ['stcoordinate_dx', 'stcoordinate_dy', 'stlinespeed_dx', 'stlinespeed_dy',
                                   'acceleration_dx', 'acceleration_dy', 'local_timestamp']].tolist()
            # theta的值处理方法：输入的角度先加上pi，然后对2pi取模,如果取模后为负，则加上2pi，调成正值
            # 最终再减去pi，就归一化到 - pi到pi了 取模使用fmod函数
            theta_tmp = math.fmod((acu_df.loc[index, 'dbyaw'] + math.pi), (2 * math.pi))
            theta = theta_tmp - math.pi if theta_tmp > 0 else theta_tmp + math.pi
            if theta > math.pi or theta < -math.pi:
                logger.warning('出现theta角度异常，index:{},theta:{}'.format(index, theta))
            obs_df.loc[index, 'theta'] = theta
            acu_df.loc[index, 'dbyaw'] = theta
            # logger.info(index)
        acu_df.to_excel(acu_file, index=False)
    else:
        obs_list = ['center_x', 'center_y', 'velocity_x', 'velocity_y', 'acceleration_x',
                    'acceleration_y', 'local_timestamp', 'theta']
        acu_list = ['stcoordinate_dx', 'stcoordinate_dy', 'stlinespeed_dx', 'stlinespeed_dy',
                    'acceleration_dx', 'acceleration_dy', 'local_timestamp', 'dbyaw']
        for i in range(len(obs_list)):
            obs_df[obs_list[i]] = acu_df[acu_list[i]]
    obs_df['id'], obs_df['center_z'], obs_df['length'], obs_df['width'], obs_df['height'], obs_df['obj_type'], \
    obs_df['velocity_z'], obs_df['acceleration_z'] = 1, 14.13, 10, 4, 3, 8, 0, 0
    obs_df['track_id'], obs_df['lane_ids'], obs_df['connection_ids'], obs_df['det_confidence'], obs_df['obs_drsuids'], \
    obs_df['is_valid'] = 1, -1, -1, 1, 810020004, 't'
    # 1615893304.1 1616035597000
    # obs_df = obs_df.drop(index=obs_df.loc[obs_df['center_y'] > UTM_Y_MAX].index)
    obs_df['timestamp'] = [start_timestamp // 1000 + 0.1 * i for i in range(obs_df.shape[0])]
    obs_df['frame_number'] = [i * 2 + 10000 for i in range(obs_df.shape[0])]
    obs_df = obs_df.reset_index(drop=True)
    obs_df.to_excel(obs_file, index=False)
    round_df = obs_df
    round_df['center_x'] = round_df['center_x'].round(2)
    round_df['center_y'] = round_df['center_y'].round(2)
    round_df.to_excel(round_file, index=False)
    logger.info('acu原始数据转换完成')
    obs_df.to_csv(os.path.join(os.path.dirname(acu_file), 'acu_.csv'), index=False)


def split_by_id(origin_file, choose_type=None, sort_values='frame_number'):
    """
    对包含该所有track的excel原始数据进行简单处理，处理后的数据存放在指定文件夹
    choose_type 是否需要对障碍物类型进行过滤
    """
    if choose_type is None:
        choose_type = [i for i in range(14)]
    # df1 = pd.read_excel('drsu03-0316-1914-1919-deal.xlsx')
    df1 = pd.read_excel(origin_file)
    # 将障碍物按id进行group
    track_ids = df1.groupby('id')
    # logger.info('类型为678的障碍物总个数：{}'.format(len(track_ids)))
    for track_id, group in track_ids:
        group_single = track_ids.get_group(track_id)
        if group_single.shape[0] < FRANE_NUM_MIN:
            continue
        obs_main_type = group_single['obj_type'].mode()
        if obs_main_type[0] in choose_type:
            obj_type = obs_main_type[0] if obs_main_type[0] or len(obs_main_type) < 2 else obs_main_type[1]
            group_single.loc[:, 'obj_type'] = obj_type
            group_single = group_single.sort_values(by=sort_values)
            group_single = group_single.reset_index(drop=True)
            dir_path = 'data\drsu_data\{}_drsu_split'.format(
                ''.join(os.path.basename(origin_file).replace('.xlsx', '')))
            check_path(dir_path)
            dir_name = ''.join([str(obj_type), '_', str(track_id)])
            draw(group_single, os.path.join(dir_path, '{}.jpg'.format(dir_name)))
            group_single.to_excel(os.path.join(dir_path, '{}.xlsx'.format(dir_name)), index=False)
            group_single.to_csv(os.path.join(dir_path, '{}.csv'.format(dir_name)), index=False)


# 将真实数据和模拟数据进行融合
def merge_orgin_parsed_data(origin_path=r'D:\test_code2\project\vis_show\add_data_all', parsed_path=r'data_by_hand'):
    ori_list = glob(os.path.join(origin_path, '*.csv'))
    ori_list.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
    pars_list = glob(os.path.join(parsed_path, '*'))
    tmp_df = pd.DataFrame()
    all_df = pd.DataFrame()
    add_df = pd.DataFrame()
    for index in range(len(ori_list)):
        df_ori = pd.read_csv(ori_list[index])
        logger.info('开始处理文件：{}'.format(ori_list[index]))
        if index == 0:
            tmp_df = df_ori
            # start_frame_number = df_ori.sort_values(by='frame_number').iloc[0].frame_number
            start_timestamp = df_ori.sort_values(by='timestamp').iloc[0].timestamp
            start_frame_number_del = round(start_timestamp * 10, 0)
            # tmp_df['frame_number'] = [i for i in range(tmp_df.shape[0])]
        df_ori['frame_number'] = (df_ori['timestamp'] * 10).round(0) - start_frame_number_del
        # df_ori['frame_number'] = (df_ori['timestamp']-start_timestamp)*10 + start_frame_number
        all_df = all_df.append(df_ori)
    ori_df = all_df.copy()
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    ori_df.to_csv(r'merge_ori\{}ori_parsed.csv'.format(rq), index=False)
    ori_df = ori_df.sort_values(by='frame_number')
    frame_numbers = ori_df.groupby('frame_number')
    logger.info('原始数据总帧数：{}'.format(len(frame_numbers)))
    index_frame = 0
    time_stamp_list = []
    frame_number_list = []
    for frame_number, group in frame_numbers:
        if index_frame == show_time:
            break
        group_single = frame_numbers.get_group(frame_number)
        group_single = group_single.reset_index(drop=True)
        time_stamp = group_single.loc[random.randint(0, group_single.shape[0] - 1), 'timestamp']
        time_stamp_list.append(time_stamp)
        frame_number_list.append(frame_number)
        index_frame += 1
    logger.info('time_stamp_list:{}'.format(time_stamp_list))
    # start_time = tmp_df.iloc[0].frame_number
    # start_timestamp = tmp_df.iloc[0].timestamp
    end_id = int(os.path.basename(ori_list[0]).split('.')[0])
    per_num = end_id // len(pars_list)
    for index1 in range(len(pars_list)):
        if pars_list[index1].endswith('csv'):
            df_par = pd.read_csv(pars_list[index1])
        elif pars_list[index1].endswith('xlsx'):
            df_par = pd.read_excel(pars_list[index1])
        else:
            logger.error('不支持的类型')
            return
        # df_par = pd.read_excel(pars_list[index1])
        # if tmp_df.shape[0] < df_par.shape[0]:
        #     logger.error('真实数据帧数小于{}帧'.format(show_time))
        #     return
        # df_par['frame_number'] = [j * frame_period + start_time for j in range(df_par.shape[0])]
        # df_par['frame_number'] = frame_number_list
        df_par['frame_number'] = tmp_df['frame_number']
        # df_par['timestamp'] = tmp_df['timestamp']
        # df_par['timestamp'] = time_stamp_list
        df_par['timestamp'] = tmp_df['timestamp']
        track_id = random.randint(per_num * index1, per_num * (index1 + 1))
        df_par['id'] = track_id
        if index1 % 10 == 0:
            df_par['obj_type'] = random.choice([7, 8])
        else:
            df_par['obj_type'] = 6
        df_par['track_id'] = track_id
        # df_par['local_timestamp'] = tmp_df.iloc[0].local_timestamp
        df_par['local_timestamp'] = tmp_df.loc[0, 'local_timestamp']
        df_par.to_csv(r'merge_ori\{}.csv'.format(index1), index=False)
        logger.info('开始处理文件：{}, track_id:{}'.format(pars_list[index1], track_id))
        add_df.append(df_par)
        all_df = all_df.append(df_par)
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    all_df.to_excel(r'data\merge_ori_{}.xlsx'.format(rq), index=False)
    add_df.to_excel(r'data\merge_add_{}.xlsx'.format(rq), index=False)


# 将真实数据和模拟数据进行融合
def merge_data(origin_path, parsed_path, mode=0):
    """
    :param origin_path: 基准数据，一般用drsu的真实数据
    :param parsed_path: 待融合数据，可以是任何其他数据
    :param mode: 融合模式。0代表全量融合，1代表增量融合
    :return:
    """
    tmp_df = pd.DataFrame()  # 记录原始数据第一帧的df
    all_df = pd.DataFrame()  # 记录所有数据融合后的df
    add_df = pd.DataFrame()  # 记录parsed_path数据融合后的df
    ori_list = glob(os.path.join(origin_path, '*.xlsx'))
    ori_list.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    if isinstance(parsed_path, list):
        pars_list = []
        for i in parsed_path:
            pars_list.append(glob(os.path.join(i, '*.xlsx')))
    elif isinstance(parsed_path, str):
        pars_list = glob(os.path.join(parsed_path, '*.xlsx'))
    else:
        logger.error('不支持的待融合文件格式')
        return
    for index in range(len(ori_list)):
        df_ori = pd.read_excel(ori_list[index])
        logger.info('开始处理文件：{}'.format(ori_list[index]))
        if index == 0:
            tmp_df = df_ori
            start_timestamp = df_ori.sort_values(by='timestamp').iloc[0].timestamp
            start_frame_number_del = round(start_timestamp * 10, 0) - 1
        df_ori['frame_number'] = (df_ori['timestamp'] * 10).round(0) - start_frame_number_del
        all_df = all_df.append(df_ori)
    all_df.to_excel(r'data\merge_data\merge_ori_{}.xlsx'.format(rq), index=False)
    end_id = int(os.path.basename(ori_list[0]).split('.')[0])
    per_num = end_id // len(pars_list)
    for index1 in range(len(pars_list)):
        if pars_list[index1].endswith('csv'):
            df_par = pd.read_csv(pars_list[index1])
        elif pars_list[index1].endswith('xlsx'):
            df_par = pd.read_excel(pars_list[index1])
        else:
            logger.error('不支持的类型')
            return
        df_par['frame_number'] = tmp_df['frame_number']
        df_par['timestamp'] = tmp_df['timestamp']
        track_id = random.randint(per_num * index1, per_num * (index1 + 1))
        df_par['id'] = track_id
        if index1 % 10 == 0:
            df_par['obj_type'] = random.choice([7, 8])
        else:
            df_par['obj_type'] = 6
        df_par['track_id'] = track_id
        df_par['local_timestamp'] = tmp_df.iloc[0].local_timestamp
        dir_path = r'data\merge_data\{}'.format(rq)
        check_path(dir_path)
        draw(df_par, os.path.join(dir_path, '{}.jpg'.format(index1)))
        df_par.to_csv(os.path.join(dir_path, '{}.csv'.format(index1)), index=False)
        logger.info('开始处理文件：{}, track_id:{}'.format(pars_list[index1], track_id))
        add_df = add_df.append(df_par)
        all_df = all_df.append(df_par)
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    all_df.to_excel(r'data\merge_data\merge_all_{}.xlsx'.format(rq), index=False)
    add_df.to_excel(r'data\merge_data\merge_add_{}.xlsx'.format(rq), index=False)


# 将真实数据和模拟数据进行融合
def merge_data_offset(origin_path, parsed_path, mode=0, offset=0):
    """
    :param origin_path: 基准数据，一般用drsu的真实数据
    :param parsed_path: 待融合数据，可以是任何其他数据
    :param mode: 融合模式。0代表全量融合，1代表增量融合
    :param offset: 两个融合数据之间的偏移量 单位为秒
    :return:
    """
    tmp_df = pd.DataFrame()  # 记录原始数据第一帧的df
    all_df = pd.DataFrame()  # 记录所有数据融合后的df
    add_df = pd.DataFrame()  # 记录parsed_path数据融合后的df
    ori_list = glob(os.path.join(origin_path, '*.xlsx'))
    ori_list.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    pars_list = glob(os.path.join(parsed_path, '*.xlsx'))
    pars_list.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))

    for index in range(len(ori_list)):
        df_ori = pd.read_excel(ori_list[index])
        logger.info('开始处理文件：{}'.format(ori_list[index]))
        if index == 0:
            tmp_df = df_ori
            start_timestamp = df_ori.sort_values(by='timestamp').iloc[0].timestamp
            start_frame_number_del = round(start_timestamp * 10, 0) - 1
            # break
        df_ori['frame_number'] = (df_ori['timestamp'] * 10).round(0) - start_frame_number_del
        all_df = all_df.append(df_ori)
    all_df.to_excel(r'data\merge_data\merge_ori_{}.xlsx'.format(rq), index=False)
    track_id = 50000
    for index1 in range(len(pars_list)):
        if pars_list[index1].endswith('csv'):
            df_par = pd.read_csv(pars_list[index1])
        elif pars_list[index1].endswith('xlsx'):
            df_par = pd.read_excel(pars_list[index1])
        else:
            logger.error('不支持的类型')
            return
        if index1 == 0:
            tmp_df_par = df_par
            start_timestamp_par = tmp_df_par.sort_values(by='timestamp').iloc[0].timestamp
            start_frame_number_del_par = round(start_timestamp_par * 10, 0) - 1
        df_par['frame_number'] = (df_par['timestamp'] * 10).round(0) - start_frame_number_del_par + offset * 10
        df_par['timestamp'] = df_par['timestamp'] - start_timestamp_par + start_timestamp + offset
        track_id += 1
        df_par['id'] = track_id
        df_par['local_timestamp'] = tmp_df.iloc[0].local_timestamp
        dir_path = r'data\merge_data_add\{}'.format(rq)
        check_path(dir_path)
        draw(df_par, os.path.join(dir_path, '{}.jpg'.format(index1)))
        df_par.to_csv(os.path.join(dir_path, '{}.csv'.format(index1)), index=False)
        logger.info('开始处理文件：{}, track_id:{}'.format(pars_list[index1], track_id))
        add_df = add_df.append(df_par)
        all_df = all_df.append(df_par)
    all_df.to_excel(r'data\merge_data\merge_all_{}.xlsx'.format(rq), index=False)
    add_df.to_excel(r'data\merge_data\merge_add_{}.xlsx'.format(rq), index=False)


# 将真实数据和模拟数据进行融合
def merge_data_offset_(origin_path, parsed_path, mode=0, offset=0):
    """
    :param origin_path: 基准数据，一般用drsu的真实数据
    :param parsed_path: 待融合数据，可以是任何其他数据
    :param mode: 融合模式。0代表全量融合，1代表增量融合
    :param offset: 两个融合数据之间的偏移量 单位为秒
    :return:
    """
    tmp_df = pd.DataFrame()  # 记录原始数据第一帧的df
    all_df = pd.DataFrame()  # 记录所有数据融合后的df
    add_df = pd.DataFrame()  # 记录parsed_path数据融合后的df
    ori_list = glob(os.path.join(origin_path, '*.xlsx'))
    ori_list.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    pars_list = glob(os.path.join(parsed_path, '*.xlsx'))
    pars_list.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))

    for index in range(len(ori_list)):
        df_ori = pd.read_excel(ori_list[index])
        logger.info('开始处理文件：{}'.format(ori_list[index]))
        if index == 0:
            # tmp_df = df_ori
            # start_timestamp = df_ori.sort_values(by='timestamp').iloc[0].timestamp
            # start_frame_number_del = round(start_timestamp * 10, 0) - 1
            start_frame_num = df_ori.loc[0, 'frame_number']
            # break
        df_ori['frame_number'] = [df_ori.loc[0, 'frame_number'] + i for i in range(df_ori.shape[0])]
        df_ori['timestamp'] = df_ori['frame_number']/10
        all_df = all_df.append(df_ori)
    all_df.to_excel(r'data\merge_data\merge_ori_{}.xlsx'.format(rq), index=False)
    track_id = 0
    for index1 in range(len(pars_list)):

        df_par = pd.read_excel(pars_list[index1])
        if df_par.center_x.mean() > 235220 and abs(df_par.loc[df_par.shape[0]-1, 'center_y'] - df_par.loc[0, 'center_y'])>50:
            continue
        # if index1 == 0:
        #     tmp_df_par = df_par
        #     start_timestamp_par = tmp_df_par.sort_values(by='timestamp').iloc[0].timestamp
        #     start_frame_number_del_par = round(start_timestamp_par * 10, 0) - 1

        df_par['frame_number'] = [df_par.loc[0, 'frame_number'] + start_frame_num + offset * 10 + i for i in range(df_par.shape[0]) ]
        df_par['timestamp'] = df_par['frame_number']/10
        track_id += 1
        df_par['id'] = track_id
        dir_path = r'data\merge_data_add\{}'.format(rq)
        check_path(dir_path)
        draw(df_par, os.path.join(dir_path, '{}.jpg'.format(index1)))
        df_par.to_csv(os.path.join(dir_path, '{}.csv'.format(index1)), index=False)
        logger.info('开始处理文件：{}, track_id:{}'.format(pars_list[index1], track_id))
        add_df = add_df.append(df_par)
        all_df = all_df.append(df_par)
    all_df.to_excel(r'data\merge_data\merge_all_{}.xlsx'.format(rq), index=False)
    add_df.to_excel(r'data\merge_data\merge_add_{}.xlsx'.format(rq), index=False)


def repair_drsu(drsu_data_dir, x_flag=True):
    csv_list = glob(os.path.join(drsu_data_dir, '*.csv'))
    for i in csv_list:
        Track_data = ParseDataFrame(i, x_flag)
        Track_data.parse_data()


def merge_add_drsu_data(file_path):
    csv_list = glob(os.path.join(file_path, '*.xlsx'))
    df = pd.DataFrame()
    for i in csv_list:
        df = df.append(pd.read_excel(i))
    # df = df.sort_values(by='frame_number')
    df = df.reset_index(drop=True)
    df.to_excel(r'data\merge_data\摆拍数据drsu2轨迹手动补全.xlsx', index=False)


if __name__ == '__main__':
    # drc2obs(r'data\origin_data\acu_status_info_history-qidi-mingxing-8.xlsx')
    # A=ACUTrackData(r'data\origin_data\acu_status_info_history-qidi-mingxing-round.xlsx')
    # A.parse_data()
    # split_by_id(r'data\origin_data\80s.xlsx')
    # split_by_id(r'data\origin_data\drsu03-0316-1914-1919-deal.xlsx')
    # A = ACUTrackData(r'data\origin_data\acu_status_info_history_6_round.xlsx', copy_num=40)
    # A.parse_data()
    #
    # origin_path = r'data\drsu_data\80s_drsu_split'
    # # parsed_path = [r'data\track_data\acu_status_info_history-qidi-mingxing-round_20',r'data\track_data\acu_status_info_history_6_round_40']
    # parsed_path = r'data\track_data\acu_status_info_history-qidi-mingxing-round_20'
    # merge_data(origin_path, parsed_path)

    # drsu_path = r'data\origin_data\traffic_report_obstacle_2d.xlsx'
    # # choose_type = [4, 6, 7, 8]/
    # choose_type = None
    # split_by_id(drsu_path, choose_type)

    # drsu_path = r'data\drsu_data\traffic_report_obstacle_2d-1-daoyan-drsu3-ok_drsu_split_4678'
    # repair_drsu(drsu_path)

    # drsu_add_path = r'data\drsu_data\摆拍数据drsu2_drsu_split_hand'
    # merge_add_drsu_data(drsu_add_path)

    # file_single = r'data\drsu_data\traffic_report_obstacle_2d-1-daoyan-drsu3-ok_drsu_split_4678\657.csv'
    # Track_data = ParseDataFrame(file_single)
    # Track_data.parse_data()

    # base_path = r'data\drsu_data\base_data'
    # drsu_add_path = r'data\drsu_data\traffic_report_obstacle_2d-1-daoyan-drsu3-ok_drsu_split_repaired'
    # merge_data(base_path, drsu_add_path)

    # drsu_path = r'data\origin_data\大屏第一幕全景.xlsx'
    # # choose_type = [4, 6, 7, 8]
    # choose_type = None
    # split_by_id(drsu_path, choose_type)

    # drsu_path = r'data\drsu_data\摆拍数据drsu2_drsu_split_4678'
    # repair_drsu(drsu_path, x_flag=False)

    # file_single = r'data\drsu_data\traffic_report_obstacle_2d-4-daoyan-drsu2-ok_drsu_split_678\767.xlsx'
    # Track_data = ParseDataFrame(file_single, x_flag=False)
    # Track_data.parse_data()
    # pass

    drsu_path = r'data\origin_data\merge_add_202104192020.xlsx'
    # choose_type = [6, 7, 8]
    choose_type = None
    split_by_id(drsu_path, choose_type)

    # base_path = r'data\drsu_data\new1748_drsu_split_repaired'
    # drsu_add_path = r'data\drsu_data\old__drsu_split'
    # merge_data_offset_(base_path, drsu_add_path, offset=0)

    # drsu_path = r'data\merge_data\merge_all_202103231429.xlsx'
    # # choose_type = [6, 7, 8]
    # choose_type = None
    # split_by_id(drsu_path, choose_type)

    #
    # frame_num = 550
    # data_path ='D:\\pythoncode\\code\\show_project\\data\\drsu_data\\traffic_report_obstacle_2d-1-daoyan-drsu3-ok_drsu_split_ori'
    # ori_path = 'D:\\pythoncode\\code\\show_project\\data\\drsu_data\\traffic_report_obstacle_2d-1-daoyan-drsu3-ok_drsu_split_678'
    # file_name = '630.xlsx'
    # df = pd.read_excel(os.path.join(ori_path, file_name))
    #
    # for i in range(frame_num):
    #     Series = df.iloc[df.shape[0]-1]
    #     Series.frame_number += 1
    #     Series.timestamp += 0.1
    #     df = df.append(Series)
    # draw(df, file=None, mode=1, show=True)
    # df.to_excel(os.path.join(data_path, file_name), index=False)
    #

    # data_path ='D:\\pythoncode\\code\\show_project\\data\\drsu_data\\traffic_report_obstacle_2d-1-daoyan-drsu3-ok_drsu_split_ori'
    # file_name = '630.xlsx'
    # file_name1 = '617.xlsx'
    # df_670 = pd.read_excel(os.path.join(data_path, file_name))
    # df_617 = pd.read_excel(os.path.join(data_path, file_name1))
    # start_frame_ = df_670.iloc[df_670.shape[0]-1].frame_number
    # df_add = df_617.loc[df_617.frame_number > start_frame_].sort_values(by='frame_number').iloc[0:540]
    # df_add.id = 630
    # df_add.obj_type = 6
    # df_add.loc[:, 'center_x'] = df_670.loc[df_670.shape[0]-1, 'center_x']
    # df_add.loc[:, 'center_y'] = df_670.loc[df_670.shape[0]-1, 'center_y']
    # df_670 = df_670.append(df_add)
    # df_670.to_excel(os.path.join(data_path,file_name),index=False)
    #
    # draw(df_670, mode=1, show=True)
    # df.obj_type = 3

    # data_path ='D:\\pythoncode\\code\\show_project\\data\\drsu_data\\traffic_report_obstacle_2d-1-daoyan-drsu3-ok_drsu_split_ori'
    # file_name = '670.csv'
    # file_name1 = '617.xlsx'
    # df_670 = pd.read_csv(os.path.join(data_path, file_name))
    # df_617 = pd.read_excel(os.path.join(data_path, file_name1))
    # start_frame_ = df_670.iloc[0].frame_number
    # df_add = df_617.loc[df_617.frame_number < start_frame_].sort_values(by='frame_number', ascending=False).iloc[0:200]
    # df_add = df_add.sort_values(by='frame_number')
    # df_add.id = 670
    # df_add.obj_type = 3
    # df_add.loc[:, 'center_x'] = df_670.loc[0, 'center_x']
    # df_add.loc[:, 'center_y'] = df_670.loc[0, 'center_y']
    # df_670 = df_add.append(df_670)
    # df_670.to_excel(os.path.join(data_path, '670.xlsx'), index=False)
    #
    # draw(df_670, mode=1, show=True)
    # df_670.obj_type = 3
    # plt.scatter()
