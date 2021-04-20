# -*- coding: utf-8 -*-
"""
@Project : analydata
@File    : constant.py
@Author  : 王白熊
@Data    ： 2020/10/30 9:30
"""


class _const:
    class ConstError(TypeError):
        pass

    class ConstCaseError(ConstError):
        pass

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise self.ConstError("Can't change const.%s" % name)
        if not name.isupper():
            raise self.ConstCaseError('const name "%s" is not all supercase' % name)

        self.__dict__[name] = value


const = _const()
const.MY_CONSTANT = "CHINA"
const.MY_SECOND_CONSTANT = "RUSSIA"

const.ANGEL_VER_VALUE = 57.3  # tan转化为角度的值，不用改
const.VOLUME_BUS = 5.2  # 小巴车的体积，宽*高
const.VOLUME_CAR = 3  # 小巴车的体积，宽*高
const.OBJ_TYPE_BUS = 8
const.OBJ_TYPE_CAR = 6
const.OBJ_TYPE_PER = 3
const.LIST_COLOR = ['red', 'blue', 'green', 'cyan', 'magenta', 'orange', 'darkred', 'black']

const.ANGEL_THRESHOLD = 20  # 三点判断法中第一条线段和第二条线段差值范围，y方向速度只有
const.VELOCITY_THRESHOLD = 1  # 判断静止物体允许的速度偏差
const.CENTER_THRESHOLD_X = 3  # 判断静止物体允许的纵向坐标偏差
const.CENTER_THRESHOLD_Y = 1  # 判断静止物体允许的横向坐标偏差
const.DIS_THRESHOLD = 20  # 判断直行车辆主行进方向大于20米
const.LOSS_FRAME_THRESHOLD = 10  # 是否严重丢帧判断，大于设置值认为严重丢帧
const.R_SQUARE_THRESHOLD = 0.1  # 轨迹直线拟合拟合优度允许的差值

const.CAR_THRESHOLD = 20  # 车辆限速72km/h
const.CAR_THRESHOLD1 = 5  # 车辆垂直于行进方向的速度
const.VOLUME_THRESHOLD = 7  # 体积允许的误差


const.MATCH_ALL = 0  # 障碍物符合场景特征
const.MATCH_FRONT = 1  # 障碍物符合前部分场景特征
const.MATCH_BACK = 2  # 障碍物符合后部分场景特征
const.MATCH_VAGUE = 3  # 非严格符合
const.MATCH_NOT = 4  # 障碍物不符合场景特征
const.MATCH_TYPE_MAIN = 0  # 主类型匹配
const.MATCH_TYPE_BACK_UP = 1  # 备类型匹配
const.MATCH_TYPE_LIKE = 2  # 相似类型匹配
const.MATCH_TYPE_NOT = 5  # 类型不匹配

const.TRACK_STATIC = 0  # 轨迹类型:静止
const.TRACK_STRAIGHT = 1  # 轨迹类型:直行
const.TRACK_STRAIGHT_VERTICAL = 2  # 轨迹类型:垂直于摄像头方向直行
const.TRACK_STRAIGHT_FRONT = 3  # 轨迹类型:前半段直行
const.TRACK_STRAIGHT_BACK = 4  # 轨迹类型:后半段直行
const.TRACK_UNDEFINED = 5  # 轨迹类型:未定义

# 此处定义可根据实际情况进行修改
const.BASE_SEARCH_DISTANCE = 12  # 基础搜寻范围，静止状态下在以acu上报坐标为中心，x，y差距该值范围内查找目标轨迹
const.SEARCH_DISTANCE_RATE = 0.3  # 搜索范围随着drsu与acu距离增大而呈现比例正大
const.BASE_SEARCH_SPEED = 1  #
const.SEARCH_SPEED_RATE = 0.2  # 查找与acu上报数据接近的目标估计 范围值为 acu数组的正负 1+value*rate
const.VARIANCE_CENTER = 1  # 坐标的方差范围，小于该值认定为静止
const.VARIANCE_VELOCITY = 1  # 速度的方差范围，小于该值认定为静止
const.COORDINATE_DIFF = 7  # drsu坐标与acu坐标的差值范围f
const.CENTER_DRSU_3 = [234788, 3344684]  # drsu3横杆坐标
const.CSV_SIZE_MIN = 2048  # csv文件的最小大小，小于该值的视为无效数据不处理
# drsu3车辆坐标
const.LOAD_ANGLE = 2.31  # drsu3 道路与utm坐标系x轴夹角为2.31度
const.LOAD_VALUE = -0.040340432670925463
const.CENTER_ACU = [(234839.8, 3344678.2), (234857.6, 3344677.8), (234874.2, 3344677.3), (234888.7, 3344676.8),
                    (234900.9, 3344676.5), (234910.4, 3344675.9), (234920.1, 3344675.3), (234930.8, 3344674.8),
                    (234940.6, 3344674.5), (234950.8, 3344673.8), (234960.1, 3344673.4), (234970.2, 3344673.1),
                    (234980.2, 3344672.7), (234990.8, 3344671.8), (234500.4, 3344671.8), (234881.0, 3344691.0),
                    (234870.6, 3344691.5), (234860.6, 3344692.0), (234850.4, 3344692.3), (234840.4, 3344692.7),
                    (234830.4, 3344693.2), (234819.8, 3344693.7)]

# ACU 部分
const.ACU_DELAY_TIME = 25.82


# DRSU 部分
const.DRSU_USER_NAME = 'broadxt'
const.DRSU_PASSWORD = 'broadxt333'
const.DRSU_HOST = '172.16.7.60'
const.DRSU_PORT = 22
const.DRSU_PATH = '/dr/drsu_sff'  # drsu所在路径
const.SCENARIO_PATH = '/dr/drsu_sff/drsu_data/test_scenario'  # 文件解压路径路径
const.TAR_PATH = '/home/broadxt/data4t/share/xiaoshan_drsu03_org_data/static_3_position22_1031'  # 压缩文件所在位置
