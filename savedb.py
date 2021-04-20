# -*- coding: utf-8 -*-
"""
@Project : show_project
@File    : savedb.py
@Author  : 王白熊
@Data    ： 2021/3/22 14:20
"""
# -*- coding:UTF-8 -*-

import pandas as pd
from Log import Logger
from sqlalchemy import create_engine

logger = Logger('excel_to_db').getlog()

import psycopg2
from io import StringIO


def psycopg2_function(df, table_name, host='10.10.10.42', port='5432', user='DRCRM', passwd='123456', db='DRCRM_ZHGS'):
    flag = False
    output = StringIO()
    df.to_csv(output, sep='\t', index=False, header=False)
    output1 = output.getvalue()

    conn = psycopg2.connect(host=host, port=port, user=user, password=passwd, dbname=db)
    cur = conn.cursor()

    # 判断表格是否存在，不存在则创建
    try:
        cur.execute("select to_regclass(" + "\'" + table_name + "\'" + ") is not null")
        rows = cur.fetchall()
    except Exception as e:
        rows = []
    if rows:
        data = rows
        flag = data[0][0]

    if flag != True:
        sql = f'''CREATE TABLE "public"."{table_name}" ( \
            "traffic_report_obstacle_2d_id" int8 NOT NULL DEFAULT nextval('traffic_report_obstacle_2d_traffic_report_obstacle_2d_id_seq'::regclass),
            "id" int4,
            "timestamp" float8,
            "center_x" float8,
            "center_y" float8,
            "center_z" float8,
            "length" float8,
            "width" float8,
            "height" float8,
            "obj_type" int4,
            "velocity_x" float8,
            "velocity_y" float8,
            "velocity_z" float8,
            "angular_velocity" float8,
            "acceleration_x" float8,
            "acceleration_y" float8,
            "acceleration_z" float8,
            "local_timestamp" timestamp(6),
            "frame_number" int8 NOT NULL DEFAULT 0,
            "theta" float8,
            "track_id" int4,
            "lane_ids" varchar(255) COLLATE "pg_catalog"."default",
            "connection_ids" varchar(255) COLLATE "pg_catalog"."default",
            "det_confidence" float8,
            "obs_drsuids" varchar COLLATE "pg_catalog"."default",
            "is_valid" bool
            )
            ;'''
    cur.execute(sql)
    conn.commit()
    logger.info(f'create table {table_name} success. ')

    # 获取列名
    columns = list(df)
    cur.copy_from(StringIO(output1), table_name, null='', columns=columns)
    conn.commit()
    cur.close()
    conn.close()


def excel_to_DB(data, host='10.10.10.42', port='5432', user='DRCRM', passwd='123456', db='DRCRM_ZHGS',
                table_name='traffic_obstacle_2d_00'):
    """
	表数据存入DB
	:param host: 数据库地址
	:param port: 数据库端口
	:param user: 数据库账号
	:param passwd: 账号密码
	:param db:   数据库名
	:param table_name: 存入数据库的表名
	"""
    if isinstance(data, pd.core.frame.DataFrame):
        dataframe = data
    elif isinstance(data, str) and data.endswith('.xlsx'):
        dataframe = pd.read_excel(data, engine='openpyxl')
    else:
        logger.error('不支持的类型')
    # dataframe = pd.DataFrame(data_excel)
    logger.info('read excel success. ')

    # DataFrame.to_sql方法
    # engine = create_engine(f'postgresql+psycopg2://{user}:{passwd}@{host}:{port}/{db}', encoding='utf8')
    # logger.info(f'create_engine() connect DB "{host}--{db}" success')
    # data_dataframe.to_sql(table_name, con=engine, if_exists='replace', index=False)

    # copy_from 方法
    psycopg2_function(dataframe, table_name, host, port, user, passwd, db)
    logger.info('write db success. ')


if __name__ == '__main__':
    import time

    # ticks_min = time.strftime("%Y-%m-%d %H:%M", time.localtime())
    # ticks_min_timestramp = int(time.mktime(time.strptime(ticks_min, "%Y-%m-%d %H:%M")))
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    table_name = f'traffic_report_obstacle_2d_wth_{rq}'


    excel_to_DB(r'data\merge_data\merge_add_drsu.xlsx',
                host='10.10.10.42', port='5432', user='DRCRM', passwd='123456', db='DRCRM_ZHGS_3',
                table_name=table_name)
