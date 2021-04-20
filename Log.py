# -*- coding: utf-8 -*-

"""
封装log方法

"""

import logging
import os
import time
import sys


def create_file(filename):
    path = filename[0:filename.rfind('/')]
    if not os.path.isdir(path):
        os.makedirs(path)
    if not os.path.isfile(filename):
        fd = open(filename, mode='w', encoding='utf-8')
        fd.close()
    else:
        pass


class Logger(object):
    def __init__(self, logger_name):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)

        path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        rq = time.strftime('%Y%m%d', time.localtime(time.time()))
        log_path = path + '/Log/'
        log_file = log_path + rq + '.log'
        err_file = log_path + rq + 'err.log'
        create_file(log_file)
        create_file(err_file)

        # 定制输出格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        handler = logging.FileHandler(log_file, encoding='utf-8', mode='a+')  # 输出到log文件
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)

        err_handler = logging.FileHandler(err_file, encoding='utf-8', mode='a+')  # 输出到err_log文件
        err_handler.setLevel(logging.WARNING)
        err_handler.setFormatter(formatter)

        ch = logging.StreamHandler(sys.stdout)  # 输出到控制台
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)

        self.logger.addHandler(handler)
        self.logger.addHandler(err_handler)
        self.logger.addHandler(ch)

    def getlog(self):
        return self.logger


if __name__ == "__main__":
    a = Logger('wth').getlog()
    a.debug("This is debug message")
    a.info("This is info message")
    a.warning("This is warning message")
    a.error("This is error")
    a.critical("This is critical message")

