# _*_ coding: utf-8 _*_
"""
# @Time : 8/27/2021 3:15 PM
# @Author : byc
# @File : log.py
# @Description :
"""
import logging
import os


class Logger:
    def __init__(self, path_log):
        log_name = os.path.basename(path_log)
        self.log_name = log_name if log_name else 'root'
        self.out_path = path_log

        log_dir = os.path.dirname(path_log)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def init_logger(self):
        logger = logging.getLogger(self.log_name)
        logger.setLevel(level=logging.INFO)

        # file handler
        file_handler = logging.FileHandler(self.out_path, 'w')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger