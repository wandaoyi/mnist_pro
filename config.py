#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/02/05 13:51
# @Author   : WanDaoYi
# @FileName : config.py
# ============================================

from easydict import EasyDict as edict
import os


__C = edict()

cfg = __C

# common options 公共配置文件
__C.COMMON = edict()
# windows 获取文件绝对路径, 方便 windows 在黑窗口 运行项目
__C.COMMON.BASE_PATH = os.path.abspath(os.path.dirname(__file__))
# # 获取当前窗口的路径, 当用 Linux 的时候切用这个，不然会报错。(windows也可以用这个)
# __C.COMMON.BASE_PATH = os.getcwd()

__C.COMMON.DATA_PATH = os.path.join(__C.COMMON.BASE_PATH, "dataset")

# 隐藏层的输出节点数
__C.COMMON.HIDDEN_1 = 300
__C.COMMON.HIDDEN_2 = 100
__C.COMMON.HIDDEN_3 = 50


# 训练配置
__C.TRAIN = edict()

# 学习率
__C.TRAIN.LEARNING_RATE = 0.01
# batch_size
__C.TRAIN.BATCH_SIZE = 32
# 迭代次数
__C.TRAIN.N_EPOCH = 10

# 模型保存路径, 使用相对路径，方便移植
__C.TRAIN.MODEL_SAVE_PATH = "./checkpoint/model_"
# dropout 的持有量，0.7 表示持有 70% 的节点。
__C.TRAIN.KEEP_PROB_DROPOUT = 0.7


# 测试配置
__C.TEST = edict()

# 测试模型保存路径
__C.TEST.CKPT_MODEL_SAVE_PATH = "./checkpoint/model_acc=0.961400.ckpt-10"


# 日志配置
__C.LOG = edict()
# 日志保存路径，后面会接上  train 或 test: 如 mnist_log_train
__C.LOG.LOG_SAVE_PATH = "./logs/mnist_log_"

