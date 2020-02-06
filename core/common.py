#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/02/05 14:09
# @Author   : WanDaoYi
# @FileName : common.py
# ============================================

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from config import cfg
import numpy as np


class Common(object):

    def __init__(self):
        # 数据路径
        self.data_file_path = cfg.COMMON.DATA_PATH

        self.hidden_01 = cfg.COMMON.HIDDEN_1
        self.hidden_02 = cfg.COMMON.HIDDEN_2
        self.hidden_03 = cfg.COMMON.HIDDEN_3

        pass

    # 读取数据
    def read_data(self):
        # 数据下载地址: http://yann.lecun.com/exdb/mnist/
        mnist_data = input_data.read_data_sets(self.data_file_path, one_hot=True)
        train_image = mnist_data.train.images
        train_label = mnist_data.train.labels
        _, n_feature = train_image.shape
        _, n_label = train_label.shape

        return mnist_data, n_feature, n_label

    # bn 操作
    def layer_bn(self, input_data, is_training, name, momentum=0.999, eps=1e-3):
        """
        :param inputdata: 输入数据
        :param is_training: 是否是训练 ，True 为训练
        :param name: 名字
        :param momentum: 动量因子
        :param eps:
        :return:
        """

        return tf.layers.batch_normalization(inputs=input_data, training=is_training,
                                             name=name, momentum=momentum,
                                             epsilon=eps)

    # dropout 处理
    def deal_dropout(self, hidden_layer, keep_prob):
        with tf.name_scope("dropout"):
            tf.summary.scalar('dropout_keep_probability', keep_prob)
            dropped = tf.nn.dropout(hidden_layer, keep_prob)
            tf.summary.histogram('dropped', dropped)
            return dropped
        pass

    # 神经网络层
    def neural_layer(self, x, n_neuron, name, activation=None):
        # 包含所有的计算节点对于这一层, name_scope 可写可不写
        with tf.name_scope(name=name):
            n_input = int(x.get_shape()[1])
            stddev = 2 / np.sqrt(n_input)

            # 这层里面的w可以看成是二维数组，每个神经元对于一组w参数
            # truncated normal distribution 比 regular normal distribution的值小
            # 不会出现任何大的权重值，确保慢慢的稳健的训练
            # 使用这种标准方差会让收敛快
            # w参数需要随机，不能为0，否则输出为0，最后调整都是一个幅度没意义
            with tf.name_scope("weights"):
                init_w = tf.truncated_normal((n_input, n_neuron), stddev=stddev)
                w = tf.Variable(init_w, name="weight")
                self.variable_summaries(w)

            with tf.name_scope("biases"):
                b = tf.Variable(tf.zeros([n_neuron]), name="bias")
                self.variable_summaries(b)
            with tf.name_scope("wx_plus_b"):
                z = tf.matmul(x, w) + b
                tf.summary.histogram('pre_activations', z)

            if activation == "relu":
                activation_result = tf.nn.relu(z)
                tf.summary.histogram('activation_result', activation_result)
                return activation_result
            else:
                return z

    def dnn_layer(self, x, n_label, keep_prob):
        # 隐藏层
        with tf.name_scope("dnn"):
            # 这里要注意矩阵匹配
            x_scale = self.layer_bn(x, is_training=True, name="x_bn")
            hidden_1 = self.neural_layer(x_scale, self.hidden_01, "hidden_01", activation="relu")
            dropped_hidden_1 = self.deal_dropout(hidden_1, keep_prob)

            hidden_scale_1 = self.layer_bn(dropped_hidden_1, is_training=True, name="hidden_bn_1")
            hidden_2 = self.neural_layer(hidden_scale_1, self.hidden_02, "hidden_02", activation="relu")
            dropped_hidden_2 = self.deal_dropout(hidden_2, keep_prob)

            hidden_scale_2 = self.layer_bn(dropped_hidden_2, is_training=True, name="hidden_bn_2")
            hidden_3 = self.neural_layer(hidden_scale_2, self.hidden_03, "hidden_03", activation="relu")
            dropped_hidden_3 = self.deal_dropout(hidden_3, keep_prob)

            hidden_scale_3 = self.layer_bn(dropped_hidden_3, is_training=True, name="hidden_bn_3")
            logits = self.neural_layer(hidden_scale_3, n_label, name="logits")

            return logits

    # 定义Variable变量的数据汇总函数，我们计算出变量的mean、stddev、max、min
    # 对这些标量数据使用tf.summary.scalar进行记录和汇总
    # 使用tf.summary.histogram直接记录变量var的直方图数据
    def variable_summaries(self, param):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(param)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(param - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(param))
            tf.summary.scalar('min', tf.reduce_min(param))
            tf.summary.histogram('histogram', param)

