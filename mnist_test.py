#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/02/05 13:52
# @Author   : WanDaoYi
# @FileName : mnist_test.py
# ============================================

from datetime import datetime
import tensorflow as tf
import numpy as np
from config import cfg
from core.common import Common


class MnistTest(object):

    def __init__(self):

        self.common = Common()
        # 读取数据和 维度
        self.mnist_data, self.n_feature, self.n_label = self.common.read_data()

        # ckpt 模型
        self.test_ckpt_model = cfg.TEST.CKPT_MODEL_SAVE_PATH
        print("test_ckpt_model: {}".format(self.test_ckpt_model))

        # tf.reset_default_graph()
        # 创建设计图
        with tf.name_scope(name="input"):
            self.x = tf.placeholder(dtype=tf.float32, shape=(None, self.n_feature), name="input_data")
            self.y = tf.placeholder(dtype=tf.float32, shape=(None, self.n_label), name="input_labels")

        # 获取最后一层的输出
        self.logits = self.common.dnn_layer(self.x, self.n_label, 1)

    # 使用 ckpt 模型测试
    def do_ckpt_test(self):

        saver = tf.train.Saver()

        with tf.Session() as sess:
            # 重载模型
            saver.restore(sess, self.test_ckpt_model)

            # 预测
            output = self.logits.eval(feed_dict={self.x: self.mnist_data.test.images})

            # 将 one-hot 预测值转为 数字
            y_perd = np.argmax(output, axis=1)
            print("预测值: {}".format(y_perd[: 5]))

            # 真实值
            y_true = np.argmax(self.mnist_data.test.labels, axis=1)
            print("真实值: {}".format(y_true[: 5]))

        pass


if __name__ == "__main__":
    # 代码开始时间
    start_time = datetime.now()
    print("开始时间: {}".format(start_time))

    demo = MnistTest()
    # 使用 ckpt 模型测试
    demo.do_ckpt_test()

    # 代码结束时间
    end_time = datetime.now()
    print("结束时间: {}, 训练模型耗时: {}".format(end_time, end_time - start_time))
