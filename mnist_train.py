#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/02/05 13:52
# @Author   : WanDaoYi
# @FileName : mnist_train.py
# ============================================

from datetime import datetime
import tensorflow as tf
from config import cfg
from core.common import Common
import numpy as np


class MnistTrain(object):

    def __init__(self):
        # 模型保存路径
        self.model_save_path = cfg.TRAIN.MODEL_SAVE_PATH
        self.log_path = cfg.LOG.LOG_SAVE_PATH

        self.learning_rate = cfg.TRAIN.LEARNING_RATE
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.n_epoch = cfg.TRAIN.N_EPOCH

        self.common = Common()
        # 读取数据和 维度
        self.mnist_data, self.n_feature, self.n_label = self.common.read_data()

        # 创建设计图
        with tf.name_scope(name="input_data"):
            self.x = tf.placeholder(dtype=tf.float32, shape=(None, self.n_feature), name="input_data")
            self.y = tf.placeholder(dtype=tf.float32, shape=(None, self.n_label), name="input_labels")

        with tf.name_scope(name="input_shape"):
            # 784维度变形为图片保持到节点
            # -1 代表进来的图片的数量、28，28是图片的高和宽，1是图片的颜色通道
            image_shaped_input = tf.reshape(self.x, [-1, 28, 28, 1])
            tf.summary.image('input', image_shaped_input, self.n_label)

        self.keep_prob_dropout = cfg.TRAIN.KEEP_PROB_DROPOUT
        self.keep_prob = tf.placeholder(tf.float32)

        # 获取最后一层的输出
        self.logits = self.common.dnn_layer(self.x, self.n_label, self.keep_prob_dropout)

        # self.config = tf.ConfigProto()
        # self.config.gpu_options.allow_growth = True
        # self.sess = tf.Session(config=self.config)
        self.sess = tf.InteractiveSession()
        # 保存训练模型
        self.saver = tf.train.Saver()
        pass

    # 灌入数据
    def feed_dict(self, train_flag=True):
        # 训练样本
        if train_flag:
            # 获取下一批次样本
            x_data, y_data = self.mnist_data.train.next_batch(self.batch_size)
            keep_prob = self.keep_prob_dropout
            pass
        # 验证样本
        else:
            x_data, y_data = self.mnist_data.test.images, self.mnist_data.test.labels
            keep_prob = 1.0
            pass
        return {self.x: x_data, self.y: y_data, self.keep_prob: keep_prob}
        pass

    # 训练
    def do_train(self):

        # 计算loss 损失
        with tf.name_scope("train_loss"):
            # softmax_cross_entropy_with_logits 只会给 one-hot 编码
            # sparse_softmax_cross_entropy_with_logits 只会给没有 one-hot 编码，使用的会给0-9分类号
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
            loss = tf.reduce_mean(cross_entropy, name="loss")
            tf.summary.scalar("cross_entropy", loss)
            pass

        # 构建优化器
        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            training_op = optimizer.minimize(loss=loss)

        # 比对正确的率
        # 只处理没 one-hot 编码，获取logits里面最大的那1位和y比较类别好是否相同，返回True或者False一组值
        # correct = tf.nn.in_top_k(logits, y, 1) 处理 one-hot 编码后的 y
        with tf.name_scope("accuracy"):
            correct = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
            acc = tf.reduce_mean(tf.cast(correct, tf.float32))
            tf.summary.scalar("accuracy", acc)

        # 因为我们之前定义了太多的tf.summary汇总操作，逐一执行这些操作太麻烦，
        # 使用tf.summary.merge_all()直接获取所有汇总操作，以便后面执行
        merged = tf.summary.merge_all()

        # 定义两个tf.summary.FileWriter文件记录器再不同的子目录，分别用来存储训练和测试的日志数据
        # 同时，将Session计算图sess.graph加入训练过程，这样再TensorBoard的GRAPHS窗口中就能展示
        train_writer = tf.summary.FileWriter(self.log_path + 'train', self.sess.graph)
        test_writer = tf.summary.FileWriter(self.log_path + 'test')

        # 构建初始化变量
        init_variable = tf.global_variables_initializer()

        self.sess.run(init_variable)

        test_acc = None

        for epoch in range(self.n_epoch):
            # 获取总样本数量
            batch_number = self.mnist_data.train.num_examples
            # 获取总样本一共几个批次
            size_number = int(batch_number / self.batch_size)

            for number in range(size_number):
                summary, _ = self.sess.run([merged, training_op], feed_dict=self.feed_dict())
                # 第几次循环
                i = epoch * size_number + number + 1
                train_writer.add_summary(summary, i)
                pass

                if number == size_number - 1:
                    # 获取下一批次样本
                    x_batch, y_batch = self.mnist_data.train.next_batch(self.batch_size)
                    acc_train = acc.eval(feed_dict={self.x: x_batch, self.y: y_batch})
                    print("acc_train: {}".format(acc_train))

            # 测试
            output = self.logits.eval(feed_dict={self.x: self.mnist_data.test.images})
            y_perd = np.argmax(output, axis=1)
            print("y_perd: {}".format(y_perd[: 5]))
            y_true = np.argmax(self.mnist_data.test.labels, axis=1)
            print("y_true: {}".format(y_true[: 5]))

            # 验证 方法一
            acc_test = acc.eval(feed_dict={self.x: self.mnist_data.test.images,
                                           self.y: self.mnist_data.test.labels})

            print("epoch: {}, acc_test: {}".format(epoch + 1, acc_test))

            # 验证 方法二 两个方法，随便挑一个都可以的。
            test_summary, acc_test_2 = self.sess.run([merged, acc], feed_dict=self.feed_dict(False))
            print("epoch: {}, acc_test_2: {}".format(epoch + 1, acc_test_2))
            test_writer.add_summary(test_summary, epoch + 1)

            test_acc = acc_test

        save_path = self.model_save_path + "acc={:.6f}".format(test_acc) + ".ckpt"
        # 保存模型
        self.saver.save(self.sess, save_path, global_step=self.n_epoch)

        train_writer.close()
        test_writer.close()
        pass


if __name__ == "__main__":
    # 代码开始时间
    start_time = datetime.now()
    print("开始时间: {}".format(start_time))

    demo = MnistTrain()
    demo.do_train()

    # 代码结束时间
    end_time = datetime.now()
    print("结束时间: {}, 训练模型耗时: {}".format(end_time, end_time - start_time))


