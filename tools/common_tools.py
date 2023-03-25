# -*- coding: utf-8 -*-
"""
# @file name  : common_tools.py
# @brief      : 通用函数
"""

from . network.cheapNet import CheapNet
import torch
import os
import logging
import sys
from . metrics import *


def get_cheapnet(device, groups=64, slope=0.2, vis_model=False, path_state_dict=None):
    """
    创建模型，加载参数
    :param device: 运算设备
    :param groups: 分组卷积的分组数
    :param slope: leaky_relu参数
    :param vis_model: 是否打印模型结构
    :param path_state_dict:
    :return: 预训练模型
    """
    model = CheapNet(groups=groups, slope=slope)  # 创建模型结构

    if path_state_dict:
        pretrained_state_dict = torch.load(path_state_dict)  # 读取预训练模型
        model.load_state_dict(pretrained_state_dict['CheapNet'])  # 将预训练模型载入模型

    model.eval()  # 开启验证模式

    if vis_model:  # 是否打印模型结构
        from torchinfo import summary
        summary(model, input_size=(1, 1, 512, 512), device="cpu")

    model.to(device)  # 将模型推至运算设备
    return model


class CheapNetTrainer(object):

    @staticmethod
    def train(data_loader, model, loss_f, optimizer, epoch_id, device, max_epoch, logger):
        """
        每次传入一个epoch的数据进行模型训练
        :param data_loader: 训练集加载器
        :param model: 模型
        :param loss_f: 损失函数
        :param optimizer: 优化器
        :param epoch_id: 第几个epoch
        :param device: 运算设备
        :param max_epoch: 最大训练轮数
        :param logger: 日志
        :return: 平均loss
        """
        model.train()  # 开启模型训练模式

        loss_avg = []  # 平均loss
        for i, data in enumerate(data_loader):  # 迭代训练集加载器,得到iteration和相关图像data

            x, target = data  # 通过data得到图像数据
            x = x.to(device)  # 传入运算设备
            target = target.to(device)  # 传入运算设备
            y = model(x)  # 载入模型,得到预测值

            optimizer.zero_grad()  # 优化器梯度归零
            loss = loss_f(y, target)  # 计算每个预测值与x的损失
            loss.backward()  # 反向传播,计算梯度
            optimizer.step()  # 更新梯度

            loss_avg.append(loss.item())  # 记录每次的loss值

            logger.info(f'Train | Epoch[{epoch_id + 1:0>3}/{max_epoch:0>3}] '
                        f'Iteration[{i + 1:0>3}/{len(data_loader):0>3}] '
                        f'Train loss: {np.mean(loss_avg):.4f}')

        return np.mean(loss_avg)

    @staticmethod
    def valid(data_loader, model, loss_f, epoch_id, device, max_epoch, logger):
        """
        模型验证
        :param data_loader: 验证集加载器
        :param model: 模型
        :param loss_f: 损失函数
        :param epoch_id: 第几个epoch
        :param device: 运算设备
        :param max_epoch: 最大训练轮数
        :param logger: 日志
        :return: 平均loss
        """
        model.eval()  # 模型验证模式
        running_metrics_val = RunningScore(2)  # 创建2*2的混淆矩阵

        loss_avg = []  # 平均loss
        for i, data in enumerate(data_loader):  # 迭代验证集加载器,得到iteration和相关data
            running_metrics_val.reset()  # 初始化混淆矩阵

            x, target = data  # 通过data得到图像数据和对应的label
            x = x.to(device)  # 传入运算设备
            target = target.to(device)  # 传入运算设备

            y = model(x)  # 载入模型,得到预测值

            loss = loss_f(y, target)  # 计算loss
            loss_avg.append(loss.item())  # 记录每次的loss值

            logger.info(f'Valid | Epoch[{epoch_id + 1:0>3}/{max_epoch:0>3}] '
                        f'Iteration[{i + 1:0>3}/{len(data_loader):0>3}] '
                        f'Valid loss: {np.mean(loss_avg):.4f}')

            # 计算混淆矩阵以及相关指标
            predict = torch.ge(y.detach(), 0.5).squeeze().cpu().numpy()  # [4,512,512]
            label = torch.eq(target.detach().int(), 1).squeeze().cpu().numpy()  # [4,512,512]
            running_metrics_val.update(label, predict)  # 更新混淆矩阵

        metrics = running_metrics_val.get_scores()
        valid_miou = metrics[0]['mIou']
        valid_acc = metrics[0]['pixel_acc']
        valid_class_acc = metrics[0]['class_acc']
        logger.info(f'Valid | Epoch[{epoch_id + 1:0>3}/{max_epoch:0>3}] MIou: {valid_miou}')
        logger.info(f'Valid | Epoch[{epoch_id + 1:0>3}/{max_epoch:0>3}] Pixel Accuracy: {valid_acc}')
        logger.info(f'Valid | Epoch[{epoch_id + 1:0>3}/{max_epoch:0>3}] Mean Pixel Accuracy: {valid_class_acc}')

        return np.mean(loss_avg), valid_miou


def get_logger(log_dir, log_name):
    log_file = os.path.join(log_dir, log_name)

    # 创建log
    logger = logging.getLogger('train')  # log初始化
    logger.setLevel(logging.INFO)  # 设置log级别, INFO是程序正常运行时输出的信息

    # Formatter 设置日志输出格式
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # StreamHandler 日志输出1 -> 输出到控制台
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # FileHandler 日志输出2 -> 保存到文件
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
