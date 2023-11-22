# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pandas as pd


def divide_data(path):
    # 读取数据
    dataface = pd.read_csv(path)
    # 提取label数据
    dataface_y = dataface[['label']]
    # 提取feature（即像素）数据
    dataface_x = dataface[['feature']]
    # 将label写入label.csv
    dataface_y.to_csv('cnn_label.csv', index=False, header=False)
    # 将feature数据写入data.csv
    dataface_x.to_csv('cnn_data.csv', index=False, header=False)


def face_picture(path):
    # 读取像素数据
    data = np.loadtxt('cnn_data.csv')

    # 按行取数据
    for i in range(data.shape[0]):
        face_array = data[i, :].reshape((48, 48))  # 循环读取第i行图像数据，利用reshape更改为48*48大小图片
        cv2.imwrite(path + '\\' + '{}.jpg'.format(i), face_array)  # opencv保存图片


if __name__ == '__main__':
    divide_data('cnn_train.csv')  # 原始数据集路径
    face_picture('face')  # 图像存放路径
