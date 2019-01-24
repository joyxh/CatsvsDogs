# new-data 8.7

import numpy as np
import os
from sklearn.utils import shuffle
import cv2
import tensorflow as tf
import time
import matplotlib.pyplot as plt

Mean = [106.08114841,115.83754386,124.2740883]
image_dir = 'Data/train'
tv_proportion = 0.15
batch_size = 16
index = 0
epoch_completed = 0
index1 = 0
epoch_completed1 = 0

def get_test_and_val(image_dir, tv_proportion):
    print('start parting dataset...')
    path_cats = []
    path_dogs = []
    label_cats = []
    label_dogs = []
    for file in os.listdir(image_dir):
        #用'.'将file切片
        animal = file.split(sep='.')
        if animal[0] == 'cat':
            label_cats.append(0)
            path_cats.append(os.path.join(image_dir, file))
        else:
            label_dogs.append(1)
            path_dogs.append(os.path.join(image_dir, file))

    l = len(label_cats)+len(label_dogs)
    s = int(tv_proportion * l/2)
    #取前s个猫和前s个狗作为测试集，并打乱顺序
    test_path = np.hstack((path_cats[:s],path_dogs[:s]))
    test_label = np.hstack((label_cats[:s],label_dogs[:s]))
    test_path = list(test_path)
    test_label = list(test_label)
    test_path, test_label = shuffle(test_path, test_label)

    #取第s：2s的猫和第s：2s狗作为验证集
    val_path = np.hstack((path_cats[s:2*s],path_dogs[s:2*s]))
    val_label = np.hstack((label_cats[s:2*s],label_dogs[s:2*s]))
    val_path = list(val_path)
    val_label = list(val_label)
    val_path, val_label = shuffle(val_path, val_label)
    #剩余部分为新的训练集
    train_path = np.hstack((path_cats[2*s:],path_dogs[2*s:]))
    train_label = np.hstack((label_cats[2*s:],label_dogs[2*s:]))
    train_path = list(train_path)
    train_label = list(train_label)
    train_path, train_label = shuffle(train_path, train_label)
    print('len(test)=%d ,len(val)=%d ,len(train)=%d  '%(len(test_label),len(val_label),len(train_label)))
    return test_label, test_path , val_path, val_label, train_path, train_label



def get_next_batch(path, label, index, epoch_completed):

    start = index
    num = len(label)

    # print('before  \n',path[:5],'\n',path[16:21])
    # if epoch_completed == 0 and start == 0:
    #     #打乱数据
    #     path, label = shuffle(path, label)
    # print('after  \n',path[:5],'\n',path[16:21])

    #当数据集被遍历完一次
    if start + batch_size > num:
        #数据集遍历代数+1
        epoch_completed +=1
        rest_num = num - start
        path_rest = path[start:num]
        label_rest = label[start:num]
        #进行打乱
        path, label = shuffle(path, label)
        start = 0
        index = batch_size - rest_num
        end = index
        path_new = path[start:end]
        label_new= label[start:end]
        return path_rest + path_new, label_rest +label_new, index, epoch_completed
    else:
        index += batch_size
        end = index
        return path[start:end], label[start:end], index, epoch_completed

def get_batch_data(path, label, index, epoch_completed):
    image_paths, labels, index, epoch_completed = get_next_batch(path, label, index, epoch_completed)
    images = []
    for path in image_paths:
        image = cv2.imread(path)
        image = cv2.resize(image, (224, 224), 0, 0, cv2.INTER_LINEAR)
        image = image - Mean
        images.append(image)
    images = np.array(images, dtype=float)
    labels = np.array(labels)

    #转换成one-hot
    # targets = np.array(labels).reshape(-1)
    # labels = np.eye(2)[targets]

    return images, labels, index, epoch_completed


if __name__ == '__main__':
    test_label, test_path, val_path, val_label, train_path, train_label = get_test_and_val(image_dir, tv_proportion)
    # print(train_path[17486:17488])
    # print(train_path[17498:])
    # print(val_path[3742:3744])
    # print(val_path[3748:])

    # print('train_total')
    # print(train_path[:4])
    # print(train_path[16:20])
    # print('val_total')
    # print(val_path[:4])
    # print(val_path[16:20])
    print('get_batch')
    for i in range (3):
        train_batch_path, train_batch_label, index, epoch_completed = \
            get_next_batch(train_path, train_label, index, epoch_completed)
        # val_batch_path, val_batch_label, index1, epoch_completed1 = \
        #     get_next_batch(val_path, val_label, index1, epoch_completed1)
        print('i ==%d' % i)
        print('train')
        print(train_batch_path[:5])
        # print('val')
        # print(val_batch_path[:5])
        # if i == 0:
        #     print('i ==%d' % i)
        #     print(train_batch_path[:4])
        #     print('path')
        #     # print(path[:2])
        #     print(val_batch_path[:4])
        #     print('path1')
        # if i == 233:
        #     print('val')
        #
        #     print('i ==%d' % i)
        #     # print(epoch_completed1)
        #     # print(path[:2])
        #     print(val_batch_path[14:])
        # if i == 234:
        #     print('i ==%d' % i)
        #     # print(epoch_completed1)
        #     print(val_batch_path[4:6])
        #     print(val_batch_path[6:10])
        #
        # if i == 1092:
        #     print('train')
        #     print('i ==%d' % i)
        #     # print(epoch_completed)
        #     # print(path[:2])
        #     print(train_batch_path[14:])
        # if i == 1093:
        #     print('i ==%d' % i)
        #     # print(epoch_completed)
        #     print(train_batch_path[10:12])
        #     print(train_batch_path[12:])
        # images, labels, index, epoch_completed = get_batch_data(train_path, train_label,index, epoch_completed)
        # print(images.shape,labels.shape)




