#new-models2  8.3

import tensorflow as tf

n_class = 2
batch_size = 16


def VGG16(X):
    #conv1
    with tf.variable_scope("conv1_1"):
        weights = tf.get_variable("weights",
                                  shape=[3, 3, 3, 64],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005),
                                  trainable = False)
        biases = tf.get_variable("biases",
                                 shape=[64],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0),
                                 trainable=False)
        conv = tf.nn.conv2d(X, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1_1 = tf.nn.relu(pre_activation, name="conv1_1")# size=[num,224,224,64]

    with tf.variable_scope("conv1_2"):
        weights= tf.get_variable("weights",
                                  shape=[3, 3, 64, 64],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005),
                                  trainable = False)
        biases = tf.get_variable("biases",
                                 shape=[64],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0),
                                 trainable=False)
        conv = tf.nn.conv2d(conv1_1, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1_2 = tf.nn.relu(pre_activation, name="conv1_2")# size=[num,224,224,64]


    # pool1
    with tf.variable_scope("pooling1"):
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding="SAME", name="pooling1")# size=[num,112,112,64]


    # conv2
    with tf.variable_scope("conv2_1"):
        weights = tf.get_variable("weights",
                                  shape=[3, 3, 64, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005),
                                  trainable=False)
        biases = tf.get_variable("biases",
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0),
                                 trainable=False)
        conv = tf.nn.conv2d(pool1, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2_1 = tf.nn.relu(pre_activation, name="conv2_1")# size=[num,112,112,128]

    with tf.variable_scope("conv2_2"):
        weights = tf.get_variable("weights",
                                  shape=[3, 3, 128, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005),
                                  trainable=False)
        biases = tf.get_variable("biases",
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0),
                                 trainable=False)
        conv = tf.nn.conv2d(conv2_1, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2_2 = tf.nn.relu(pre_activation, name="conv2_2")# size=[num,112,112,128]


    # pool2
    with tf.variable_scope("pooling2"):
        pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding="SAME", name="pooling2")# size=[num,56,56,128]


    # conv3
    with tf.variable_scope("conv3_1"):
        weights = tf.get_variable("weights",
                                  shape=[3, 3, 128, 256],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005),
                                  trainable=False)
        biases = tf.get_variable("biases",
                                 shape=[256],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0),
                                 trainable=False)
        conv = tf.nn.conv2d(pool2, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3_1 = tf.nn.relu(pre_activation, name="conv3_1")  # size=[num,56,56,256]

    with tf.variable_scope("conv3_2"):
        weights = tf.get_variable("weights",
                                  shape=[3, 3, 256, 256],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005),
                                  trainable=False)
        biases = tf.get_variable("biases",
                                 shape=[256],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0),
                                 trainable=False)
        conv = tf.nn.conv2d(conv3_1, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3_2 = tf.nn.relu(pre_activation, name="conv3_2")  # size=[num,56,56,256]

    with tf.variable_scope("conv3_3"):
        weights = tf.get_variable("weights",
                                  shape=[3, 3, 256, 256],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005),
                                  trainable=False)
        biases = tf.get_variable("biases",
                                 shape=[256],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0),
                                 trainable=False)
        conv = tf.nn.conv2d(conv3_2, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3_3 = tf.nn.relu(pre_activation, name="conv3_3")  # size=[num,56,56,256]


    # pool3
    with tf.variable_scope("pooling3"):
        pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding="SAME", name="pooling3")  # size=[num,28,28,256]


    # conv4
    with tf.variable_scope("conv4_1"):
        weights = tf.get_variable("weights",
                                  shape=[3, 3, 256, 512],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005),
                                  trainable=False)

        biases = tf.get_variable("biases",
                                 shape=[512],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0),
                                 trainable=False)
        conv = tf.nn.conv2d(pool3, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4_1 = tf.nn.relu(pre_activation, name="conv4_1")  # size=[num,28,28,512]

    with tf.variable_scope("conv4_2"):
        weights = tf.get_variable("weights",
                                  shape=[3, 3, 512, 512],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005),
                                  trainable=False)
        biases = tf.get_variable("biases",
                                 shape=[512],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0),
                                 trainable=False)
        conv = tf.nn.conv2d(conv4_1, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4_2 = tf.nn.relu(pre_activation, name="conv4_2")  # size=[num,28,28,512]

    with tf.variable_scope("conv4_3"):
        weights = tf.get_variable("weights",
                                  shape=[3, 3, 512, 512],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005),
                                  trainable=False)
        biases = tf.get_variable("biases",
                                 shape=[512],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0),
                                 trainable=False)
        conv = tf.nn.conv2d(conv4_2, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4_3 = tf.nn.relu(pre_activation, name="conv4_3")  # size=[num,28,28,512]


    # pool4
    with tf.variable_scope("pooling4"):
        pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding="SAME", name="pooling4")  # size=[num,14,14,512]


    # conv5
    with tf.variable_scope("conv5_1"):
        weights = tf.get_variable("weights",
                                  shape=[3, 3, 512, 512],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005),
                                  trainable=False)
        biases = tf.get_variable("biases",
                                 shape=[512],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0),
                                 trainable=False)
        conv = tf.nn.conv2d(pool4, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv5_1 = tf.nn.relu(pre_activation, name="conv5_1")  # size=[num,14,14,512]

    with tf.variable_scope("conv5_2"):
        weights = tf.get_variable("weights",
                                  shape=[3, 3, 512, 512],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005),
                                  trainable=False)
        biases = tf.get_variable("biases",
                                 shape=[512],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0),
                                 trainable=False)
        conv = tf.nn.conv2d(conv5_1, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv5_2 = tf.nn.relu(pre_activation, name="conv5_2")  # size=[num,14,14,512]

    with tf.variable_scope("conv5_3"):
        weights = tf.get_variable("weights",
                                  shape=[3, 3, 512, 512],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005),
                                  trainable = False)
        biases = tf.get_variable("biases",
                                 shape=[512],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0),
                                 trainable=False)
        conv = tf.nn.conv2d(conv5_2, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv5_3 = tf.nn.relu(pre_activation, name="conv5_3")  # size=[num,14,14,512]


    # pool5
    with tf.variable_scope("pooling4"):
        pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding="SAME", name="pooling5")  # size=[num,7,7,512]


    # full-connect1
    with tf.variable_scope("fc1"):
        reshape = tf.reshape(pool5, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable("weights",
                                  shape=[dim, 256],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005))
        biases = tf.get_variable("biases",
                                 shape=[256],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0))
        fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name="fc1")
        fc1_drop = tf.nn.dropout(fc1, 0.5, name="fc1_drop")#[4096]


    # full-connect2
    with tf.variable_scope("fc2"):
        weights = tf.get_variable("weights",
                                  shape=[256, 256],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005))
        biases = tf.get_variable("biases",
                                 shape=[256],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0))
        fc2 = tf.nn.relu(tf.matmul(fc1_drop, weights) + biases, name="fc2")#[4096]


    # full-connect3
    with tf.variable_scope("fc3"):
        weights = tf.get_variable("weights",
                                  shape=[256, n_class],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005))
        biases = tf.get_variable("biases",
                                 shape=[n_class],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0))
        fc3 = tf.matmul(fc2, weights) + biases#[2]

    return fc3
