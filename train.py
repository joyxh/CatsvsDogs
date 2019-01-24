#new-train 8.7
import os
import numpy as np
import tensorflow as tf
import data
import models

n_class = 2
batch_size = 16
train_max_epoch = 70000
# learning_rate = 0.0001
# #验证集和测试集所占比例
# tv_proportion = 0.15
# # 存放图片文件夹路径
# image_dir = 'Data/train'


def run_training(image_dir, tv_proportion):
    #分别初始化测试集和验证集在get batch时的index和epoch_completed
    train_index = 0
    train_epoch_completed = 0
    val_index = 0
    val_epoch_completed = 0
    # 保存生成文件的目录
    logs_train_dir = './generated_file/train'
    logs_val_dir = './generated_file/val'

    #从train data中划分出验证集和测试集
    test_label, test_path, val_path, val_label, train_path, train_label = \
        data.get_test_and_val(image_dir, tv_proportion)
    #分别为image和label创建占位符
    X = tf.placeholder(tf.float32, shape=[batch_size, 224, 224, 3], name="X")
    Y = tf.placeholder(tf.int32, shape=[batch_size], name="Y")
    #定义计算图op
    logit = models.VGG16(X)
    loss = losses(logit,Y)
    acc = evaluation(logit,Y)
    #初始化learning_rate，没有则创建变量
    learning_rate = tf.get_variable(name='lr', initializer=0.001)
    train_op = training(loss, learning_rate)

    #开启会话，并初始化全局变量
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    #写入图表本身和即时数据具体值
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    val_writer = tf.summary.FileWriter(logs_val_dir,sess.graph)

    #汇总数据，进行保存
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()

    # 读入预训练数据
    data_dict = np.load('Data/VGG_imagenet.npy', encoding="latin1").item()
    print('start loading the pre-trained model...')
    for key in data_dict:
        with tf.variable_scope(key, reuse=True):
            for subkey in data_dict[key]:
                # print("assign pretrain model " + subkey + " to " + key)
                try:
                    var = tf.get_variable(subkey)
                    # 把相应层数的参数替换为预训练好的参数，从data_dict到var
                    sess.run(var.assign(data_dict[key][subkey]))
                    print("assign pretrain model " + subkey + " to " + key)
                except ValueError:
                    print("ignore " + key)

    print("Start training...")

    accuracy = []
    #进行迭代
    for step in range(train_max_epoch):
        #每批取batch_size个训练样本，放入字典
        train_batch_X, train_batch_Y, train_index, train_epoch_completed = data.get_batch_data(\
            train_path, train_label, train_index, train_epoch_completed)
        feed_dict_train = {X: train_batch_X, Y: train_batch_Y}

        #每10k步，学习率衰减为原来的0.1倍
        if step != 0 and step % 10000 == 0:
            sess.run(learning_rate.assign(sess.run(learning_rate)*0.1))
        # print('lr:', learning_rate.eval(session=sess))

        #feed数据，进行训练，求损失和准确率，保存数据
        _, summary_str, tra_loss, tra_acc = sess.run(fetches=[train_op, summary_op, loss, acc], feed_dict=feed_dict_train)
        train_writer.add_summary(summary_str, step)

        #每10步打印一次训练结果
        if step % 10 == 0 or (step + 1) == train_max_epoch:
            print('step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
            # summary_str = sess.run(summary_op)

        #每40步，进行一次验证，保存并打印结果，保存数据
        if step % 40 == 0 or (step + 1) == train_max_epoch:
            val_batch_X, val_batch_Y, val_index, val_epoch_completed = data.get_batch_data( \
                val_path, val_label, val_index, val_epoch_completed)
            feed_dict_val = {X: val_batch_X, Y: val_batch_Y}
            summary_str, val_loss, val_acc = sess.run(fetches=[summary_op, loss, acc], \
                                         feed_dict=feed_dict_val)
            accuracy.append(val_acc)
            print('step %d, val loss = %.2f, val accuracy = %.2f%%' % (step, val_loss, val_acc * 100.0))
            # summary_str = sess.run(summary_op)
            val_writer.add_summary(summary_str, step)

        #生成chenckpoints文件，保存最后一次迭代的结果
        if (step + 1) == train_max_epoch:
            checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step+1)

    #计算验证集的平均准确率
    val_total_acc = np.array(accuracy)
    val_total_acc = np.mean(val_total_acc) * 100

    sess.close()

    return val_total_acc


def losses(logits, labels):
    '''损失函数'''
    with tf.variable_scope('loss') as scope:
        #计算交叉熵,sparse将label转化为one-hot
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits \
            (logits=logits, labels=labels, name='xentropy_per_example')
        #总损失为batchsize下交叉熵平均值
        loss = tf.reduce_mean(cross_entropy, name='loss')
        #汇总loss，画图用
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss


def training(loss, learning_rate):
    '''优化函数'''
    with tf.name_scope('optimizer'):
        #Adam最速下降法
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # global_step = tf.Variable(0, name='global_step', trainable=False)
        # 求min（loss）的最优参数
        train_op = optimizer.minimize(loss)
    return train_op



def evaluation(logits, labels):
    '''评估函数'''
    with tf.variable_scope('accuracy') as scope:
        #预测为真时，判定正确
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        #准确率为预测为真的数目除以总数
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy



image_dir = 'Data/train'
tv_proportion = 0.15
if __name__ == '__main__':
    val_total_acc = run_training(image_dir, tv_proportion)
    print('val_total_accuracy = %.2f%%' % val_total_acc)

