# new-test 8.7
import os
import numpy as np
import tensorflow as tf
import models
import data
import train


# n_classes = 2
batch_size = 16
test_max_epoch = 1875
# 存放图片文件夹路径
image_dir = 'Data/train'
tv_proportion = 0.15

def testing(test_path,test_label):
    #保存文件地址
    logs_test_dir = './generated_file/test'
    logs_train_dir = './generated_file/train'
    # 初始化测试集在get batch时的index和epoch_completed
    test_index = 0
    test_epoch_completed = 0
    print('start testing...')
    # 每批取batch_size个测试样本
    X = tf.placeholder(tf.float32, shape=[batch_size, 224, 224, 3], name="X")
    Y = tf.placeholder(tf.int32, shape=[batch_size], name="Y")

    test_logits = models.VGG16(X)
    test_loss = train.losses(test_logits,Y)
    test_acc = train.evaluation(test_logits,Y)

    # 释放事件文件，合并即时数据
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        accuracy = []
        ckpt = tf.train.get_checkpoint_state(logs_train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            #加载模型， 对最新的模型进行测试验证
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success, global_step is %s' % global_step)

        else:
            print('No checkpoint file found')

        # 写入图表本身和即时数据具体值
        test_writer = tf.summary.FileWriter(logs_test_dir, sess.graph)

        for step in range(test_max_epoch):
            test_batch_X, test_batch_Y, test_index, test_epoch_completed = data.get_batch_data( \
                test_path, test_label, test_index, test_epoch_completed)
            # test_batch_X = np.array(test_batch_X, dtype=np.float32)
            feed_dict_test = {X: test_batch_X, Y: test_batch_Y}
            summary_str, te_loss, te_acc = sess.run(fetches=[summary_op, test_loss, test_acc], \
                                                      feed_dict=feed_dict_test)
            test_writer.add_summary(summary_str, step)
            accuracy.append(te_acc)

            if step % 50 == 0 or (step + 1) == test_max_epoch :
                print('step %d, test loss = %.2f, test accuracy = %.2f%%' % (step, te_loss, te_acc * 100.0))
                # summary_str = sess.run(summary_op)

            # if (step + 1) == test_max_epoch:
            #     # 保存chenckpoints文件（保存测试结果/模型参数的文件）
            #     checkpoint_path = os.path.join(logs_test_dir, 'test.ckpt')
            #     saver.save(sess, checkpoint_path, global_step=step+1)


        test_total_acc = np.array(accuracy)
        test_total_acc = np.mean(test_total_acc) * 100

    return test_total_acc


if __name__ == '__main__':
    test_label, test_path, val_path, val_label, train_path, train_label = \
        data.get_test_and_val(image_dir, tv_proportion)
    test_total_acc = testing(test_path, test_label)
    print('test_total_accuracy = %.2f%%' % test_total_acc)
