# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 09:53:59 2018

@author: hzm
"""

from __future__ import print_function
from PIL import  Image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from model import ImageOperation, ImagePolicy

def parse_img(filename, train=True, aug=True):
    img = Image.open(filename)
    if train == True:
        img = ImageOperation.pad_to_bounding_box(img, 4, 4, 40, 40)
        img = ImageOperation.random_crop(img, size=[32, 32])
        img = ImageOperation.random_flip_left_right(img)
        if aug == True:
            img = ImagePolicy.policy(img)
            img = ImageOperation.cutout(img, 16)
    img = ImageOperation.per_image_standarization(img)
    return img
    
def batch_parse(xs_batch, ys_batch, train=True, mixup_alpha=0.0):
    ys_one_hot = np.zeros([len(ys_batch), 10])
    ys_one_hot[range(len(ys_batch)), ys_batch] = 1
    
    xs, ys = [], []
    if train == True and mixup_alpha > 0.0:
        shuff = np.random.permutation(range(len(xs_batch)))
        lams = np.random.beta(mixup_alpha, mixup_alpha, len(xs_batch))
        for i in range(len(xs_batch)):
            x_a, y_a = parse_img(xs_batch[i], train=True), ys_one_hot[i]
            x_b, y_b = parse_img(xs_batch[shuff[i]], train=True), ys_one_hot[shuff[i]]
            xs.append(lams[i] * x_a + (1 - lams[i]) * x_b)
            ys.append(lams[i] * y_a + (1 - lams[i]) * y_b)
    else:
        for x, y in zip(xs_batch, ys_one_hot):
            xs.append(parse_img(x, train=train))
            ys.append(y)
    return xs, ys

def test(ckpt_meta, ckpt, xs_test, ys_test, test_batch_size=100):
    
    xs_test, ys_test = batch_parse(xs_test, ys_test, train=False)
    xs_test, ys_test = np.stack(xs_test), np.stack(ys_test)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    tf.reset_default_graph()
    
    with tf.Session(config=config) as sess:
        saver = tf.train.import_meta_graph(ckpt_meta)
        saver.restore(sess, ckpt)
        
        graph = tf.get_default_graph()
        val_x = graph.get_operation_by_name('val_x').outputs[0]
        val_y = graph.get_operation_by_name('val_y').outputs[0]
        train_flag = graph.get_operation_by_name('training_flag').outputs[0]
        
        label = tf.get_collection('batch_label')[0]
        pred = tf.get_collection('pred')[0]
        loss = tf.get_collection('loss')[0]
        val_iter_initializer = tf.get_collection('val_iter_initializer')[0]
        
        print('Testing ......')
        sess.run(val_iter_initializer, feed_dict={val_x: xs_test, val_y: ys_test})
        losses_test, acc_test = 0., 0.
        test_batches = ys_test.shape[0] // test_batch_size
        for i in range(test_batches):
            loss_test, label_test, pred_test = sess.run([loss, label, pred], feed_dict={train_flag: False})
            acc_test += 100.0 * np.sum(np.argmax(pred_test, axis=1) == np.argmax(label_test, axis=1))
            losses_test += loss_test
        losses_test /= test_batches
        acc_test /= (test_batches * test_batch_size)
        print('Test: Loss = {:.3f}, Test_acc = {:.2f}'.format(losses_test, acc_test))

def save_training_result(file_name, train_loss, train_acc, val_loss, val_acc):
        
    epochs = len(train_loss)
    
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].plot(range(epochs), train_loss, 'b--', label='training')
    ax[0].plot(range(epochs), val_loss, 'r-', label='validation')
    
    ax[1].plot(range(epochs), train_acc, 'b--', label='training')
    ax[1].plot(range(epochs), val_acc, 'r-', label='validation')
    
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend(loc='best')
    
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend(loc='best')
    plt.savefig(file_name + '.png')
    
    model = open(file_name + '_result.txt', 'a')
    for e in range(epochs):
        print('', file=model)
        print('Epoch {}: Train_loss = {:.3f}, Train_acc = {:.2f}'.format(e+1, train_loss[e], train_acc[e]), file=model)
        print('    Validation: Loss = {:.3f},   Val_acc = {:.2f}'.format(val_loss[e], val_acc[e]), file=model)
        print('', file=model)
    model.close()
    
    



