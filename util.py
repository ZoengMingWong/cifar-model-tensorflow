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

def parse_img(filename, train=True, autoAug=True):
    img = Image.open(filename)
    if train == True:
        img = ImageOperation.pad_to_bounding_box(img, 4, 4, 40, 40)
        img = ImageOperation.random_crop(img, size=[32, 32])
        img = ImageOperation.random_flip_left_right(img)
        if autoAug == True:
            img = ImagePolicy.policy(img)
            img = ImageOperation.cutout(img, 16)
    img = ImageOperation.per_image_standarization(img)
    return img
    
def batch_parse(xs_batch, ys_batch, train=True, mixup_alpha=0.0, autoAug=True):
    ys_one_hot = np.zeros([len(ys_batch), 10])
    ys_one_hot[range(len(ys_batch)), ys_batch] = 1
    
    xs, ys = [], []
    if train == True and mixup_alpha > 0.0:
        shuff = np.random.permutation(range(len(xs_batch)))
        lams = np.random.beta(mixup_alpha, mixup_alpha, len(xs_batch))
        for i in range(len(xs_batch)):
            x_a, y_a = parse_img(xs_batch[i], train=True, autoAug=autoAug), ys_one_hot[i]
            x_b, y_b = parse_img(xs_batch[shuff[i]], train=True, autoAug=autoAug), ys_one_hot[shuff[i]]
            xs.append(lams[i] * x_a + (1 - lams[i]) * x_b)
            ys.append(lams[i] * y_a + (1 - lams[i]) * y_b)
    else:
        for x, y in zip(xs_batch, ys_one_hot):
            xs.append(parse_img(x, train=train, autoAug=autoAug))
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
        losses_test, err_test = 0., 0.
        test_batches = ys_test.shape[0] // test_batch_size
        for i in range(test_batches):
            loss_test, label_test, pred_test = sess.run([loss, label, pred], feed_dict={train_flag: False})
            err_test += 100.0 * np.sum(np.argmax(pred_test, axis=1) != np.argmax(label_test, axis=1))
            losses_test += loss_test
        losses_test /= test_batches
        err_test /= (test_batches * test_batch_size)
        print('Test: Loss = {:.3f}, Test_err = {:.2f}'.format(losses_test, err_test))

def save_training_result(file_name, train_loss, train_err, val_loss, val_err):
        
    epochs = len(train_loss)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    l1 = ax1.plot(range(epochs), train_loss, 'b--', label='train_loss')
    l2 = ax1.plot(range(epochs), val_loss, 'r-', label='val_loss')
    ax1.set_ylim(0, 5)
    ax1.set_yticks(np.arange(0, 5, 0.5))
    ax1.grid(linestyle='--')
    
    ax2 = ax1.twinx()
    l3 = ax2.plot(range(epochs), train_err, 'k--', label='train_error', lw=1)
    l4 = ax2.plot(range(epochs), val_err, 'm-', label='val_error', lw=1)
    ax2.set_ylim(0, 50)
    ax2.set_yticks(np.arange(0, 50, 5))
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax2.set_ylabel('Error')
    
    ls = l1 + l3 + l2 + l4
    ls_label = [l_i.get_label() for l_i in ls]
    ax1.legend(ls, ls_label, loc='best')
    
    plt.savefig(file_name + '.png', dpi='figure')
    
    model = open(file_name + '.txt', 'a')
    for e in range(epochs):
        print('', file=model)
        print('Epoch {}: Train_loss = {:.3f}, Train_err = {:.2f}'.format(e+1, train_loss[e], train_err[e]), file=model)
        print('    Validation: Loss = {:.3f},   Val_err = {:.2f}'.format(val_loss[e], val_err[e]), file=model)
        print('', file=model)
    model.close()
    
    



