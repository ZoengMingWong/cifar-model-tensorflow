# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 09:53:59 2018

@author: hzm
"""

from __future__ import print_function
import os, time
from PIL import  Image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from model import ImageOperation, ImagePolicy

np.random.seed(0)

def cifar10_to_png(src_dir, dst_dir=None):
    """
    Encode the orignal cifar10 dataset to PNG images, and we can parse the label
    from the filename, e.g. train0_6.png indicates the first training image's label is 6.
    It is memory efficient if we can dynamically decode the images while needed.
    """
    import cPickle
    xs_train = []
    ys_train = []
    for i in '12345':
        with open(src_dir+'/data_batch_'+i, 'rb') as f:
            train_data = cPickle.load(f)
            xs_train.append(train_data['data'])
            ys_train.extend(train_data['labels'])
    with open(src_dir+'/test_batch', 'rb') as f:
        test_data = cPickle.load(f)
        xs_test = np.hstack(tuple([test_data['data'][:, i::1024] for i in range(1024)]))
        ys_test = np.array(test_data['labels'])
    
    xs_train = np.vstack(tuple(xs_train))
    xs_train = np.hstack(tuple([xs_train[:, i::1024] for i in range(1024)]))
    ys_train = np.array(ys_train)
    
    if dst_dir is None:
        dst_dir = src_dir
    os.mkdir(dst_dir)
    os.mkdir(dst_dir+'/test')
    for i in range(ys_test.shape[0]):
        plt.imsave(dst_dir+'/test/'+str(i)+'_'+str(ys_test[i])+'.png', xs_test[i].reshape(32, 32, 3))
    
    os.mkdir(dst_dir+'/train')
    for i in range(ys_train.shape[0]):
        plt.imsave(dst_dir+'/train/'+str(i)+'_'+str(ys_train[i])+'.png', xs_train[i].reshape(32, 32, 3))
    return None
    
def parse_time(sec):
    h = int(sec // 3600)
    m = int(sec % 3600 // 60)
    s = sec % 60
    return h, m, s

def parse_img(filename, train=True, autoAug=True):
    """
    Parse a PNG image from its filename, randomly disturb and standarize it.
    """
    img = Image.open(filename)
    if train == True:
        if autoAug == True:
            img = ImagePolicy.policy(img)
            img = ImageOperation.cutout(img, 16)
        img = ImageOperation.pad_to_bounding_box(img, 4, 4, 40, 40)
        img = ImageOperation.random_crop(img, size=[32, 32])
        img = ImageOperation.random_flip_left_right(img)
        
    img = ImageOperation.image_standarization(img)
    return img
    
def batch_parse(xs_batch, ys_batch, train=True, mixup_alpha=0.0, autoAug=True):
    """
    Parse a batch of images.
    """
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
        xs = graph.get_operation_by_name('Data_loader/xs').outputs[0]
        ys = graph.get_operation_by_name('Data_loader/ys').outputs[0]
        batch_size = graph.get_operation_by_name('Data_loader/batch_size').outputs[0]
        train_flag = graph.get_operation_by_name('training_flag').outputs[0]
        
        error = tf.get_collection('error')[0]
        loss = tf.get_collection('loss')[0]
        data_loader_initializer = tf.get_collection('data_loader_initializer')[0]
        
        print('Testing ......')
        begin = time.time()
        sess.run(data_loader_initializer, feed_dict={xs: xs_test, ys: ys_test, batch_size: test_batch_size})
        losses_test, err_test = 0., 0.
        test_batches = ys_test.shape[0] // test_batch_size
        for i in range(test_batches):
            loss_i, err_i = sess.run([loss, error], feed_dict={train_flag: False, batch_size: test_batch_size})
            err_test += err_i
            losses_test += loss_i
        losses_test /= test_batches
        err_test /= test_batches
        print('Test: Loss = {:.3f}, Test_err = {:.2f}'.format(losses_test, err_test))
        print('Time has passed {:.2f}s.'.format(time.time()-begin))
    return None
    
def average_gradients(tower_grads):
    """
    For multiGPUs implementation, average the gradients gathered from different GPUs.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expand_g = tf.expand_dims(g, 0)
            grads.append(expand_g)
        grad = tf.concat(grads, axis=0)
        grad = tf.reduce_mean(grad, axis=0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def save_training_result(file_name, train_loss, train_err, val_loss, val_err):
        
    epochs = len(train_loss)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    l1 = ax1.plot(range(epochs), train_loss, 'b--', label='train_loss', lw=1, alpha=0.8)
    l2 = ax1.plot(range(epochs), val_loss, 'r-', label='val_loss', lw=1, alpha=0.8)
    ax1.set_ylim(0, 5)
    ax1.set_yticks(np.arange(0, 5, 0.5))
    ax1.grid(linestyle='--')
    
    ax2 = ax1.twinx()
    l3 = ax2.plot(range(epochs), train_err, 'k--', label='train_error')
    l4 = ax2.plot(range(epochs), val_err, 'm-', label='val_error')
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
        print('Epoch {}:   Val_loss = {:.3f},   Val_err = {:.2f}'.format(e+1, val_loss[e], val_err[e]), file=model)
        print('', file=model)
    model.close()
    return None
    
    



