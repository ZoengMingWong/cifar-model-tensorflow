# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 10:09:08 2018

@author: hzm
"""
from __future__ import division, print_function
import tensorflow as tf
from tensorflow import data
import numpy as np
import os, sys, time
from multiprocessing import Pool
from model import layers, resnet, resnext, wideResnet
import util

os.environ['CUDA_DEVICES_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


if __name__ == '__main__':
    
    # Loading the dataset. Be careful that here I have saved the original dataset to .png pitures, 
    # and the program decodes the pictures while needed.
    data_path = '/home/hzm/cifar_data'
    xs_train = np.array([data_path + '/train/' + f for f in os.listdir(data_path + '/train/')])
    ys_train = np.array([int(f[-5]) for f in os.listdir(data_path + '/train/')])
    xs_test = np.array([data_path + '/test/' + f for f in os.listdir(data_path + '/test/')])
    ys_test = np.array([int(f[-5]) for f in os.listdir(data_path + '/test/')])
    
    # Below are some typical models in papers, you can define some other models yourself.
    model_dict = {'PreResNet18':resnet.ResNet18, 'PreResNet50':resnet.PreResNet50, 
                  'PreResNeXt29_32x4d':resnext.ResNeXt29_32x4d, 
                  'WRN_28_10':wideResnet.WRN_28_10, 'WRN_16_4':wideResnet.WRN_16_4}
    
    # Set the parameter.
    net = model_dict['PreResNet18']
    classes = 10
    epochs = 200
    init_lr = 0.1
    # learning_rate should be callable function with the EPOCH as its input parameter.
    learning_rate = lambda e: init_lr if e < 100 else (init_lr / 10) if e < 150 else (init_lr / 100)
    weight_decay = 1e-4
    grad_clip = 5.0
    
    # Below are some augment methods. Set MIXUP_ALPHA to zero to disable the mixup augmentation, 
    # and a non-zero float number represents the alpha (here equal to beta) of the BETA distribution.
    # Set AUTOAUGMENT to TRUE to enable the auto-Augmentation introduced by Google.
    # Set mixup_alpha = 0 and autoAugment = False to use the baseline augment methods.
    mixup_alpha = 1.0
    autoAugment = False
    
    val_ratio = 0.1
    train_batch_size = 128
    val_batch_size = 100
    optimizer = tf.train.MomentumOptimizer
    # Set the parameters of the MomentumOptimizer
    momentum = 0.9
    use_nesterov = True
    
    save_optim = False
    np.random.seed(0)
    
    val_size = int(ys_train.shape[0] * val_ratio)
    val_batches = val_size // val_batch_size
    val_size = val_batches * val_batch_size
    
    xs_train, xs_val = xs_train[:-val_size], xs_train[-val_size:]
    ys_train, ys_val = ys_train[:-val_size], ys_train[-val_size:]
    train_batches = ys_train.shape[0] // train_batch_size
    
    ###########################################################################
    # Preprocess the validating images with multiprocessing.
    procs = 5
    xs_val = np.split(xs_val[:(xs_val.shape[0] // procs * procs)], procs)
    ys_val = np.split(ys_val[:(ys_val.shape[0] // procs * procs)], procs)
    
    pool = Pool(procs)
    results = []
    for i in range(procs):
        results.append(pool.apply_async(util.batch_parse, (xs_val[i], ys_val[i], False, mixup_alpha, autoAugment, )))
        
    pool.close()
    pool.join()
    
    xs_batch, ys_batch = [], []
    for result in results:
        a, b = result.get()
        xs_batch.extend(a)
        ys_batch.extend(b)
    xs_val, ys_val = np.stack(xs_batch), np.stack(ys_batch)
    ###########################################################################
    
    # Setup the datasets.
    tf.reset_default_graph()
    
    train_x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='train_x')
    train_y = tf.placeholder(tf.float32, shape=[None, classes], name='train_y')
    train_set = data.Dataset.from_tensor_slices((train_x, train_y)).batch(train_batch_size)
    train_iter = train_set.make_initializable_iterator()
    
    val_x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='val_x')
    val_y = tf.placeholder(tf.float32, shape=[None, classes], name='val_y')
    val_set = data.Dataset.from_tensor_slices((val_x, val_y)).batch(val_batch_size).repeat()
    val_iter = val_set.make_initializable_iterator()
    
    ###########################################################################
    # Setup the network
    train_flag = tf.placeholder(tf.bool, shape=[], name='training_flag')
    img, label = tf.cond(train_flag, train_iter.get_next, val_iter.get_next, name='dataset_selector')
    
    global_avg = net(img, train_flag)
    pred = layers.linear(global_avg, classes, activation=None, use_bias=True, name='prediction')
    
    loss_no_reg = tf.losses.softmax_cross_entropy(label, pred)
    # Apply the weight decay.
    reg_loss = []
    for var in tf.trainable_variables():
        reg_loss.append(tf.nn.l2_loss(var))
    loss = tf.add(loss_no_reg, tf.multiply(weight_decay, tf.add_n(reg_loss)), name='loss_with_reg')
    
    lr = tf.placeholder(tf.float32, shape=[], name='lr')
    if optimizer is tf.train.MomentumOptimizer:
        optim = optimizer(lr, momentum=momentum, use_nesterov=use_nesterov)
    else:
        optim = optimizer(lr)
    
    # Apply the backpropagation with gradient clipping.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        grads, variables = zip(*optim.compute_gradients(loss))
        grads, global_norm = tf.clip_by_global_norm(grads, grad_clip)
        train_op = optim.apply_gradients(zip(grads, variables))
    ###########################################################################
    
    ###########################################################################
    # Config the saver to save the necessary variables
    tf.add_to_collection('batch_label', label)
    tf.add_to_collection('pred', pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('train_op', train_op)
    tf.add_to_collection('train_iter_initializer', train_iter.initializer)
    tf.add_to_collection('val_iter_initializer', val_iter.initializer)
    
    var_list = tf.trainable_variables()
    '''
        To continue traing with a checkpoint in the future, we must save the state of the optimizer, 
        or it would occur some errors as 'tf.Saver()' wouldn't save these variables by defalut.
        
        However, the size of these variables would be somewhat large, if you have no need to 
        retrain the model with checkpoints, or you want to train the model with a new optimizer,
        just comment the statement below.
    '''
    if save_optim == True:
        var_list += optim.variables()
    
    '''
        As the 'moving_mean' and 'moving_variance' (necessary for the inference in batch normalization) 
        are not the trainable variables, we should fetch them from the global variables.
    '''
    global_var = tf.global_variables()
    moving_vars = [g for g in global_var if 'moving_mean' in g.name]
    moving_vars += [g for g in global_var if 'moving_variance' in g.name]
    var_list += moving_vars
    
    saver = tf.train.Saver(var_list=var_list, max_to_keep=5)
    ###########################################################################
        
    """
        Config the devices and start training.
    """
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(val_iter.initializer, feed_dict={val_x: xs_val, val_y: ys_val})
        
        #######################################################################
        # Augment the training data with multiprocess.
        print('Data preparing ...')
        procs = 5
        xs = np.split(xs_train[:(xs_train.shape[0] // procs * procs)], procs)
        ys = np.split(ys_train[:(ys_train.shape[0] // procs * procs)], procs)
        
        pool = Pool(procs)
        results = []
        for i in range(procs):
            results.append(pool.apply_async(util.batch_parse, (xs[i], ys[i], True, mixup_alpha, autoAugment, )))
        pool.close()
        pool.join()
        
        xs_batch, ys_batch = [], []
        for result in results:
            a, b = result.get()
            xs_batch.extend(a)
            ys_batch.extend(b)
        xs, ys = np.stack(xs_batch), np.stack(ys_batch)
        #######################################################################
        
        print('Training ...')
        begin = time.time()
        
        rnd_samples = np.arange(ys_train.shape[0])
        train_losses, train_err = np.zeros([2, epochs])
        val_losses, val_err = np.zeros([2, epochs])
        best_val = 100.0
        for e in range(epochs):
            
            sess.run(train_iter.initializer, feed_dict={train_x: xs, train_y: ys})
            
            ###################################################################
            # Augment the training data every epoch, 
            # and do it with another process to save time.
            
            np.random.shuffle(rnd_samples)
            xs_train, ys_train = xs_train[rnd_samples], ys_train[rnd_samples]
            
            pool = Pool(1)
            result = pool.apply_async(util.batch_parse, (xs_train, ys_train, True, mixup_alpha, autoAugment))
            pool.close()
            ###################################################################
            
            for i in range(train_batches):
                batch_time = time.time()
                _, loss_i, label_i, pred_i = sess.run([train_op, loss, label, pred], feed_dict={train_flag: True, lr: learning_rate(e)})
                err_batch = 100.0 * np.sum(np.argmax(pred_i, axis=1) != np.argmax(label_i, axis=1)) / train_batch_size
                
                train_losses[e] += loss_i
                train_err[e] += err_batch
                
                sys.stdout.write('Epoch {}: {} / {} batches.  Error: {:.2f}  Loss: {:.3f}  {:.2f}s   '.format(
                                    e+1, i+1, train_batches, err_batch, loss_i, time.time()-batch_time) + '\r')
                sys.stdout.flush()
                
            train_losses[e] /= train_batches
            train_err[e] /= train_batches
            print('')
            print('Epoch {}: Train_loss = {:.3f}, Train_err = {:.2f}'.format(e+1, train_losses[e], train_err[e]))
            
            pool.join()
            a, b = result.get()
            xs, ys = np.stack(a), np.stack(b)
            
            ###################################################################
            # Make the validation per epoch to trace the model performance.
            # You'd better to set the val_bacth_size to a proper number so that all the samples could be tested.
            for i in range(val_batches):
                loss_val, label_val, pred_val = sess.run([loss, label, pred], feed_dict={train_flag: False})
                val_err[e] += 100.0 * np.sum(np.argmax(pred_val, axis=1) != np.argmax(label_val, axis=1))
                val_losses[e] += loss_val
            val_losses[e] /= val_batches
            val_err[e] /= (val_batches * val_batch_size)
            print('    Validation: Loss = {:.3f},   Val_err = {:.2f}  ({} samples)'.format(val_losses[e], val_err[e], val_size))
            print('')
            
            if val_err[e] < best_val and e > 50:
                best_val = val_err[e]
                saver.save(sess, 'ckpt/model', global_step=e+1, write_meta_graph=True)
        
        # Make a final checkpoint.
        saver.save(sess, 'ckpt/model-final', write_meta_graph=True)
        print('Training time: {:.2f}'.format(time.time() - begin))
        util.save_training_result('ckpt/training_result', train_losses, train_err, val_losses, val_err)
        
    del(xs, xs_val, xs_train)
    util.test('ckpt/model-final.meta', 'ckpt/model-final', xs_test, ys_test)





