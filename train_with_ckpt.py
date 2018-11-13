# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 10:02:04 2018

@author: hzm
"""

from __future__ import division, print_function
import tensorflow as tf
import numpy as np
import os, sys, time
from multiprocessing import Pool
import util

os.environ['CUDA_DEVICES_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':
    
    data_path = '/home/hzm/cifar_data'
    xs_train = np.array([data_path + '/train/' + f for f in os.listdir(data_path + '/train/')])
    ys_train = np.array([int(f[-5]) for f in os.listdir(data_path + '/train/')])
    
    ckpt_meta = 'ckpt/model-final.meta'
    ckpt = 'ckpt/model-final'
    
    epochs = 200
    init_lr = 0.1
    learning_rate = lambda e: init_lr if e < 100 else (init_lr / 10) if e < 150 else (init_lr / 100)
    grad_clip = 5.0
    
    mixup_alpha = 1.0
    autoAugment = False
    
    val_ratio = 0.1
    train_batch_size = 128
    val_batch_size = 100
    
    # You can set the optimizer to None to use the original ones to train the model, 
    # however, the checkpoint must contain the variables of the optimizer, 
    # which may result in a checkpoint with somewhat large size.
    optimizer = tf.train.MomentumOptimizer
    momentum = 0.9
    use_nesterov = True
    np.random.seed(0)
    
    val_size = int(ys_train.shape[0] * val_ratio)
    val_batches = val_size // val_batch_size
    val_size = val_batches * val_batch_size
    
    xs_train, xs_val = xs_train[:-val_size], xs_train[-val_size:]
    ys_train, ys_val = ys_train[:-val_size], ys_train[-val_size:]
    train_batches = ys_train.shape[0] // train_batch_size
    
    ###########################################################################
    # preprocess the validating images with multiprocess
    procs = 5
    xs_val = np.split(xs_val[:(xs_val.shape[0] // procs * procs)], procs)
    ys_val = np.split(ys_val[:(ys_val.shape[0] // procs * procs)], procs)
    
    pool = Pool(procs)
    results = []
    for i in range(procs):
        results.append(pool.apply_async(util.batch_parse, (xs_val[i], ys_val[i], False, )))
        
    pool.close()
    pool.join()
    
    xs_batch, ys_batch = [], []
    for result in results:
        a, b = result.get()
        xs_batch.extend(a)
        ys_batch.extend(b)
    xs_val, ys_val = np.stack(xs_batch), np.stack(ys_batch)
    ###########################################################################
    
    """
        Setup the datasets of training and validating (testing) set.
    """
        
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:
        saver = tf.train.import_meta_graph(ckpt_meta)
        saver.restore(sess, ckpt)
        
        #######################################################################
        # restore the network
        graph = tf.get_default_graph()
        train_x = graph.get_operation_by_name('train_x').outputs[0]
        train_y = graph.get_operation_by_name('train_y').outputs[0]
        
        val_x = graph.get_operation_by_name('val_x').outputs[0]
        val_y = graph.get_operation_by_name('val_y').outputs[0]
        train_flag = graph.get_operation_by_name('training_flag').outputs[0]
        lr = graph.get_operation_by_name('lr').outputs[0]
        
        label = tf.get_collection('batch_label')[0]
        pred = tf.get_collection('pred')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]
        
        train_iter_initializer = tf.get_collection('train_iter_initializer')[0]
        val_iter_initializer = tf.get_collection('val_iter_initializer')[0]
        
        # define a new optimizer
        if optimizer is not None:
            if optimizer is tf.train.MomentumOptimizer:
                optim = optimizer(lr, momentum=0.9, name='new_optimizer')
            else:
                optim = optimizer(lr, name='new_optimizer')
            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                grads, variables = zip(*optim.compute_gradients(loss))
                grads, global_norm = tf.clip_by_global_norm(grads, grad_clip)
                train_op = optim.apply_gradients(zip(grads, variables))
            sess.run(tf.variables_initializer(optim.variables()))
        
        # gather the variables to save
        var_list = tf.trainable_variables()
        
        global_var = tf.global_variables()
        moving_vars = [g for g in global_var if 'moving_mean' in g.name]
        moving_vars += [g for g in global_var if 'moving_variance' in g.name]
        var_list += moving_vars
        
        saver = tf.train.Saver(var_list=var_list, max_to_keep=5)
        #######################################################################
        
        print('Training ......')
        begin = time.time()
        rnd_samples = np.arange(ys_train.shape[0])
        
        #######################################################################
        # Augment the training data for the first epoch
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
        
        sess.run(val_iter_initializer, feed_dict={val_x: xs_val, val_y: ys_val})
        train_losses, train_err = np.zeros([2, epochs])
        val_losses, val_err = np.zeros([2, epochs])
        rnd_samples = np.arange(ys_train.shape[0])
        best_val = 100.0
        for e in range(epochs):
            
            ###################################################################
            # Augment the training data every epoch by another process to save time.
            sess.run(train_iter_initializer, feed_dict={train_x: xs, train_y: ys})
            
            np.random.shuffle(rnd_samples)
            xs_train, ys_train = xs_train[rnd_samples], ys_train[rnd_samples]
            
            pool = Pool(1)
            result = pool.apply_async(util.batch_parse, (xs_train, ys_train, True, mixup_alpha, autoAugment, ))
            pool.close()
            
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
            if val_err[e] < best_val:
                best_val = val_err[e]
                saver.save(sess, 'new_ckpt/model', global_step=(e+1), write_meta_graph=True)
        
        #Make a final checkpoint.
        saver.save(sess, 'new_ckpt/model-final', write_meta_graph=True)
        print('Training time: {:.2f}'.format(time.time() - begin))
        util.save_training_result('new_ckpt/training_result', train_losses, train_err, val_losses, val_err)
    
    
    