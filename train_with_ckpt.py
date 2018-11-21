# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 10:02:04 2018

@author: hzm
"""

from __future__ import division, print_function
import tensorflow as tf
import numpy as np
import os, sys, time, re
from multiprocessing import Pool
import util

os.environ['CUDA_DEVICES_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':
    
    data_path = '/home/hzm/cifar_data'  # data path
    ckpt_meta = 'ckpt/model-final.meta' # meta checkpoint the resotre the graph
    ckpt = 'ckpt/model-final'           # checkpoint to resotre the variables
    classes = 10                        # total classes of the label
    epochs = 200                        # epochs to train
    init_lr = 0.1                       # initial learning rate
    # learning_rate should be callable function with the EPOCH as its input parameter.
    learning_rate = lambda e: init_lr if e < 100 else (init_lr / 10) if e < 150 else (init_lr / 100)
    
    # If you haven't save the optimizer (default) in your checkpoint, you must define a 
    # new optimizer to create the train_op.
    optimizer = tf.train.MomentumOptimizer  # the optimizer to use, as you please
    grad_clip = 5.0                     # gradient clipping to avoid exploration
    
    momentum = 0.9                      # the momentuum Momentum optimizer
    use_nesterov = True                 # whether use the Nesterov Momentum
    save_optim = False                  # whether save the variables of the optimizer while making a checkpoint
    
    val_ratio = 0.1                     # take a part of the traing set to apply validation
    train_batch_size = 128              # batch size of the training set
    val_batch_size = 100                # batch size of the validating set
    
    # data augmentation methods
    mixup_alpha = 1.0
    autoAugment = False
    
    np.random.seed(0)
    xs_train = np.array([data_path + '/train/' + f for f in os.listdir(data_path + '/train/')])
    ys_train = np.array([int(re.split('[_.]', f)[1]) for f in os.listdir(data_path + '/train/')])
    xs_test = np.array([data_path + '/test/' + f for f in os.listdir(data_path + '/test/')])
    ys_test = np.array([int(re.split('[_.]', f)[1]) for f in os.listdir(data_path + '/test/')])
    
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
    
    xs_val, ys_val = [], []
    for result in results:
        a, b = result.get()
        xs_val.extend(a)
        ys_val.extend(b)
    xs_val, ys_val = np.stack(xs_val), np.stack(ys_val)
    del (results, result, a, b)
    ###########################################################################
    
    """
    ---------------------------------------------------------------------------
    Restore the network and start training.
    ---------------------------------------------------------------------------
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:
        
        saver = tf.train.import_meta_graph(ckpt_meta)
        saver.restore(sess, ckpt)
        
        graph = tf.get_default_graph()
        xs = graph.get_operation_by_name('xs').outputs[0]
        ys = graph.get_operation_by_name('ys').outputs[0]
        batch_size = graph.get_operation_by_name('batch_size').outputs[0]
        train_flag = graph.get_operation_by_name('training_flag').outputs[0]
        lr = graph.get_operation_by_name('lr').outputs[0]
        
        label = tf.get_collection('batch_label')[0]
        pred = tf.get_collection('pred')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]
        data_loader_initializer = tf.get_collection('data_loader_initializer')[0]
        
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
        
        writer = tf.summary.FileWriter('graph1', sess.graph)
        writer.close()
        
        #######################################################################
        # Augment the training data with multiprocess.
        print('Data preparing ...')
        procs = 5
        xs_train1 = np.split(xs_train[:(xs_train.shape[0] // procs * procs)], procs)
        ys_train1 = np.split(ys_train[:(ys_train.shape[0] // procs * procs)], procs)
        
        pool = Pool(procs)
        results = []
        for i in range(procs):
            results.append(pool.apply_async(util.batch_parse, (xs_train1[i], ys_train1[i], True, mixup_alpha, autoAugment, )))
        pool.close()
        pool.join()
        
        xs_train1, ys_train1 = [], []
        for result in results:
            a, b = result.get()
            xs_train1.extend(a)
            ys_train1.extend(b)
        xs_train1, ys_train1 = np.stack(xs_train1), np.stack(ys_train1)
        del (pool, results, result, a, b)
        #######################################################################
        
        print('Training ...')
        begin = time.time()
        
        rnd_samples = np.arange(ys_train.shape[0])
        train_losses, train_err = np.zeros([2, epochs])
        val_losses, val_err = np.zeros([2, epochs])
        best_val = 100.0
        for e in range(epochs):
            
            """
            -------------------------------------------------------------------
            Training stage.
            -------------------------------------------------------------------
            """
            sess.run(data_loader_initializer, feed_dict={xs: xs_train1, ys: ys_train1, batch_size: train_batch_size})
            ###################################################################
            # Augment the training data every epoch, 
            # and do it with another process to save time.
            np.random.shuffle(rnd_samples)
            xs_train, ys_train = xs_train[rnd_samples], ys_train[rnd_samples]
            
            pool = Pool(1)
            result = pool.apply_async(util.batch_parse, (xs_train, ys_train, True, mixup_alpha, autoAugment, ))
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
            xs_train1, ys_train1 = np.stack(a), np.stack(b)
            del (pool, result, a, b)
            
            """
            -------------------------------------------------------------------
            Validation stage.
            -------------------------------------------------------------------
            """
            val_time = time.time()
            sess.run(data_loader_initializer, feed_dict={xs: xs_val, ys: ys_val, batch_size: val_batch_size})
            for i in range(val_batches):
                loss_val, label_val, pred_val = sess.run([loss, label, pred], feed_dict={train_flag: False})
                val_err[e] += 100.0 * np.sum(np.argmax(pred_val, axis=1) != np.argmax(label_val, axis=1))
                val_losses[e] += loss_val
            val_losses[e] /= val_batches
            val_err[e] /= (val_batches * val_batch_size)
            
            cur_time = time.time()
            h, m, s = util.parse_time(cur_time - begin)
            print('Epoch {}:   Val_loss = {:.3f},   Val_err = {:.2f}  ({} samples)  {:.2f}s   '.format(
                                        e+1, val_losses[e], val_err[e], val_size, cur_time-val_time))
            print('Global time has passed {:.0f}:{:.0f}:{:.0f}'.format(h, m, s))
            print('')
            
            # Make a checkpoint.
            if val_err[e] < best_val:
                best_val = val_err[e]
                saver.save(sess, 'new_ckpt/model', global_step=e+1, write_meta_graph=True)
        
        # Make a final checkpoint.
        saver.save(sess, 'new_ckpt/model-final', write_meta_graph=True)
        print('Training time: {:.2f}'.format(time.time() - begin))
        util.save_training_result('new_ckpt/training_result', train_losses, train_err, val_losses, val_err)
        
    del(xs_train1, xs_val, xs_train)
    util.test('new_ckpt/model-final.meta', 'new_ckpt/model-final', xs_test, ys_test)
    