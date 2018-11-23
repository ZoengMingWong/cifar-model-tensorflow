# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 10:09:08 2018

@author: hzm
"""
from __future__ import division, print_function
import tensorflow as tf
from tensorflow import data
import numpy as np
import os, sys, time, re
from multiprocessing import Pool
from model import layers, resnet, resnext, wideResnet, pyramidNet, shakeDrop
import util

os.environ['CUDA_DEVICES_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Below are some typical models in papers, you can define some other models yourself.
model_dict = {'ResNet18': resnet.ResNet18, 'PreResNet18': resnet.PreResNet18, 
              'ResNet50': resnet.ResNet50, 'PreResNet50': resnet.PreResNet50, 
              'PreResNeXt29_32x4d': resnext.ResNeXt29_32x4d, 
              'WRN_28_10': wideResnet.WRN_28_10, 'WRN_16_4': wideResnet.WRN_16_4,
              'PyramidNet_a48_d110': pyramidNet.PyramidNet_a48_d110, 
              'BotPyramidNet_a270_d164': pyramidNet.BotPyramidNet_a270_d164,
              'BotPyramidNet_a200_d272': pyramidNet.BotPyramidNet_a200_d272,
              'ShakeDrop_272': shakeDrop.ShakeDrop_a200_d272}


if __name__ == '__main__':
    
    data_path = '/home/hzm/cifar_data'  # data path
    net = model_dict['PreResNet18']     # model to use
    classes = 10                        # total classes of the label
    epochs = 200                        # epochs to train
    init_lr = 0.1                       # initial learning rate
    # learning_rate should be a callable function with the EPOCH as its input parameter.
    learning_rate = lambda e: init_lr if e < 100 else (init_lr / 10) if e < 150 else (init_lr / 100)
    
    optimizer = tf.train.MomentumOptimizer  # the optimizer to use, as you please
    weight_decay = 1e-4                 # weight decay with L2 regularization
    grad_clip = 5.0                     # gradient clipping to avoid exploration
    
    zero_pad = True                     # whether zero-padding or 1x1 conv for the shortcut mapping
    momentum = 0.9                      # the momentuum Momentum optimizer
    use_nesterov = True                 # whether use the Nesterov Momentum
    save_optim = False                  # whether save the variables of the optimizer while making a checkpoint
    
    val_ratio = 0.1                     # take a part of the traing set to apply validation
    train_batch_size = 128              # batch size of the training set
    val_batch_size = 100                # batch size of the validating set
    
    # Below are some augment methods. Set MIXUP_ALPHA to zero to disable the mixup augmentation, 
    # and a non-zero float number represents the alpha (here equal to beta) of the BETA distribution.
    # Set AUTOAUGMENT to TRUE to enable the auto-Augmentation introduced by Google.
    # Set mixup_alpha = 0 and autoAugment = False to use the baseline augment methods.
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
    Setup the network.
    ---------------------------------------------------------------------------
    """
    tf.reset_default_graph()
    
    # Feed the data with tf.data.Dataset.
    xs = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='xs')
    ys = tf.placeholder(tf.float32, shape=[None, classes], name='ys')
    batch_size = tf.placeholder(tf.int64, shape=[], name='batch_size')
    dataset = data.Dataset.from_tensor_slices((xs, ys)).batch(batch_size)
    data_loader = dataset.make_initializable_iterator()
    
    # Distinguish the training and testing states for BN and dropout.
    train_flag = tf.placeholder(tf.bool, shape=[], name='training_flag')
    img, label = data_loader.get_next(name='batch_loader')
    
    # The output of the Conv network, the last layer is a global average pooling operation.
    global_avg = net(img, train_flag, zero_pad=zero_pad)
    # Fully connected layer WITHOUT softmax activation.
    pred = layers.linear(global_avg, classes, activation=None, use_bias=True, name='prediction')
    # Loss without regularization.
    loss_no_reg = tf.losses.softmax_cross_entropy(label, pred)
    # Add the Loss with L2 regularization.
    reg_loss = []
    params = 0
    for var in tf.trainable_variables():
        if 'weight' in var.name:
            reg_loss.append(tf.nn.l2_loss(var))
            params += np.prod(var.get_shape().as_list())
    loss = tf.add(loss_no_reg, tf.multiply(weight_decay, tf.add_n(reg_loss)), name='loss_with_reg')
    
    # Define the optimizer.
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
    
    """
    ---------------------------------------------------------------------------
    Gather the variables to make a checkpoint.
    ---------------------------------------------------------------------------
    """
    tf.add_to_collection('batch_label', label)
    tf.add_to_collection('pred', pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('train_op', train_op)
    tf.add_to_collection('data_loader_initializer', data_loader.initializer)
    
    # The trainable variables like weights and biases.
    var_list = tf.trainable_variables()
    # Variables of the optimizer, note that the size is somewhat large to save, 
    # you can redefine an optimizer if you want to retrain the model with a checkpoint.
    if save_optim == True:
        var_list += optim.variables()
    
    # As the 'moving_mean' and 'moving_variance' (necessary for the inference in batch normalization) 
    # are not the trainable variables, we should fetch them from the global variables.
    global_var = tf.global_variables()
    moving_vars = [g for g in global_var if 'moving_mean' in g.name]
    moving_vars += [g for g in global_var if 'moving_variance' in g.name]
    var_list += moving_vars
    
    saver = tf.train.Saver(var_list=var_list, max_to_keep=5)
        
    """
    ---------------------------------------------------------------------------
    Config the devices and start training.
    ---------------------------------------------------------------------------
    """
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        
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
        print('Trainable parameters: {}'.format(params))
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
            sess.run(data_loader.initializer, feed_dict={xs: xs_train1, ys: ys_train1, batch_size: train_batch_size})
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
                _, loss_i, label_i, pred_i = sess.run([train_op, loss, label, pred], 
                                                      feed_dict={train_flag: True, lr: learning_rate(e), batch_size: train_batch_size})
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
            sess.run(data_loader.initializer, feed_dict={xs: xs_val, ys: ys_val, batch_size: val_batch_size})
            for i in range(val_batches):
                loss_val, label_val, pred_val = sess.run([loss, label, pred], feed_dict={train_flag: False, batch_size: val_batch_size})
                val_err[e] += 100.0 * np.sum(np.argmax(pred_val, axis=1) != np.argmax(label_val, axis=1))
                val_losses[e] += loss_val
            val_losses[e] /= val_batches
            val_err[e] /= (val_batches * val_batch_size)
            
            cur_time = time.time()
            h, m, s = util.parse_time(cur_time - begin)
            print('Epoch {}:   Val_loss = {:.3f},   Val_err = {:.2f}  ({} samples)  {:.2f}s   '.format(
                                        e+1, val_losses[e], val_err[e], val_size, cur_time-val_time))
            print('Global time has passed {}:{}:{:.2f}'.format(h, m, s))
            print('')
            
            # Make a checkpoint.
            if val_err[e] < best_val and e > 50:
                best_val = val_err[e]
                saver.save(sess, 'ckpt/model', global_step=e+1, write_meta_graph=True)
        
        # Make a final checkpoint.
        saver.save(sess, 'ckpt/model-final', write_meta_graph=True)
        print('Training time: {:.2f}'.format(time.time() - begin))
        util.save_training_result('ckpt/training_result', train_losses, train_err, val_losses, val_err)
        
    del(xs_train1, xs_val, xs_train)
    util.test('ckpt/model-final.meta', 'ckpt/model-final', xs_test, ys_test)





