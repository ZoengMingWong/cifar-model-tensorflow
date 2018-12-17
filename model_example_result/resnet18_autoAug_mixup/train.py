# -*- coding: utf-8 -*-

from __future__ import division, print_function
import os, sys, time, re
import tensorflow as tf
from tensorflow import data
import numpy as np
from multiprocessing import Pool
from model import layers, resnet, resnext, wideResnet, pyramidNet, shakeDrop
import util

os.environ['CUDA_DEVICES_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'    # the GPU(s) can be used

# If you don't config the GPUs, TensorFlow would occupy all free memory of the visible GPUs.
# Set the `per_process_gpu_memory_fraction` to restrict the usable memory.
# Set the `allow_growth` the let it self-adaptively fetch the memory, however some memory
# would be wasted as the program would apply for more memory than it actually need.
# Empirically, 3GB for ResNet-18 and 7GB for WRN-28-10 is enough.
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.25
#config.gpu_options.allow_growth = True

# Below are some typical models in papers, you can define some other models yourself.
# The model must return the features before the fully connected layer (convolution net).
model_dict = {'ResNet18': resnet.ResNet18, 
              'PreResNet18': resnet.PreResNet18, 
              'PreResNet50': resnet.PreResNet50, 
              'PreResNet101': resnet.PreResNet101, 
              'PreResNeXt29_32x4d': resnext.ResNeXt29_32x4d, 
              'PreResNeXt50_32x4d': resnext.ResNeXt50_32x4d, 
              'WRN_28_10': wideResnet.WRN_28_10, 
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
    # You can define the learning rate foe every epoch or even every batch, 
    # by feeding the `lr` in the feed_dict (see the Training Stage codes).
    # By default we use the cosine decay rate without restart for every batch.
    # Below is the learining rate used in origingal papers for ResNet.
    # learning_rate = lambda e: init_lr if e < 100 else (init_lr / 10) if e < 150 else (init_lr / 100)
    
    optimizer = tf.train.MomentumOptimizer  # the optimizer to use
    weight_decay = 1e-4                 # weight decay with L2 regularization, you can use 5e-4 for WRN.
    grad_clip = 5.0                     # gradient clipping to avoid exploration
    zero_pad = False                    # whether zero-padding or 1x1 conv for the shortcut channels mapping
    momentum = 0.9                      # the momentum of the Momentum optimizer
    use_nesterov = True                 # whether use the Nesterov Momentum Optimizer
    save_optim = False                  # whether save the variables of the optimizer while making a checkpoint
    
    use_val = False                     # whether apply the validation with part of training set
    val_ratio = 0.1                     # take a part of the traing set to apply validation
    train_batch_size = 128              # batch size of the training set
    val_batch_size = 100                # batch size of the validating set
    
    # Below are some augment methods, read the corresponding papers for details. 
    # Set `mixip_alpha` zero to disable the mixup augmentation, 
    # and a non-zero float number represents the alpha (here equal to beta) of the BETA distribution.
    # Set `autoAugment` TRUE to enable the auto-Augmentation.
    # Set mixup_alpha = 0 and autoAugment = False to use the baseline augment methods.
    # Generally, autoAugmentation is better than mixup, and even better with both two.
    mixup_alpha = 1.0
    autoAugment = True
    
    """
    ---------------------------------------------------------------------------
    Gather the dataset.
    ---------------------------------------------------------------------------
    """
    # xs is a numpy array of STRINGs, and ys is a numpy array of INTs.
    fs = os.listdir(os.path.join(data_path, 'train'))
    xs_train = np.array([os.path.join(data_path, 'train', f) for f in fs])
    ys_train = np.array([int(re.split('[_.]', f)[1]) for f in fs])
    
    fs = os.listdir(os.path.join(data_path, 'test'))
    xs_test = np.array([os.path.join(data_path, 'test', f) for f in fs])
    ys_test = np.array([int(re.split('[_.]', f)[1]) for f in fs])
    
    if use_val == True:
        # Take a part of the traing set as the validation set.
        val_size = int(ys_train.shape[0] * val_ratio)
        val_batches = val_size // val_batch_size
        val_size = val_batches * val_batch_size
        
        rnd = np.random.permutation(range(ys_train.shape[0]))
        xs_val, xs_train = xs_train[rnd[:val_size]], xs_train[rnd[val_size:]]
        ys_val, ys_train = ys_train[rnd[:val_size]], ys_train[rnd[val_size:]]
    else:
        xs_val, ys_val = np.copy(xs_test), np.copy(ys_test)
        val_size = xs_val.shape[0]
        val_batches = val_size // val_batch_size
        
    train_batches = xs_train.shape[0] // train_batch_size
    
    ###########################################################################
    # Preprocess the validating images with multiprocessing.
    procs = 5
    splits = [(xs_val.shape[0] // procs) * i for i in range(procs)[1:]]
    xs_val = np.split(xs_val, splits)
    ys_val = np.split(ys_val, splits)
    
    pool = Pool(procs)
    results = []
    for i in range(procs):
        results.append(pool.apply_async(util.batch_parse, (xs_val[i], ys_val[i], False, mixup_alpha, autoAugment, classes)))
        
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
    Setup the compute graph.
    ---------------------------------------------------------------------------
    """
    tf.reset_default_graph()
    
    with tf.name_scope('Data_loader'):
        # Feed the data with tf.data.Dataset.
        xs = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='xs')
        ys = tf.placeholder(tf.float32, shape=[None, classes], name='ys')
        batch_size = tf.placeholder(tf.int64, shape=[], name='batch_size')
        dataset = data.Dataset.from_tensor_slices((xs, ys)).batch(batch_size)
        data_loader = dataset.make_initializable_iterator()
        img, label = data_loader.get_next(name='batch_loader')
        
        # The actual batch size on each device. if you use multiGPUs, the total batch should be splited.
        tf.add_to_collection('dev_batch_size', tf.to_int32(batch_size))
    
    # Distinguish the training and testing states for BN and dropout.
    train_flag = tf.placeholder(tf.bool, shape=[], name='training_flag')
    # The output of the Conv network, the last layer is a global average pooling operation.
    global_avg = net(img, train_flag, zero_pad=zero_pad)
    # Fully connected layer WITHOUT softmax activation.
    pred = layers.linear(global_avg, classes, activation=None, use_bias=True, name='prediction')
    
    with tf.name_scope('Loss'):
        # Loss without regularization.
        loss_no_reg = tf.losses.softmax_cross_entropy(label, pred)
        # Add the Loss with L2 regularization (weight_decay).
        reg_loss = []
        for var in tf.trainable_variables():
            if ('weight' in var.name) or ('bias' in var.name):
                reg_loss.append(tf.nn.l2_loss(var))
        loss = tf.add(loss_no_reg, tf.multiply(weight_decay, tf.add_n(reg_loss)), name='loss_with_reg')
        # gather the error rate
        num_err = tf.reduce_sum(tf.to_float(tf.not_equal(tf.argmax(pred, axis=1), tf.argmax(label, axis=1))))
        error = tf.divide(100.0 * num_err, tf.to_float(batch_size), name='error')
    
    # Define the optimizer.
    lr = tf.placeholder(tf.float32, shape=[], name='lr')
    if optimizer is tf.train.MomentumOptimizer:
        optim = optimizer(lr, momentum=momentum, use_nesterov=use_nesterov)
    else:
        optim = optimizer(lr)
    
    # Apply the backpropagation with gradient clipping.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        with tf.name_scope('grads_clip'):
            grads, variables = zip(*optim.compute_gradients(loss))
            grads, global_norm = tf.clip_by_global_norm(grads, grad_clip)
        train_op = optim.apply_gradients(zip(grads, variables))
    
    """
    ---------------------------------------------------------------------------
    Gather the variables to make a checkpoint.
    ---------------------------------------------------------------------------
    """
    # Adding the variables or operations to collections make it easy to reuse with a checkpoint.
    tf.add_to_collection('batch_label', label)
    tf.add_to_collection('pred', pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('error', error)
    tf.add_to_collection('train_op', train_op)
    tf.add_to_collection('data_loader_initializer', data_loader.initializer)
    
    # The trainable variables like weights and biases.
    var_list = tf.trainable_variables()
    params = 0      # the total training parameters representing the model complexity.
    for var in var_list:
        params += np.prod(var.get_shape().as_list())
    
    # Variables of the optimizer, note that the size is somewhat large to save.
    # We have no need to save them in the inference stage, 
    # you can redefine an optimizer if you want to retrain the model with a checkpoint.
    if save_optim == True:
        var_list += optim.variables()
    
    # As the 'moving_mean' and 'moving_variance' (necessary for the inference in batch normalization) 
    # are not the trainable variables, we should fetch them from the global variables.
    global_var = tf.global_variables()
    moving_vars = [g for g in global_var if 'moving_mean' in g.name]
    moving_vars += [g for g in global_var if 'moving_variance' in g.name]
    var_list += moving_vars
    
    # If we don't config the `var_list`, by default all saveable variables will be saved.
    saver = tf.train.Saver(var_list=var_list, max_to_keep=5)
        
    """
    ---------------------------------------------------------------------------
    Config the devices and start training.
    ---------------------------------------------------------------------------
    """
    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        
        #######################################################################
        # Augment the training data with multiprocess.
        print('Data preparing ...')
        procs = 5
        splits = [(xs_train.shape[0] // procs) * i for i in range(procs)[1:]]
        xs_train1 = np.split(xs_train, splits)
        ys_train1 = np.split(ys_train, splits)
        
        pool = Pool(procs)
        results = []
        for i in range(procs):
            results.append(pool.apply_async(util.batch_parse, (xs_train1[i], ys_train1[i], True, mixup_alpha, autoAugment, classes)))
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
            # Initialize the dataset with training data.
            sess.run(data_loader.initializer, feed_dict={xs: xs_train1, ys: ys_train1, batch_size: train_batch_size})
            
            ###################################################################
            # Shuffle and disturb the training data every epoch with another process to save time.
            np.random.shuffle(rnd_samples)
            xs_train, ys_train = xs_train[rnd_samples], ys_train[rnd_samples]
            
            pool = Pool(1)
            result = pool.apply_async(util.batch_parse, (xs_train, ys_train, True, mixup_alpha, autoAugment, classes))
            pool.close()
            ###################################################################
            
            for i in range(train_batches):
                # cosine learning rate decay without restart
                lr_batch = 0.5 * init_lr * (1 + np.cos(np.pi * (e * train_batches + i) / (epochs * train_batches)))
                batch_time = time.time()
                _, loss_i, err_i = sess.run([train_op, loss, error], feed_dict={train_flag: True, lr: lr_batch, batch_size: train_batch_size})
                train_losses[e] += loss_i
                train_err[e] += err_i
                
                sys.stdout.write('Epoch {}: {} / {} batches.  Error: {:.2f}  Loss: {:.3f}  {:.2f}s   '.format(
                                    e+1, i+1, train_batches, err_i, loss_i, time.time()-batch_time) + '\r')
                sys.stdout.flush()
                
            train_losses[e] /= train_batches
            train_err[e] /= train_batches
            print('')
            print('Epoch {}: Train_loss = {:.3f}, Train_err = {:.2f}'.format(e+1, train_losses[e], train_err[e]))
            
            # Update the training data.
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
            # Initialize the dataset with validating data.
            sess.run(data_loader.initializer, feed_dict={xs: xs_val, ys: ys_val, batch_size: val_batch_size})
            for i in range(val_batches):
                loss_val_i, err_val_i = sess.run([loss, error], feed_dict={train_flag: False, batch_size: val_batch_size})
                val_err[e] += err_val_i
                val_losses[e] += loss_val_i
            
            val_losses[e] /= val_batches
            val_err[e] /= val_batches
            
            cur_time = time.time()
            h, m, s = util.parse_time(cur_time - begin)
            print('Epoch {}:   Val_loss = {:.3f},   Val_err = {:.2f}  ({} samples)  {:.2f}s   '.format(
                                        e+1, val_losses[e], val_err[e], val_size, cur_time-val_time))
            print('Global time has passed {}:{}:{:.2f}'.format(h, m, s))
            print('')
            
            # Make a checkpoint.
            util.save_epoch_result('train_result', e, train_losses[e], train_err[e], val_losses[e], val_err[e])
            if val_err[e] < best_val:
                best_val = val_err[e]
                saver.save(sess, 'ckpt/model'.format(10*val_err[e]), global_step=e+1, write_meta_graph=True)
        
        # Make a final checkpoint.
        saver.save(sess, 'ckpt/model-final', write_meta_graph=True)
        print('Training time: {:.2f}'.format(time.time() - begin))
        util.plot_training_result('train_result', train_losses, train_err, val_losses, val_err)
        
    del(xs_train1, xs_val, xs_train)
    test_loss, test_err = util.test('ckpt/model-final.meta', 'ckpt/model-final', xs_test, ys_test, val_batch_size, classes)
    f = open('train_result.txt', 'a')
    print('Test_loss = {:.3f},  Test_err = {:.2f}'.format(test_loss, test_err), file=f)
    f.close()





