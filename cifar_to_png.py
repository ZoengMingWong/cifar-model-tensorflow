# -*- coding: utf-8 -*-

import numpy as np
import os, cPickle
import matplotlib.pyplot as plt

def cifar_to_png(src_dir, dataset='cifar10', dst_dir=None):
    """
    Encode the orignal cifar10 dataset to PNG images, and we can parse the label
    from the filename, e.g. train0_6.png indicates the first training image's label is 6.
    It is memory efficient if we can dynamically decode the images while needed.
    """
    if dataset == 'cifar10':
        xs_train = []
        ys_train = []
        for i in '12345':
            with open(src_dir+'/data_batch_'+i, 'rb') as f:
                train_data = cPickle.load(f)
                xs_train.append(train_data['data'])
                ys_train.extend(train_data['labels'])
        with open(src_dir+'/test_batch', 'rb') as f:
            test_data = cPickle.load(f)
            xs_test = test_data['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            ys_test = np.array(test_data['labels'])
        
        xs_train = np.vstack(tuple(xs_train))
        xs_train = xs_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        ys_train = np.array(ys_train)
        
    elif dataset == 'cifar100':
        with open(src_dir+'/train', 'rb') as f:
            train_data = cPickle.load(f)
            xs_test = train_data['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            ys_train = np.array(train_data['fine_labels'])
        with open(src_dir+'/test', 'rb') as f:
            test_data = cPickle.load(f)
            xs_test = test_data['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            ys_test = np.array(test_data['fine_labels'])
            
    else:
        print('Unexpected dataset name!')
        return None
    
    if dst_dir is None:
        dst_dir = src_dir + '/png'
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    os.mkdir(dst_dir+'/test')
    for i in range(ys_test.shape[0]):
        plt.imsave(dst_dir+'/test/test'+str(i)+'_'+str(ys_test[i])+'.png', xs_test[i])
    
    os.mkdir(dst_dir+'/train')
    for i in range(ys_train.shape[0]):
        plt.imsave(dst_dir+'/train/train'+str(i)+'_'+str(ys_train[i])+'.png', xs_train[i])
    return None
    
if __name__ == '__main__':
    
    src = '/home/hzm/cifar10'      # the path of the unzip cifar dataset
    dst = '/home/hzm/cifar_data'   # the destination path to save the images
    cifar_to_png(src, 'cifar10', dst)





