# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 15:33:40 2018

@author: hzm
"""
import numpy as np
import os
import util

# Notice that the VISIBLE_DEVICES must NOT LESS THAN the gpus used in the checkpoint.
os.environ['CUDA_DEVICES_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'

if __name__ == '__main__':
    
    data_path = '/home/hzm/cifar_data'
    xs_test = np.array([data_path + '/test/' + f for f in os.listdir(data_path + '/test/')])
    ys_test = np.array([int(f[-5]) for f in os.listdir(data_path + '/test/')])
    
    batch_size = 100
    util.test('ckpt/model-1.meta', 'ckpt/model-1', xs_test, ys_test, batch_size)
