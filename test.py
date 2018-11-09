# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 15:33:40 2018

@author: hzm
"""
import numpy as np
import os
import util

os.environ['CUDA_DEVICES_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':
    
    xs_test = np.array(['/home/hzm/cifar_data/test/' + f for f in os.listdir('/home/hzm/cifar_data/test/')])
    ys_test = np.array([int(f[-5]) for f in os.listdir('/home/hzm/cifar_data/test/')])
    
    util.test('ckpt1/model-final.meta', 'ckpt1/model-fianl', xs_test, ys_test)
