# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 14:39:47 2018

@author: hzm
"""

import tensorflow as tf
from . import layers
    
def basicBlock(in_feat, stride, out_chans, is_training, dropout=0.0, zero_pad=False, name=None):
    with tf.variable_scope(name, default_name='BasicBlock'):
        bn0 = layers.batch_normalization(in_feat, training=is_training, name='bn0')
        relu0 = tf.nn.relu(bn0, name='relu0')
        
        conv1 = layers.conv2d(relu0, out_chans, kernel_size=[3, 3], strides=[stride]*2, name='conv1')
        bn1 = layers.batch_normalization(conv1, training=is_training, name='bn1')
        relu1 = tf.nn.relu(bn1, name='relu1')
        
        if dropout > 0.0:
            dropout0 = tf.layers.dropout(relu1, dropout, training=is_training, name='dropout0')
            conv2 = layers.conv2d(dropout0, out_chans, kernel_size=[3, 3], strides=[1, 1], name='conv2')
        else:
            conv2 = layers.conv2d(relu1, out_chans, kernel_size=[3, 3], strides=[1, 1], name='conv2')
        
        if in_feat.shape[1:] != conv2.shape[1:]:
            if zero_pad == False:
                short_cut = layers.conv2d(relu0, out_chans, kernel_size=[1, 1], strides=[stride]*2, name='conv_shortcut')
            else:
                short_cut = layers.zero_pad_shortcut(in_feat, out_chans, [stride]*2, name='pad_shortcut')
            out_feat = tf.add(conv2, short_cut, name='add')
        else:
            out_feat = tf.add(conv2, in_feat, name='add')
    return out_feat

def WideResNet(img, blocks, strides, k, chans, is_training, dropout=0.0, zero_pad=False):
    
    stride = []
    for b, s in zip(blocks, strides):
        stride += ([s] + [1] * (b - 1))
    with tf.variable_scope('Begin'):
        out = layers.conv2d(img, chans, kernel_size=[3, 3], strides=[1, 1], name='conv')
    
    chans *= k
    for i in range(len(stride)):
        chans *= stride[i]
        out = basicBlock(out, stride[i], chans, is_training, dropout=dropout, zero_pad=zero_pad, name='Block_'+str(i))
        
    with tf.variable_scope('End'):
        out = layers.batch_normalization(out, training=is_training, name='bn')
        out = tf.nn.relu(out, name='relu')
    out = tf.reduce_mean(out, axis=[1, 2], name='global_avg_pooling')
    return out

def WRN_40_2(img, is_training, dropout=0.0, zero_pad=False):
    return WideResNet(img, [6, 6, 6], [1, 2, 2], 2, 16, is_training, dropout, zero_pad=zero_pad)

def WRN_16_4(img, is_training, dropout=0.0, zero_pad=False):
    return WideResNet(img, [2, 2, 2], [1, 2, 2], 4, 16, is_training, dropout, zero_pad=zero_pad)

def WRN_28_10(img, is_training, dropout=0.0, zero_pad=False):
    return WideResNet(img, [4, 4, 4], [1, 2, 2], 10, 16, is_training, dropout, zero_pad=zero_pad)




