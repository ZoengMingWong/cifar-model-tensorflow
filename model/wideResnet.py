# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 14:39:47 2018

@author: hzm
"""

import tensorflow as tf
from . import layers
    
def basicBlock(in_feat, stride, out_chans, is_training, block_id, dropout=0.3, weight_decay=1e-4):
    
    bn0 = layers.batch_normalization(in_feat, training=is_training, name='Block_'+str(block_id)+'_bn0')
    relu0 = tf.nn.relu(bn0, name='Block_'+str(block_id)+'_relu0')
    
    conv1 = layers.conv2d(relu0, out_chans, kernel_size=[3, 3], strides=[stride]*2, padding='SAME', 
                          activation=None, use_bias=False, weight_decay=weight_decay, name='Block_'+str(block_id)+'_conv1')
    bn1 = layers.batch_normalization(conv1, training=is_training, name='Block_'+str(block_id)+'_bn1')
    relu1 = tf.nn.relu(bn1, name='Block_'+str(block_id)+'_relu1')
    
    if dropout > 0.0:
        dropout0 = tf.layers.dropout(relu1, 0.3, training=is_training, name='Block_'+str(block_id)+'_dropout0')
    else:
        dropout0 = tf.identity(relu1, name='Block_'+str(block_id)+'_no_dropout')
    
    conv2 = layers.conv2d(dropout0, out_chans, kernel_size=[3, 3], strides=[1, 1], padding='SAME', 
                          activation=None, use_bias=False, weight_decay=weight_decay, name='Block_'+str(block_id)+'_conv2')
    
    if in_feat.shape[3] != out_chans:
        short_cut = layers.conv2d(relu0, out_chans, kernel_size=[1, 1], strides=[stride]*2, 
                                  activation=None, use_bias=False, weight_decay=weight_decay, name='Block_'+str(block_id)+'_short_cut_Conv')
    else:
        short_cut = tf.identity(in_feat, name='Block_'+str(block_id)+'_short_cut')
    
    out_feat = tf.add(short_cut, conv2, name='Block_'+str(block_id)+'_out_feat')
    
    return out_feat

def WideResNet(img, blocks, strides, k, chans, is_training, weight_decay=1e-4, dropout=0.3):
    
    stride = []
    for b, s in zip(blocks, strides):
        stride += ([s] + [1] * (b - 1))
    
    in_feat0_conv = layers.conv2d(img, chans, kernel_size=[3, 3], strides=[1, 1], padding='SAME', 
                                  activation=None, use_bias=False, weight_decay=weight_decay, name='img_conv')
    in_feat0_bn = layers.batch_normalization(in_feat0_conv, training=is_training, name='img_bn')
    in_feat0_relu = tf.nn.relu(in_feat0_bn, name='img_relu')
    
    chans *= k
    resBlock = [in_feat0_relu]
    for i in range(len(stride)):
        chans *= stride[i]
        resBlock.append(basicBlock(resBlock[-1], stride[i], chans, is_training, block_id=i, dropout=dropout, weight_decay=weight_decay))
    
    global_avg = tf.reduce_mean(resBlock[-1], axis=[1, 2], name='global_avg_pooling')
    pred = layers.linear(global_avg, 10, activation=None, use_bias=True, weight_decay=weight_decay, name='prediction')
    
    return pred

def WRN_40_2(img, is_training, weight_decay=1e-4, dropout=0.3):
    return WideResNet(img, [6, 6, 6], [1, 2, 2], 2, 16, is_training, weight_decay, dropout)

def WRN_16_4(img, is_training, weight_decay=1e-4, dropout=0.3):
    return WideResNet(img, [2, 2, 2], [1, 2, 2], 4, 16, is_training, weight_decay, dropout)

def WRN_28_10(img, is_training, weight_decay=1e-4, dropout=0.3):
    return WideResNet(img, [4, 4, 4], [1, 2, 2], 10, 16, is_training, weight_decay, dropout)




