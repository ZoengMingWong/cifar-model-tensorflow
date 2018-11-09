# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 13:43:16 2018

@author: hzm
"""

import tensorflow as tf
from . import layers

def PreActBlock(in_feat, stride, out_chans, bottleneck, cardinality, is_training, block_id, weight_decay=1e-4):
    
    bn0 = layers.batch_normalization(in_feat, training=is_training, name='Block_'+str(block_id)+'_bn0')
    relu0 = tf.nn.relu(bn0, name='Block_'+str(block_id)+'_relu0')
    
    conv1 = layers.conv2d(relu0, bottleneck*cardinality, kernel_size=[1, 1], strides=[1, 1], activation=None, 
                          use_bias=False, weight_decay=weight_decay, name='Block_'+str(block_id)+'_conv1')
    bn1 = layers.batch_normalization(conv1, training=is_training, name='Block_'+str(block_id)+'_bn1')
    relu1 = tf.nn.relu(bn1, name='Block_'+str(block_id)+'_relu1')
    
    #Group convolution
    relu1_splits = tf.split(relu1, cardinality, axis=3, name='Block_'+str(block_id)+'_relu1_split')
    conv2 = []
    for i in range(cardinality):
        conv2.append(layers.conv2d(relu1_splits[i], bottleneck, kernel_size=[3, 3], strides=[stride]*2, activation=None, 
                                   padding='SAME', use_bias=False, weight_decay=weight_decay, name='Block_'+str(block_id)+'_conv2_split_'+str(i)))
                             
    conv2_groups = tf.concat(conv2, axis=3, name='Block_'+str(block_id)+'_conv2_concat')
    
    bn2 = layers.batch_normalization(conv2_groups, training=is_training, name='Block_'+str(block_id)+'_bn2')
    relu2 = tf.nn.relu(bn2, name='Block_'+str(block_id)+'_relu2')
    
    conv3 = layers.conv2d(relu2, out_chans, kernel_size=[1, 1], strides=[1, 1], activation=None, 
                          use_bias=False, weight_decay=weight_decay, name='Block_'+str(block_id)+'_conv3')
        
    if in_feat.shape[3] != out_chans:
        short_cut = layers.conv2d(relu0, out_chans, kernel_size=[1, 1], strides=[stride]*2, activation=None, 
                                  use_bias=False, weight_decay=weight_decay, name='Block_'+str(block_id)+'_short_cut_conv')
    else:
        short_cut = tf.identity(in_feat, name='Block_'+str(block_id)+'_short_cut')
    
    out_feat = tf.add(short_cut, conv3, name='Block_'+str(block_id)+'_out_feat')
    
    return out_feat
    
def BasicBlock(in_feat, stride, out_chans, bottleneck, cardinality, is_training, block_id, weight_decay=1e-4):
    
    conv1 = layers.conv2d(in_feat, bottleneck, kernel_size=[1, 1], strides=[1, 1], activation=None, 
                          use_bias=False, weight_decay=weight_decay, name='Block_'+str(block_id)+'_conv1')
    bn1 = layers.batch_normalization(conv1, training=is_training, name='Block_'+str(block_id)+'_bn1')
    relu1 = tf.nn.relu(bn1, name='Block_'+str(block_id)+'_relu1')
    
    relu1_splits = tf.split(relu1, cardinality, axis=3, name='Block_'+str(block_id)+'_relu_split')
    conv2 = []
    for i in range(cardinality):
        conv2.append(layers.conv2d(relu1_splits[i], bottleneck, kernel_size=[3, 3], strides=[stride]*2, activation=None, 
                                   padding='SAME', use_bias=False, weight_decay=weight_decay, name='Block_'+str(block_id)+'_conv2_split_'+str(i)))
                             
    conv2_groups = tf.concat(conv2, axis=3, name='Block_'+str(block_id)+'_conv2_concat')
    
    bn2 = layers.batch_normalization(conv2_groups, training=is_training, name='Block_'+str(block_id)+'_bn2')
    relu2 = tf.nn.relu(bn2, name='Block_'+str(block_id)+'_relu2')
    
    conv3 = layers.conv2d(relu2, out_chans, kernel_size=[1, 1], strides=[1, 1], activation=None, 
                          use_bias=False, weight_decay=weight_decay, name='Block_'+str(block_id)+'_conv3')
    bn3 = layers.batch_normalization(conv3, training=is_training, name='Block_'+str(block_id)+'_bn3')
        
    if in_feat.shape[3] != out_chans:
        short_cut_conv = layers.conv2d(in_feat, out_chans, kernel_size=[1, 1], strides=[stride]*2, activation=None, 
                                       use_bias=False, weight_decay=weight_decay, name='Block_'+str(block_id)+'_short_cut_conv')
        short_cut = layers.batch_normalization(short_cut_conv, training=is_training, name='Block_'+str(block_id)+'_short_cut_bn')
    else:
        short_cut = tf.identity(in_feat, name='Block_'+str(block_id)+'_short_cut')
    
    out_feat = tf.add(short_cut, bn3, name='Block_'+str(block_id)+'_out_feat')
    out_feat_relu = tf.nn.relu(out_feat, name='Block_'+str(block_id)+'_out_feat_relu')
    
    return out_feat_relu
    
def ResNeXt(img, blocks, strides, chans, bottleneck, cardinality, is_training, weight_decay=1e-4, preAct=True):
    
    stride = []
    for b, s in zip(blocks, strides):
        stride += ([s] + [1] * (b - 1))
    
    in_feat0 = layers.conv2d(img, chans, kernel_size=[3, 3], strides=[1, 1], activation=None, 
                             padding='SAME', use_bias=False, weight_decay=weight_decay, name='img_conv')
    in_feat0_bn = layers.batch_normalization(in_feat0, training=is_training, name='img_bn')
    in_feat0_relu = tf.nn.relu(in_feat0_bn, name='img_relu')   
        
    resBlock = [in_feat0_relu]
    chans *= 4
    
    for i in range(len(stride)):
        chans *= stride[i]
        bottleneck *= stride[i]
        if preAct == True:
            resBlock.append(PreActBlock(resBlock[-1], stride[i], chans, bottleneck, cardinality, is_training, block_id=i, weight_decay=weight_decay))
        else:
            resBlock.append(BasicBlock(resBlock[-1], stride[i], chans, bottleneck, cardinality, is_training, block_id=i, weight_decay=weight_decay))
        
    global_avg = tf.reduce_mean(resBlock[-1], axis=[1, 2], name='global_avg_pooling')
    pred = layers.linear(global_avg, 10, activation=None, use_bias=True, weight_decay=weight_decay, name='prediction')
    
    return pred
    
def ResNeXt29_32x4d(img, is_training, weight_decay=1e-4):
    return ResNeXt(img, [3, 3, 3], [1, 2, 2], chans=64, bottleneck=4, cardinality=32, is_training=is_training, weight_decay=weight_decay)
    
def ResNeXt29_16x4d(img, is_training, weight_decay=1e-4):
    return ResNeXt(img, [3, 3, 3], [1, 2, 2], chans=32, bottleneck=4, cardinality=16, is_training=is_training, weight_decay=weight_decay)



