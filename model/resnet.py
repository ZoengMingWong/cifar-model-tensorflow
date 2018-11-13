# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 15:59:46 2018

@author: hzm
"""

import tensorflow as tf
from . import layers

def PreActBlock(in_feat, stride, out_chans, is_training, block_id):
    
    bn0 = layers.batch_normalization(in_feat, training=is_training, name='Block_'+str(block_id)+'_bn0')
    relu0 = tf.nn.relu(bn0, name='Block_'+str(block_id)+'_relu0')
    
    conv1 = layers.conv2d(relu0, out_chans, kernel_size=[3, 3], strides=[stride]*2, name='Block_'+str(block_id)+'_conv1')
    bn1 = layers.batch_normalization(conv1, training=is_training, name='Block_'+str(block_id)+'_bn1')
    relu1 = tf.nn.relu(bn1, name='Block_'+str(block_id)+'_relu1')
    
    conv2 = layers.conv2d(relu1, out_chans, kernel_size=[3, 3], strides=[1, 1], name='Block_'+str(block_id)+'_conv2')
    
    if in_feat.shape[3] != out_chans:
        short_cut = layers.conv2d(relu0, out_chans, kernel_size=[1, 1], strides=[stride]*2, name='Block_'+str(block_id)+'_short_cut_Conv')
    else:
        short_cut = tf.identity(in_feat, name='Block_'+str(block_id)+'_short_cut')
    
    out_feat = tf.add(short_cut, conv2, name='Block_'+str(block_id)+'_out_feat')
    
    return out_feat
    
    
def BasicBlock(in_feat, stride, out_chans, is_training, block_id):
    
    conv1 = layers.conv2d(in_feat, out_chans, kernel_size=[3, 3], strides=[stride]*2, name='Block_'+str(block_id)+'_conv1')
    bn1 = layers.batch_normalization(conv1, training=is_training, name='Block_'+str(block_id)+'_bn1')
    relu1 = tf.nn.relu(bn1, name='Block_'+str(block_id)+'_relu1')
    
    conv2 = layers.conv2d(relu1, out_chans, kernel_size=[3, 3], strides=[1, 1], name='Block_'+str(block_id)+'_conv2')
    bn2 = layers.batch_normalization(conv2, training=is_training, name='Block_'+str(block_id)+'_bn2')
        
    if in_feat.shape[3] != out_chans:
        short_cut_conv = layers.conv2d(in_feat, out_chans, kernel_size=[1, 1], strides=[stride]*2, name='Block_'+str(block_id)+'_short_cut_conv')
        short_cut = layers.batch_normalization(short_cut_conv, training=is_training, name='Block_'+str(block_id)+'_short_cut_bn')
    else:
        short_cut = tf.identity(in_feat, name='Block_'+str(block_id)+'_short_cut')
    
    out_feat = tf.add(short_cut, bn2, name='Block_'+str(block_id)+'_out_feat')
    out_feat_relu = tf.nn.relu(out_feat, name='Block_'+str(block_id)+'_out_feat_relu')
    
    return out_feat_relu
    
def PreActBottleneck(in_feat, stride, bottleneck, is_training, block_id):
    
    bn0 = layers.batch_normalization(in_feat, training=is_training, name='Block_'+str(block_id)+'_bn0')
    relu0 = tf.nn.relu(bn0, name='Block_'+str(block_id)+'_relu0')
    
    conv1 = layers.conv2d(relu0, bottleneck, kernel_size=[1, 1], strides=[1, 1], name='Block_'+str(block_id)+'_conv1')
    bn1 = layers.batch_normalization(conv1, training=is_training, name='Block_'+str(block_id)+'bn1')
    relu1 = tf.nn.relu(bn1, name='Block_'+str(block_id)+'_relu1')
    
    conv2 = layers.conv2d(relu1, bottleneck, kernel_size=[3, 3], strides=[stride]*2, name='Block_'+str(block_id)+'_conv2')
    bn2 = layers.batch_normalization(conv2, training=is_training, name='Block_'+str(block_id)+'_bn2')
    relu2 = tf.nn.relu(bn2, name='Block_'+str(block_id)+'_relu2')
    
    conv3 = layers.conv2d(relu2, bottleneck*4, kernel_size=[1, 1], strides=[1, 1], name='Block_'+str(block_id)+'_conv3')
        
    if in_feat.shape[3] != (bottleneck * 4):
        short_cut = layers.conv2d(relu0, bottleneck*4, kernel_size=[1, 1], strides=[stride]*2, name='Block_'+str(block_id)+'_short_cut_Conv')
    else:
        short_cut = tf.identity(in_feat, name='Block_'+str(block_id)+'_short_cut')
    
    out_feat = tf.add(short_cut, conv3, name='Block_'+str(block_id)+'out_feat')
    
    return out_feat
    
def Bottleneck(in_feat, stride, bottleneck, is_training, block_id):
    
    conv1 = layers.conv2d(in_feat, bottleneck, kernel_size=[1, 1], strides=[1, 1], name='Block_'+str(block_id)+'_conv1')
    bn1 = layers.batch_normalization(conv1, training=is_training, name='Block_'+str(block_id)+'_bn1')
    relu1 = tf.nn.relu(bn1, name='Block_'+str(block_id)+'_relu1')
    
    conv2 = layers.conv2d(relu1, bottleneck, kernel_size=[3, 3], strides=[stride]*2, name='Block_'+str(block_id)+'_conv2')
    bn2 = layers.batch_normalization(conv2, training=is_training, name='Block_'+str(block_id)+'_bn2')
    relu2 = tf.nn.relu(bn2, name='Block_'+str(block_id)+'_reku2')
    
    conv3 = layers.conv2d(relu2, bottleneck*4, kernel_size=[1, 1], strides=[1, 1], name='Block_'+str(block_id)+'_conv3')
    bn3 = layers.batch_normalization(conv3, training=is_training, name='Block_'+str(block_id)+'_bn3')
        
    if in_feat.shape[3] != (bottleneck * 4):
        short_cut_conv = layers.conv2d(in_feat, bottleneck*4, kernel_size=[1, 1], strides=[stride]*2, name='Block_'+str(block_id)+'_short_cut_conv')
        short_cut = layers.batch_normalization(short_cut_conv, training=is_training, name='Block_'+str(block_id)+'_short_cut_bn')
    else:
        short_cut = tf.identity(in_feat, name='Block_'+str(block_id)+'_short_cut')
    
    out_feat = tf.add(short_cut, bn3, name='Block_'+str(block_id)+'_out_feat')
    out_feat_relu = tf.nn.relu(out_feat, name='Block_'+str(block_id)+'_out_feat_relu')
    
    return out_feat_relu
    
def ResNet(img, blocks, strides, chans, is_training, preAct=True):
    
    stride = []
    for b, s in zip(blocks, strides):
        stride += ([s] + [1] * (b - 1))
    
    out = layers.conv2d(img, chans, kernel_size=[3, 3], strides=[1, 1], name='img_conv')
    
    for i in range(len(stride)):
        chans *= stride[i]
        if preAct == True:
            out = PreActBlock(out, stride[i], chans, is_training, block_id=i)
        else:
            out = BasicBlock(out, stride[i], chans, is_training, block_id=i)
    
    out = layers.batch_normalization(out, training=is_training, name='end_bn')
    out = tf.nn.relu(out, name='end_relu')
    out = tf.reduce_mean(out, axis=[1, 2], name='global_avg_pooling')
    return out
    
def BottleneckResNet(img, blocks, strides, chans, bottleneck, is_training, preAct=True):
    
    stride = []
    for b, s in zip(blocks, strides):
        stride += ([s] + [1] * (b - 1))
    
    out = layers.conv2d(img, chans, kernel_size=[3, 3], strides=[1, 1], name='img_conv')
    
    for i in range(len(stride)):
        bottleneck *= stride[i]
        if preAct == True:
            out = PreActBottleneck(out, stride[i], bottleneck, is_training, block_id=i)
        else:
            out = Bottleneck(out, stride[i], bottleneck, is_training, block_id=i)
    
    out = layers.batch_normalization(out, training=is_training, name='end_bn')
    out = tf.nn.relu(out, name='end_relu')
    out = tf.reduce_mean(out, axis=[1, 2], name='global_avg_pooling')
    return out

def ResNet18(img, is_training):
    return ResNet(img, [2, 2, 2, 2], [1, 2, 2, 2], 64, is_training=is_training, preAct=False)
    
def PreResNet18(img, is_training):
    return ResNet(img, [2, 2, 2, 2], [1, 2, 2, 2], 64, is_training=is_training, preAct=True)
    
def ResNet50(img, is_training):
    return BottleneckResNet(img, [3, 4, 6, 3], [1, 2, 2, 2], 64, 64, is_training=is_training, preAct=False)
    
def PreResNet50(img, is_training):
    return BottleneckResNet(img, [3, 4, 6, 3], [1, 2, 2, 2], 64, 64, is_training=is_training, preAct=True)




