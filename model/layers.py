# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 16:23:26 2018

@author: hzm
"""
import tensorflow as tf

def conv2d(in_feat, out_chans, kernel_size, strides, padding='SAME', activation=None, use_bias=False, name=None):
    shape = kernel_size + [in_feat.shape[-1].value, out_chans]
    strides = [1] + strides + [1]
    elements = float(kernel_size[0] * kernel_size[1] * out_chans)
    stdv = tf.sqrt(2.0 / elements)
    with tf.name_scope(name=name):
        weight = tf.Variable(tf.random_normal(shape, stddev=stdv, seed=0), trainable=True, name='weight')
        
        conv = tf.nn.conv2d(in_feat, weight, strides, padding, name='conv')
        
        if use_bias:
            bias = tf.Variable(tf.constant(0.0, shape=[out_chans]), trainable=True, name='bias')
            conv = tf.add(conv, bias, name='add_bias')
        if activation is not None:
            conv = activation(conv, name='activation')
    return conv
    
def linear(in_feat, out_classes, activation=False, use_bias=True, name=None):
    shape = [in_feat.shape[-1].value, out_classes]
    stdv = 1.0 / tf.sqrt(float(out_classes))
    with tf.name_scope(name=name):
        weight = tf.Variable(tf.random_uniform(shape, -stdv, stdv, seed=0), trainable=True, name='weight')
        
        full = tf.matmul(in_feat, weight, name='full_connect')
        
        if use_bias:
            bias = tf.Variable(tf.constant(0.0, shape=[out_classes]), trainable=True, name='bias')
            full = tf.add(full, bias, name='add_bias')
        if activation is not None:
            full = activation(full, name='activation')
    return full

def batch_normalization(in_feat, training, name=None):
    out_feat = tf.layers.batch_normalization(in_feat, epsilon=1e-5, momentum=0.9, training=training, name=name)
    return out_feat

def zero_pad_shortcut(in_feat, out_chans, strides, name=None):
    if strides != [1, 1]:
        in_feat = tf.layers.average_pooling2d(in_feat, strides, strides, 'SAME', name=name+'_avg_pool')
    in_chans = in_feat.shape[-1].value
    before = (out_chans - in_chans) // 2
    after = out_chans - in_chans - before
    out = tf.pad(in_feat, paddings=[[0, 0], [0, 0], [0, 0], [before, after]], name=name+'_zero_pad')
    return out
    
def conv_shortcut(in_feat, out_chans, strides, training=False, name=None):
    out = conv2d(in_feat, out_chans, [1, 1], strides, name=name+'_conv')
    out = batch_normalization(out, training=training, name=name+'_bn')
    out = tf.nn.relu(out, name=name+'_relu')
    return out
    





