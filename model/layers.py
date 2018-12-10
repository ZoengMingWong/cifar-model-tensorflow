# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 16:23:26 2018

@author: hzm
"""
import tensorflow as tf

def conv2d(in_feat, out_chans, kernel_size, strides, padding='SAME', activation=None, use_bias=False, name=None):
    with tf.variable_scope(name, 'Conv', [in_feat], reuse=tf.AUTO_REUSE):
        shape = kernel_size + [in_feat.shape[-1].value, out_chans]
        strides = [1] + strides + [1]
        elements = float(kernel_size[0] * kernel_size[1] * out_chans)
        stdv = tf.sqrt(2.0 / elements)
        weight_init = tf.random_uniform_initializer(-stdv, stdv, seed=None)
        weight = tf.get_variable('weight', shape, initializer=weight_init, trainable=True)
        conv = tf.nn.conv2d(in_feat, weight, strides, padding, name='conv')
        
        if use_bias:
            bias_init = tf.constant_initializer(0.0)
            bias = tf.get_variable('bias', [out_chans], initializer=bias_init, trainable=True)
            conv = tf.add(conv, bias, name='add_bias')
        if activation is not None:
            conv = activation(conv, name='activation')
    return conv
    
def linear(in_feat, out_classes, activation=False, use_bias=True, name=None):
    with tf.variable_scope(name, 'Fully', [in_feat], reuse=tf.AUTO_REUSE):
        shape = [in_feat.shape[-1].value, out_classes]
        stdv = 1.0 / tf.sqrt(float(out_classes))
        weight_init = tf.random_uniform_initializer(-stdv, stdv, seed=None)
        weight = tf.get_variable('weight', shape, initializer=weight_init, trainable=True)
        
        full = tf.matmul(in_feat, weight, name='fully_connect')
        
        if use_bias:
            bias_init = tf.constant_initializer(0.0)
            bias = tf.get_variable('bias', [out_classes], initializer=bias_init, trainable=True)
            full = tf.add(full, bias, name='add_bias')
        if activation is not None:
            full = activation(full, name='activation')
    return full

def batch_normalization(in_feat, training, name=None):
    out_feat = tf.layers.batch_normalization(in_feat, epsilon=1e-5, momentum=0.9, training=training, name=name)
    return out_feat

def zero_pad_shortcut(in_feat, out_chans, strides, name=None):
    with tf.name_scope(name):
        if strides != [1, 1]:
            in_feat = tf.layers.average_pooling2d(in_feat, strides, strides, 'SAME', name='avg_pool')
        in_chans = in_feat.shape[-1].value
        b = (out_chans - in_chans) // 2
        e = out_chans - in_chans - b
        out = tf.pad(in_feat, paddings=[[0, 0], [0, 0], [0, 0], [b, e]], name='zero_pad')
    return out
    
def conv_shortcut(in_feat, out_chans, strides, training=False, name=None):
    with tf.variable_scope(name):
        out = conv2d(in_feat, out_chans, [1, 1], strides, name='conv')
        out = batch_normalization(out, training=training, name='bn')
    return out
    
def shake(in_feat, training, bern_prob=1., alpha=[-1., 1.], beta=[0., 1.], name=None):
    batch_size = in_feat.shape[0].value
    if batch_size is None:
        batch_size = tf.get_collection('dev_batch_size')[0]
    with tf.name_scope(name):
        assert alpha[1] > alpha[0]
        assert beta[1] > beta[0]
        def train_shake():
            rnd_shape = [batch_size, 1, 1, 1]
            rnd_alpha = tf.random_uniform(rnd_shape, alpha[0], alpha[1], name='rnd_alpha', seed=None)
            rnd_beta = tf.random_uniform(rnd_shape, beta[0], beta[1], name='rnd_beta', seed=None)
            rnd_bern = tf.floor(tf.random_uniform(rnd_shape, seed=None) + bern_prob, name='rnd_bern')
            rnd_forward = tf.add(rnd_bern, (1.0 - rnd_bern) * rnd_alpha, name='rnd_forward')
            rnd_backward = tf.add(rnd_bern, (1.0 - rnd_bern) * rnd_beta, name='rnd_backward')
            shake = tf.add(in_feat * rnd_backward, tf.stop_gradient((in_feat * (rnd_forward - rnd_backward))), name='train_shake')
            return shake
            
        def test_shake():
            E_shake = bern_prob + (1.0 - bern_prob) * (alpha[1] + alpha[0]) / 2.0
            return tf.multiply(in_feat, E_shake, name='test_shake')
        shake = tf.cond(training, train_shake, test_shake, name='shake')
    return shake
    





