#!/usr/bin/env python
# -*- coding:utf-8 -*-
# title            : nn_skeleton.py
# description      : define all the modules of neural network
# author           : Zhijun Tu
# email            : tzj19970116@163.com
# date             : 2017/07/16
# version          : 1.0
# notes            : 
# python version   : 2.7.12 which is also applicable to 3.5
# ============================================================== #
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import sys
import math
import numpy as np
import tensorflow as tf
from easydict import EasyDict as edict
from tensorflow.python.training import moving_averages

from core.config import cfg

def debug_collect(tensor):
    '''
    use to debug: add a tensor in 'debug' set
    sess.run this 'debug' set outside and check the value
    '''
    tf.add_to_collection('debug',tensor)

def numpy2tensor(data):
    '''
    convert a numpy array(or data) to tensor
    '''
    return tf.convert_to_tensor(data, dtype=tf.float32)


def cabs(tensor, bits):
    '''
    clip the tensor to range of 2^(bits-1)-1 to -2^(bits-1)
    '''
    upper_limit = 2**(bits-1)-1
    lower_limit = -2**(bits-1)
    new_tensor = tf.clip_by_value(tensor, lower_limit, upper_limit)
    return new_tensor


# 疑问？这个最大值最小值决定的max——value，是不是会变的呀？？？
def get_clip(tensor, bits=8, method='round'):
    '''
    get the max N of scaling a small float to a int range
    there are two method: round(may get a better result)
                        floor(safe)
    '''
    limit = numpy2tensor(2**(bits-1))
    max_value = tf.maximum(tf.abs(tf.reduce_min(tensor)), tf.abs(tf.reduce_max(tensor)))
    ratio = tf.div(limit, max_value)
    if method == 'round':
        clip = tf.round(tf.div(tf.log(ratio), tf.log(numpy2tensor(2))))
        return clip
    elif method == 'floor':
        clip = tf.floor(tf.div(tf.log(ratio), numpy2tensor(2)))
        return clip
    else:
        print("error")


def quantize(x, k):
    '''
    skip the gradient of round(x*n)/n
    '''
    G = tf.get_default_graph()
    n = float(2**k)
    with G.gradient_override_map({"Round": "Identity"}):
        return tf.round(x * n) / n


def quantize_plus(x):
    '''
    skip the gradient of round(x)
    '''
    G = tf.get_default_graph()
    with G.gradient_override_map({"Round": "Identity"}):
        return tf.round(x)


def handle_round(tensor):
    '''
    a new round method
    handle_round(x) = ceil(x) when the decimal part of a float point is .5
                  = round(x) others  
    '''
    tensor_2x = tf.multiply(tensor, numpy2tensor(2))
    tensor_2x_floor = tf.floor(tensor_2x)
    new_tensor = tf.where(tf.equal(tensor_2x, tensor_2x_floor), tf.ceil(tensor), tf.round(tensor))
    return new_tensor


def quant_tensor(tensor, bits=8, fix=False, scale=1, return_type='int', check=False):
    '''
    quantify a float tensor to a quant int tensor
    and a quant float tensor of fix bits
    '''
    if not fix:
        clip = get_clip(tensor, bits)
        # if tf.greater(clip, tf.constant(4.0)).eval:
        # if check:
        #             clip = tf.constant(3.0)
        scale = tf.pow(numpy2tensor(2), clip)

    quant_data = handle_round(tf.multiply(tensor, scale))
    quant_int_data = cabs(quant_data, bits)  # 限制在一定范围内
    quant_float_data = tf.div(quant_int_data, scale)

    if not fix and return_type == 'int':
        return quant_int_data, clip
    elif not fix and return_type == 'float':
        return quant_float_data, clip
    elif fix and return_type == 'int':
        return quant_int_data
    elif fix and return_type == 'float':
        return quant_float_data
    else:
        print('=> Error in quant_tensor module !!! ')


def _w_fold(w, gama, var, epsilon):
    """fold the BN into weight"""
    return tf.div(tf.multiply(gama, w), tf.sqrt(var + epsilon))


def _bias_fold(beta, gama, mean, var, epsilon):
    """fold the batch norm layer & build a bias_fold"""
    return tf.subtract(beta, tf.div(tf.multiply(gama, mean), tf.sqrt(var + epsilon)))


def _variable_on_device(name, shape, initializer, trainable=True):
    '''
    create a new variable of tensor
    '''
    dtype = tf.float32
    if not callable(initializer):
        var = tf.get_variable(name, initializer=initializer, trainable=trainable)
    else:
        var = tf.get_variable(
            name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
    return var


def _variable_on_device_reuse(name, shape, initializer, trainable=True):
    '''
    create a new variable of tensor and reuse after
    '''
    dtype = tf.float32
    with tf.variable_scope('v_scope', reuse=tf.AUTO_REUSE) as scope1:
        if not callable(initializer):
            var = tf.get_variable('v_scope'+name, initializer=initializer, trainable=trainable)
        else:
            var = tf.get_variable(
            'v_scope'+name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
    return var


def _variable_with_weight_decay(name, shape, wd, initializer, trainable=True):
    '''
    create a new weight variable of tensor and add L2 norm operation
    '''
    var = _variable_on_device(name, shape, initializer, trainable)
    if wd is not None and trainable:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


class ModelSkeleton(object):
    """
    build basic module of all kinds of Nerual Networks models.
    """
    def __init__(self, trainable, BATCH_NORM_EPSILON):
        self.decay = cfg.DECAY
        self.stddev = cfg.STDDEV
        self.alpha = cfg.ALPHA
        self.WEIGHT_DECAY = cfg.WEIGHT_DECAY
        self.BATCH_NORM_EPSILON = BATCH_NORM_EPSILON
        # pass parameter from other place
        self.testing = False if trainable else True
        # self.testing = False
        self.quant = cfg.QUANT
        self.model_params = []

    def bn_fusion(self, input_data, layer_name, filters_shape, strides, padding, trainable, check=False):
        '''
        fusion batch normlization into convolution layer
        '''
        with tf.variable_scope(layer_name) as scope:
            channels = input_data.get_shape()[3]
            kernel_val = tf.truncated_normal_initializer(stddev=self.stddev, dtype=tf.float32)
            mean_val = tf.constant_initializer(0.0)
            var_val = tf.constant_initializer(1.0)
            gamma_val = tf.constant_initializer(1.0)
            beta_val = tf.constant_initializer(0)
            scale_init = tf.constant_initializer(0.5)

            weight = _variable_with_weight_decay(
             'weight', shape=filters_shape, wd=self.WEIGHT_DECAY, initializer=kernel_val, trainable=trainable)

            conv = tf.nn.conv2d(input_data, weight, strides, padding=padding, name='convolution')
            parameter_bn_shape = conv.get_shape()[-1:]
            gamma = _variable_on_device('gamma', parameter_bn_shape, gamma_val,
                                        trainable=trainable)
            beta = _variable_on_device('beta', parameter_bn_shape, beta_val,
                                       trainable=trainable)
            moving_mean = _variable_on_device('moving_mean', parameter_bn_shape, mean_val, trainable=False)
            moving_variance = _variable_on_device('moving_variance', parameter_bn_shape, var_val, trainable=False)

            # fold weight and bias
            mean, variance = tf.nn.moments(conv, list(range(len(conv.get_shape()) - 1)))
            update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, self.decay,
                                                                       zero_debias=False)
            update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, self.decay,
                                                                           zero_debias=False)
        def mean_var_with_update():
            with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                return tf.identity(mean), tf.identity(variance)
        if self.testing:
            mean_1, var = moving_mean, moving_variance
        else:
            mean_1, var = mean_var_with_update()

        if check:
            tf.add_to_collection("var", var)
            tf.add_to_collection("mean", mean_1)
            tf.add_to_collection("gamma", gamma)

        # 防止动态范围过大，把小方差去除掉
        # logic = tf.where(tf.greater(var,0.001),tf.ones_like(var),tf.zeros_like(var))
        # gamma = tf.multiply(gamma, logic)
        w_fold = _w_fold(weight, gamma, var, self.BATCH_NORM_EPSILON)
        bias_fold = _bias_fold(beta, gamma, mean_1, var, self.BATCH_NORM_EPSILON)

        return w_fold, bias_fold

    def bn_fusion_separable(self, input_data, layer_name, filters_shape, strides, padding, trainable, rate=(1, 1), separable=False):
        '''
        fusion batch normlization into convolution layer
        '''
        with tf.variable_scope(layer_name) as scope:
            channels = input_data.get_shape()[3]
            kernel_val = tf.truncated_normal_initializer(stddev=self.stddev, dtype=tf.float32)
            mean_val = tf.constant_initializer(0.0)
            var_val = tf.constant_initializer(1.0)
            gamma_val = tf.constant_initializer(1.0)
            beta_val = tf.constant_initializer(0)
            scale_init = tf.constant_initializer(0.5)

            weight = _variable_with_weight_decay(
             'weight', shape=filters_shape, wd=self.WEIGHT_DECAY, initializer=kernel_val, trainable=trainable)

            conv = tf.nn.depthwise_conv2d(input_data, weight, strides, padding=padding, name='convolution', rate=rate)
            parameter_bn_shape = conv.get_shape()[-1:]
            gamma = _variable_on_device('gamma', parameter_bn_shape, gamma_val,
                                        trainable=trainable)
            beta = _variable_on_device('beta', parameter_bn_shape, beta_val,
                                       trainable=trainable)
            moving_mean = _variable_on_device('moving_mean', parameter_bn_shape, mean_val, trainable=False)
            moving_variance = _variable_on_device('moving_variance', parameter_bn_shape, var_val, trainable=False)

        mean, variance = tf.nn.moments(conv, list(range(len(conv.get_shape()) - 1)))
        update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, self.decay,
                                                                   zero_debias=False)
        update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, self.decay,
                                                                       zero_debias=False)

        def mean_var_with_update():
            with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                return tf.identity(mean), tf.identity(variance)
        if self.testing:
            mean_1, var = moving_mean, moving_variance
        else:
            mean_1, var = mean_var_with_update()

        weight = tf.transpose(weight, [0, 1, 3, 2])

        # 防止动态范围过大，把小方差去除掉
        logic = tf.where(tf.greater(var, 0.001), tf.ones_like(var), tf.zeros_like(var))
        gamma = tf.multiply(gamma, logic)

        w_fold = _w_fold(weight, gamma, var, self.BATCH_NORM_EPSILON)
        w_fold = tf.transpose(w_fold, [0, 1, 3, 2])
        bias_fold = _bias_fold(beta, gamma, mean_1, var, self.BATCH_NORM_EPSILON)

        return w_fold, bias_fold

    def quant_model_float(self, input_data, weight, bias, strides, padding, activate, act_fun='relu', separable=False, rate=(1, 1), check=False):
        '''
        quantify a float tensor to a quant float tensor of given range with fixed bits
        '''
        if check:
            tf.add_to_collection("input", input_data)
        quant_input, clip_x = quant_tensor(input_data, return_type='float')
        quant_weight, clip_w = quant_tensor(weight, return_type='float', check=check)
        # if check:
        #     quant_weight = tf.where(tf.greater(tf.abs(weight), 16), tf.round(weight), quant_weight)  # clip依然为3，但是将大于16的不直接用16做近似，效果好了很多

        clip = clip_x+clip_w
        fix_ratio = tf.pow(numpy2tensor(2), clip)
        quant_bias = quant_tensor(bias, bits=32, fix=True, scale=fix_ratio, return_type='float')  # 为什么是32位
        if check:
            tf.add_to_collection("check", quant_input)
            tf.add_to_collection("clip_w", clip_w)
            tf.add_to_collection("quant_weight", quant_weight)
            tf.add_to_collection("weight", weight)
            tf.add_to_collection("quant_bias", quant_bias)
            tf.add_to_collection("bias", bias)
            tf.add_to_collection("clip", clip)


        if separable:
            quant_origin = tf.nn.depthwise_conv2d(quant_input, quant_weight, strides, padding=padding, name="depth", rate=rate)
        else:
            quant_origin = tf.nn.conv2d(quant_input, quant_weight, strides, padding=padding, name="conv")
        quant_conv = tf.nn.bias_add(quant_origin, quant_bias)
        if check:
            tf.add_to_collection("output", quant_conv)
        if activate:
            if act_fun == 'leakyrelu':
                quant_conv = tf.nn.leaky_relu(quant_conv, alpha=self.alpha)
            elif act_fun == 'relu':
                quant_conv = tf.nn.relu(quant_conv)

        return quant_conv

    def conv_bn_relu_layer(self, input_data, layer_name, filters_shape, trainable, activate=True, bn_fusion=True, downsample=False, check=False):
        '''
        convolution + batch normlization + leakyrelu(option)
        '''

        strides = [1, 1, 1, 1]
        if downsample:
            strides = [1, 2, 2, 1]
        padding = "SAME"
        if bn_fusion:
            weight, bias = self.bn_fusion(input_data=input_data, filters_shape=filters_shape, layer_name=layer_name,
                                          strides=strides, padding=padding, trainable=trainable, check=check)

            if self.quant or check:
                conv = self.quant_model_float(input_data, weight, bias, strides, padding, activate=True, act_fun='relu', check=check)
                if check:
                    conv1 = tf.nn.conv2d(input_data, weight, strides, padding=padding, name="convlution")
                    conv1 = tf.nn.bias_add(conv1, bias)
                    tf.add_to_collection("convTrue", conv1)
            else:
                conv = tf.nn.conv2d(input_data, weight, strides, padding=padding, name="convlution")
                conv = tf.nn.bias_add(conv, bias)
                if activate:
                    conv = tf.nn.relu(conv)
        else:
            with tf.variable_scope(layer_name) as scope:
                channels = input_data.get_shape()[3]
                kernel_val = tf.truncated_normal_initializer(stddev=self.stddev, dtype=tf.float32)
                mean_val = tf.constant_initializer(0.0)
                var_val = tf.constant_initializer(1.0)
                gamma_val = tf.constant_initializer(1.0)
                beta_val = tf.constant_initializer(0)
                scale_init = tf.constant_initializer(0.5)

                weight = _variable_with_weight_decay(
                    'weight', shape=filters_shape, wd=self.WEIGHT_DECAY, initializer=kernel_val, trainable=trainable)

                conv = tf.nn.conv2d(input_data, weight, strides, padding=padding, name="convlution")
                conv = tf.layers.batch_normalization(conv, training = trainable)
                conv = tf.nn.relu(conv)

        return conv

    def separable_conv2d(self, input_data, layer_name, filters_shape, strides, trainable, rate=(1, 1), bn_fusion=True, check=False):

        padding = "SAME"
        if bn_fusion:
            weight, bias = self.bn_fusion_separable(input_data=input_data, filters_shape=filters_shape,
                                                    layer_name=layer_name, strides=strides, padding=padding,
                                                    trainable=trainable, separable=True, rate=rate)

            if self.quant or check:
                conv = self.quant_model_float(input_data, weight, bias, strides, padding, activate=False, rate=rate,
                                              separable=True, check=check)
                if check:
                    conv1 = tf.nn.depthwise_conv2d(input_data, weight, strides, padding=padding, name="convlution")
                    conv1 = tf.nn.bias_add(conv1, bias)
                    tf.add_to_collection("convTrue", conv1)

            else:
                conv = tf.nn.depthwise_conv2d(input_data, weight, strides, padding=padding, rate=rate, )
                # conv = tf.layers.batch_normalization(conv)
                conv = tf.nn.bias_add(conv, bias)
        else:
            with tf.variable_scope(layer_name) as scope:
                channels = input_data.get_shape()[3]
                kernel_val = tf.truncated_normal_initializer(stddev=self.stddev, dtype=tf.float32)
                mean_val = tf.constant_initializer(0.0)
                var_val = tf.constant_initializer(1.0)
                gamma_val = tf.constant_initializer(1.0)
                beta_val = tf.constant_initializer(0)
                scale_init = tf.constant_initializer(0.5)

                weight = _variable_with_weight_decay(
                    'weight', shape=filters_shape, wd=self.WEIGHT_DECAY, initializer=kernel_val, trainable=trainable)

                conv = tf.nn.depthwise_conv2d(input_data, weight, strides, padding=padding, name='convolution',
                                              rate=rate)
                conv = tf.layers.batch_normalization(conv)

        return conv

    def conv_layer(self, input_data, layer_name, filters_shape, trainable):
        '''
        无bn与relu时使用
        '''

        padding = "SAME"
        strides = [1, 1, 1, 1]

        with tf.variable_scope(layer_name) as scope:
            weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=trainable,
                                   shape=filters_shape, initializer=tf.random_normal_initializer(stddev=self.stddev))
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=trainable,
                                     dtype=tf.float32, initializer=tf.constant_initializer(0))
        if self.quant:
            conv = self.quant_model_float(input_data, weight, bias, strides, padding, activate=False)

        else:
            conv = tf.nn.conv2d(input_data, weight, strides, padding=padding, name='convolution')
            conv = tf.nn.bias_add(conv, bias)

        return conv

    def basic_unit_with_downsampling(self, x, trainable, out_channels=None, stride=2, rate=(1, 1), bn_fusion=True, check=False):
        # 下采样单元
        in_channels = x.get_shape().as_list()[3]
        out_channels = 2 * in_channels if out_channels is None else out_channels

        y = self.conv_bn_relu_layer(x, "right_a", (1, 1, in_channels, in_channels), trainable, bn_fusion=bn_fusion)  # 不写激活函数默认为ReLu
        y = self.separable_conv2d(y, "right_b", (3, 3, in_channels, 1), (1, stride, stride, 1), trainable, rate=rate, bn_fusion=bn_fusion)
        y = self.conv_bn_relu_layer(y, "right_c", (1, 1, in_channels, out_channels // 2), trainable, bn_fusion=bn_fusion)

        x = self.separable_conv2d(x, "left_a", (3, 3, in_channels, 1), (1, stride, stride, 1), trainable, rate=rate, bn_fusion=bn_fusion, check=check)
        x = self.conv_bn_relu_layer(x, "left_b", (1, 1, in_channels, out_channels // 2), trainable, bn_fusion=bn_fusion)
        return x, y

    def basic_unit(self, x, rate, trainable, bn_fusion=True):
        # shufflenet中普通单元
        in_channels = x.shape[3]
        x = self.conv_bn_relu_layer(x, "right_a", [1, 1, in_channels, in_channels], trainable, bn_fusion=bn_fusion)
        x = self.separable_conv2d(x, "right_b", (3, 3, in_channels, 1), (1, 1, 1, 1), trainable, rate=rate, bn_fusion=bn_fusion)
        x = self.conv_bn_relu_layer(x, "right_c", (1, 1, in_channels, in_channels), trainable, bn_fusion=bn_fusion)
        return x

    def concat_shuffle_split(self, x, y):
        shape = x.get_shape().as_list()
        batch_size = shape[0]
        height, width = shape[1], shape[2]
        depth = shape[3]
        # shape [batch_size, height, width, 2, depth]
        z = tf.concat([x, y], axis=3)  # 连接矩阵，将通道维度合并
        # to be compatible with tflite
        # 实现shuffle split，操作有点猛！厉害！
        z = tf.reshape(z, shape=[batch_size, -1, 2, depth])
        z = tf.transpose(z, [0, 1, 3, 2])
        z = tf.reshape(z, [batch_size, height, width, 2 * depth])
        x, y = tf.split(z, num_or_size_splits=2, axis=3)  # 在通道上分为两个子张量，传入basic unit前需要Channel split
        return x, y



