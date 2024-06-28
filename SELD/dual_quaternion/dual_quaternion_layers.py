import numpy as np
import torch
from numpy.random import RandomState
import torch.nn as nn

from torch.nn import Module
from torch.nn.parameter import Parameter

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'dual_quaternion'))

from dual_quaternion_ops import *

"""
注：该文件代码与Titouan Parcollet的quaternion_layers.py文件相似，原代码是基于单四元数的，本代码在其基础上增加了双四元数的支持
"""

class DepthwiseSeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv2D, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DepthwiseSeparableConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv1D, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DualQuaternionConv(Module):
    """
    实现了一个双四元数卷积层
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilatation=1, padding=0, groups=1, bias=True, init_criterion='glorot',
                 weight_init='quaternion', seed=None, operation='convolution2d', rotation=False, quaternion_format=True, scale=False):

        super(DualQuaternionConv, self).__init__()

        self.in_channels       = in_channels  // 8
        self.out_channels      = out_channels // 8
        self.stride            = stride
        self.padding           = padding
        self.groups            = groups
        self.dilatation        = dilatation
        self.init_criterion    = init_criterion
        self.weight_init       = weight_init
        self.seed              = seed if seed is not None else np.random.randint(0,1234)
        self.rng               = RandomState(self.seed)
        self.operation         = operation
        self.rotation          = rotation
        self.quaternion_format = quaternion_format
        self.winit             =    {'quaternion': quaternion_init,
                                     'unitary'   : unitary_init,
                                     'random'    : random_init}[self.weight_init]
        self.scale             = scale


        (self.kernel_size, self.w_shape) = get_kernel_and_weight_shape( self.operation,
            self.in_channels, self.out_channels, kernel_size )

        # quaternion 1
        self.r_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.i_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.j_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.k_weight  = Parameter(torch.Tensor(*self.w_shape))
        # quaternion 2
        self.r_weight_2  = Parameter(torch.Tensor(*self.w_shape))
        self.i_weight_2  = Parameter(torch.Tensor(*self.w_shape))
        self.j_weight_2  = Parameter(torch.Tensor(*self.w_shape))
        self.k_weight_2  = Parameter(torch.Tensor(*self.w_shape))

        if self.scale:
            self.scale_param  = Parameter(torch.Tensor(self.r_weight.shape))
        else:
            self.scale_param  = None

        if self.rotation:
            self.zero_kernel = Parameter(torch.zeros(self.r_weight.shape), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        affect_init_conv(self.r_weight, self.i_weight, self.j_weight, self.k_weight,
                    self.kernel_size, self.winit, self.rng, self.init_criterion, \
                         self.r_weight_2, self.i_weight_2, self.j_weight_2, self.k_weight_2)
        if self.scale_param is not None:
            torch.nn.init.xavier_uniform_(self.scale_param.data)
        if self.bias is not None:
           self.bias.data.zero_()

    def forward(self, input):

        return dual_quaternion_conv(input, self.r_weight, self.i_weight, self.j_weight, self.k_weight, \
             self.r_weight_2, self.i_weight_2, self.j_weight_2, self.k_weight_2, \
             self.bias, self.stride, self.padding, self.groups, self.dilatation)


    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_channels='      + str(self.in_channels) \
            + ', out_channels='   + str(self.out_channels) \
            + ', bias='           + str(self.bias is not None) \
            + ', kernel_size='    + str(self.kernel_size) \
            + ', stride='         + str(self.stride) \
            + ', padding='        + str(self.padding) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init='    + str(self.weight_init) \
            + ', seed='           + str(self.seed) \
            + ', rotation='       + str(self.rotation) \
            + ', q_format='       + str(self.quaternion_format) \
            + ', operation='      + str(self.operation) + ')'


class DualQuaternionLinear(Module):
    """
    实现了一个双四元数线性变换层
    """

    def __init__(self, in_features, out_features, bias=True,
                 init_criterion='he', weight_init='quaternion',
                 seed=None):

        super(DualQuaternionLinear, self).__init__()
        self.in_features  = in_features // 8
        self.out_features = out_features // 8
        # quaternion 1
        self.r_weight     = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.i_weight     = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.j_weight     = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.k_weight     = Parameter(torch.Tensor(self.in_features, self.out_features))
        # quaternion 2
        self.r_weight_2     = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.i_weight_2     = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.j_weight_2     = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.k_weight_2     = Parameter(torch.Tensor(self.in_features, self.out_features))

        if bias:
            self.bias     = Parameter(torch.Tensor(self.out_features*8))
        else:
            self.register_parameter('bias', None)

        self.init_criterion = init_criterion
        self.weight_init    = weight_init
        self.seed           = seed if seed is not None else np.random.randint(0,1234)
        self.rng            = RandomState(self.seed)
        self.reset_parameters()

    def reset_parameters(self):
        winit = {'quaternion': quaternion_init,
                 'unitary': unitary_init}[self.weight_init]
        if self.bias is not None:
            self.bias.data.fill_(0)
        affect_init(self.r_weight, self.i_weight, self.j_weight, self.k_weight, \
                    self.r_weight_2, self.i_weight_2, self.j_weight_2, self.k_weight_2, \
                    winit, self.rng, self.init_criterion)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        if input.dim() == 3:
            T, N, C = input.size()
            input  = input.reshape(T * N, C)
            output = dual_quaternion_linear(input=input, r_weight=self.r_weight, i_weight=self.i_weight, j_weight=self.j_weight, k_weight=self.k_weight, \
                r_weight_2=self.r_weight_2, i_weight_2=self.i_weight_2, j_weight_2=self.j_weight_2, k_weight_2=self.k_weight_2, bias=self.bias)
            # output = QuaternionLinearFunction.apply(input, self.r_weight, self.i_weight, self.j_weight, self.k_weight, self.bias)
            output = output.reshape(T, N, output.size(1))
        elif input.dim() == 2:
            output = dual_quaternion_linear(input=input, r_weight=self.r_weight, i_weight=self.i_weight, j_weight=self.j_weight, k_weight=self.k_weight, \
                r_weight_2=self.r_weight_2, i_weight_2=self.i_weight_2, j_weight_2=self.j_weight_2, k_weight_2=self.k_weight_2, bias=self.bias)
            # output = QuaternionLinearFunction.apply(input, self.r_weight, self.i_weight, self.j_weight, self.k_weight, self.bias)
        else:
            raise NotImplementedError

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init=' + str(self.weight_init) \
            + ', seed=' + str(self.seed) + ')'


# 进行测试
if __name__ == '__main__':
    # 参数
    batch_size = 4
    in_channels = 8
    height = 32
    width = 32
    
    out_channels = 16
    kernel_size = 3
    stride = 1
    padding = 1
    dilatation = 1
    
    # 卷积层
    input = torch.randn(batch_size, in_channels, height, width)
    conv = DualQuaternionConv(in_channels, out_channels, kernel_size, stride, dilatation, padding)
    print("卷积层")
    print(conv)
    output = conv(input)
    print("输入数据形状:", input.shape)
    print("输出数据形状:", output.shape)

    # 线性层（2D）
    in_features = 16
    out_features = 32
    input = torch.randn(batch_size, in_features)
    linear = DualQuaternionLinear(in_features, out_features)
    print("线性层(2D)")
    print(linear)
    output = linear(input)
    print("输入数据形状:", input.shape)
    print("输出数据形状:", output.shape)

    # 线性层（3D）
    sequence_length = 10
    input_3d = torch.randn(sequence_length, batch_size, in_features)
    linear = DualQuaternionLinear(in_features, out_features)
    print("线性层(3D)")
    print(linear)
    output_3d = linear(input_3d)
    print("3D 输入形状:", input_3d.shape)
    print("3D 输出形状:", output_3d.shape)