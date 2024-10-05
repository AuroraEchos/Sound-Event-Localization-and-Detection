##########################################################
# pytorch-qnn v1.0
# Titouan Parcollet
# LIA, Université d'Avignon et des Pays du Vaucluse
# ORKIS, Aix-en-provence
# October 2018
##########################################################

from torch.nn import Module
from torch.nn.parameter import Parameter

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'quaternion'))

from quaternion_ops import *


class QuaternionTransposeConv(Module):
    """
    接收一个四元数表示的张量作为输入，应用四元数反卷积操作后，输出一个新的四元数张量
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilatation=1, padding=0, output_padding=0, groups=1, bias=True, init_criterion='glorot',
                 weight_init='quaternion', seed=None, operation='convolution2d', rotation=False,
                 quaternion_format=False):

        super(QuaternionTransposeConv, self).__init__()

        self.in_channels = in_channels // 4
        self.out_channels = out_channels // 4
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.dilatation = dilatation
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else np.random.randint(0, 1234)
        self.rng = RandomState(self.seed)
        self.operation = operation
        self.rotation = rotation
        self.quaternion_format = quaternion_format
        self.winit = {'quaternion': quaternion_init,
                      'unitary': unitary_init,
                      'random': random_init}[self.weight_init]

        (self.kernel_size, self.w_shape) = get_kernel_and_weight_shape(self.operation,
                                                                       self.out_channels, self.in_channels, kernel_size)

        self.r_weight = Parameter(torch.Tensor(*self.w_shape))
        self.i_weight = Parameter(torch.Tensor(*self.w_shape))
        self.j_weight = Parameter(torch.Tensor(*self.w_shape))
        self.k_weight = Parameter(torch.Tensor(*self.w_shape))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        affect_init_conv(self.r_weight, self.i_weight, self.j_weight, self.k_weight,
                         self.kernel_size, self.winit, self.rng, self.init_criterion)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        """
        输入：形状为 (batch_size, in_channels, height, width) 的张量。
        输出：形状为 (batch_size, out_channels, new_height, new_width) 的张量。
        """

        if self.rotation:
            return quaternion_transpose_conv_rotation(input, self.r_weight, self.i_weight,
                                                      self.j_weight, self.k_weight, self.bias, self.stride,
                                                      self.padding,
                                                      self.output_padding, self.groups, self.dilatation,
                                                      self.quaternion_format)
        else:
            return quaternion_transpose_conv(input, self.r_weight, self.i_weight, self.j_weight,
                                             self.k_weight, self.bias, self.stride, self.padding, self.output_padding,
                                             self.groups, self.dilatation)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_channels=' + str(self.in_channels) \
               + ', out_channels=' + str(self.out_channels) \
               + ', bias=' + str(self.bias is not None) \
               + ', kernel_size=' + str(self.kernel_size) \
               + ', stride=' + str(self.stride) \
               + ', padding=' + str(self.padding) \
               + ', dilatation=' + str(self.dilatation) \
               + ', init_criterion=' + str(self.init_criterion) \
               + ', weight_init=' + str(self.weight_init) \
               + ', seed=' + str(self.seed) \
               + ', operation=' + str(self.operation) + ')'

class QuaternionConv(Module):
    """
    接收一个四元数表示的张量作为输入，应用四元数卷积操作后，输出一个新的四元数张量
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilatation=1, padding=0, groups=1, bias=True, init_criterion='glorot',
                 weight_init='quaternion', seed=None, operation='convolution2d', rotation=False,
                 quaternion_format=False):

        super(QuaternionConv, self).__init__()

        self.in_channels = in_channels // 4
        self.out_channels = out_channels // 4
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilatation = dilatation
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else np.random.randint(0, 1234)
        self.rng = RandomState(self.seed)
        self.operation = operation
        self.rotation = rotation
        self.quaternion_format = quaternion_format
        self.winit = {'quaternion': quaternion_init,
                      'unitary': unitary_init,
                      'random': random_init}[self.weight_init]

        (self.kernel_size, self.w_shape) = get_kernel_and_weight_shape(self.operation,
                                                                       self.in_channels, self.out_channels, kernel_size)
        
        self.r_weight = Parameter(torch.Tensor(*self.w_shape))
        self.i_weight = Parameter(torch.Tensor(*self.w_shape))
        self.j_weight = Parameter(torch.Tensor(*self.w_shape))
        self.k_weight = Parameter(torch.Tensor(*self.w_shape))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        affect_init_conv(self.r_weight, self.i_weight, self.j_weight, self.k_weight,
                         self.kernel_size, self.winit, self.rng, self.init_criterion)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):

        if self.rotation:
            return quaternion_conv_rotation(input, self.r_weight, self.i_weight, self.j_weight,
                                            self.k_weight, self.bias, self.stride, self.padding, self.groups,
                                            self.dilatation,
                                            self.quaternion_format)
        else:
            return quaternion_conv(input, self.r_weight, self.i_weight, self.j_weight,
                                   self.k_weight, self.bias, self.stride, self.padding, self.groups, self.dilatation)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_channels=' + str(self.in_channels) \
               + ', out_channels=' + str(self.out_channels) \
               + ', bias=' + str(self.bias is not None) \
               + ', kernel_size=' + str(self.kernel_size) \
               + ', stride=' + str(self.stride) \
               + ', padding=' + str(self.padding) \
               + ', dilatation=' + str(self.dilatation) \
               + ', init_criterion=' + str(self.init_criterion) \
               + ', weight_init=' + str(self.weight_init) \
               + ', seed=' + str(self.seed) \
               + ', operation=' + str(self.operation) + ')'

class QuaternionLinearAutograd(Module):
    """
    实现四元数线性变换, 利用自定义的 Autograd 函数来减少显存消耗，但计算时间会较慢。
    """

    def __init__(self, in_features, out_features, bias=True,
                 init_criterion='glorot', weight_init='quaternion',
                 seed=None, rotation=False, quaternion_format=False):

        super(QuaternionLinearAutograd, self).__init__()
        self.in_features = in_features // 4
        self.out_features = out_features // 4
        self.rotation = rotation
        self.quaternion_format = quaternion_format
        self.r_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.i_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.j_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.k_weight = Parameter(torch.Tensor(self.in_features, self.out_features))

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features * 4))
        else:
            self.register_parameter('bias', None)
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else np.random.randint(0, 1234)
        self.rng = RandomState(self.seed)
        self.reset_parameters()

    def reset_parameters(self):
        winit = {'quaternion': quaternion_init, 'unitary': unitary_init, 'random': random_init}[self.weight_init]
        if self.bias is not None:
            self.bias.data.fill_(0)
        affect_init(self.r_weight, self.i_weight, self.j_weight, self.k_weight, winit,
                    self.rng, self.init_criterion)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        if self.rotation:
            return quaternion_linear_rotation(input, self.r_weight, self.i_weight, self.j_weight, self.k_weight,
                                              self.bias, self.quaternion_format)
        else:
            return quaternion_linear(input, self.r_weight, self.i_weight, self.j_weight, self.k_weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', bias=' + str(self.bias is not None) \
               + ', init_criterion=' + str(self.init_criterion) \
               + ', weight_init=' + str(self.weight_init) \
               + ', seed=' + str(self.seed) + ')'

class QuaternionLinear(Module):
    """
    实现四元数线性变换
    """

    def __init__(self, in_features, out_features, bias=True,
                 init_criterion='glorot', weight_init='quaternion',
                 seed=None):

        super(QuaternionLinear, self).__init__()
        self.in_features = in_features // 4
        self.out_features = out_features // 4
        self.r_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.i_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.j_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.k_weight = Parameter(torch.Tensor(self.in_features, self.out_features))

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features * 4))
        else:
            self.register_parameter('bias', None)

        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else np.random.randint(0, 1234)
        self.rng = RandomState(self.seed)
        self.reset_parameters()

    def reset_parameters(self):
        winit = {'quaternion': quaternion_init,
                 'unitary': unitary_init}[self.weight_init]
        if self.bias is not None:
            self.bias.data.fill_(0)
        affect_init(self.r_weight, self.i_weight, self.j_weight, self.k_weight, winit,
                    self.rng, self.init_criterion)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        if input.dim() == 3:
            T, N, C = input.size()
            input = input.contiguous().view(T * N, C)#view instead of reshape      old ###input = input.view(T * N, C)#view instead of reshape
            output = QuaternionLinearFunction.apply(input, self.r_weight, self.i_weight, self.j_weight, self.k_weight,
                                                    self.bias)
            output = output.view(T, N, output.size(1))
        elif input.dim() == 2:
            output = QuaternionLinearFunction.apply(input, self.r_weight, self.i_weight, self.j_weight, self.k_weight,
                                                    self.bias)
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

"""
QuaternionLinearAutograd 和 QuaternionLinear 的区别：

    1、Autograd 使用：
        QuaternionLinearAutograd 使用了自定义的 Autograd 函数 quaternion_linear 和 quaternion_linear_rotation，旨在减少显存消耗，但计算时间较慢。
        QuaternionLinear 直接使用 QuaternionLinearFunction.apply 来进行四元数线性变换。

    2、Rotation 支持：
        QuaternionLinearAutograd 支持 rotation，可以通过设置 rotation 参数来选择是否应用旋转。
        QuaternionLinear 没有直接支持旋转。

    3、输入维度处理：
        QuaternionLinear 处理 2D 和 3D 输入数据，对于 3D 数据，会先展平为 2D 再进行处理，然后再将输出重塑为 3D。
        QuaternionLinearAutograd 没有明确处理不同维度的输入，但假设输入为 2D 数据。

    选择哪个类取决于具体需求：
        如果需要显存优化和旋转操作，可以使用 QuaternionLinearAutograd。
        如果需要更快的计算且输入数据有可能是 3D 的，可以使用 QuaternionLinear。
"""


if __name__ == "__main__":
    # 参数
    batch_size = 1
    in_channels = 4
    height = 4
    width = 4
    
    out_channels = 8
    kernel_size = 3
    stride = 1
    padding = 1
    dilatation = 1

    # 反卷积层
    input = torch.randn(batch_size, in_channels, height, width)
    conv = QuaternionTransposeConv(in_channels, out_channels, kernel_size, stride)
    print("反卷积层")
    print(conv)
    output = conv(input)
    print("输入数据形状:", input.shape)
    print("输出数据形状:", output.shape)

    # 卷积层
    input = torch.randn(batch_size, in_channels, height, width)
    conv = QuaternionConv(in_channels, out_channels, kernel_size, stride)
    print("卷积层")
    print(conv)
    output = conv(input)
    print("输入数据形状:", input.shape)
    print("输出数据形状:", output.shape)

    # 线性层（自定义 Autograd）
    in_features = 4
    out_features = 8
    input = torch.randn(batch_size, in_features)
    linear = QuaternionLinear(in_features, out_features)
    print("线性层")
    print(linear)
    output = linear(input)
    print("输入数据形状:", input.shape)
    print("输出数据形状:", output.shape)

    # 线性层
    in_features = 4
    out_features = 8
    input = torch.randn(batch_size, in_features)
    linear = QuaternionLinearAutograd(in_features, out_features)
    print("线性层")
    print(linear)
    output = linear(input)
    print("输入数据形状:", input.shape)
    print("输出数据形状:", output.shape)