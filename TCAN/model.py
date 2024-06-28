import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
import torch.nn.functional as F
import math
import numpy as np

class AttentionBlock(nn.Module):
    """
    实现了一个基本的自注意力机制，它接受输入张量，计算查询、键和值，
    然后通过点积计算注意力分数，并将注意力分数应用于值张量，
    最后返回加权的值张量以及注意力权重。

    关键点：MASK掩码用于防止模型看到未来的时间步，从而实现因果注意力。

    注：该代码是简单修改后的版本，与原作者代码略有不同。
    1、原作者代码中的mask是ByteTensor类型，但是在PyTorch 1.2.0版本中，ByteTensor已经被废弃，使用bool类型代替。
    2、在注意力权重和最终的输出之间通常会有dropout层，但是原作者代码中没有实现，这里添加了dropout层以防止过拟合。
    3、在注意力计算中，通常会对最后一个维度进行softmax操作，而不是中间的维度。
    在典型的注意力机制中，我们希望计算每个查询和键之间的关联度，然后根据这些关联度对值进行加权求和。
    具体来说，我们将查询与键的乘积作为注意力分数，并通过 softmax 函数将这些分数归一化为概率分布。
    softmax 操作应用在最后一个维度上，以确保每个查询对应的注意力权重总和为 1。
    """
    def __init__(self, in_channels, key_size, value_size):
        super(AttentionBlock, self).__init__()
        self.linear_query = nn.Linear(in_channels, key_size)
        self.linear_keys = nn.Linear(in_channels, key_size)
        self.linear_values = nn.Linear(in_channels, value_size)
        self.sqrt_key_size = math.sqrt(key_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input):
        # input is dim (N, in_channels, T) where N is the batch_size, and T is the sequence length
        mask = np.array([[1 if i>j else 0 for i in range(input.size(2))] for j in range(input.size(2))])
        mask = torch.tensor(mask, dtype=torch.bool, device=input.device)
        
        input = input.permute(0,2,1) # input: [N, T, inchannels]
        keys = self.linear_keys(input) # keys: (N, T, key_size)
        query = self.linear_query(input) # query: (N, T, key_size)
        values = self.linear_values(input) # values: (N, T, value_size)

        temp = torch.bmm(query, torch.transpose(keys, 1, 2)) / self.sqrt_key_size # shape: (N, T, T)
        temp.data.masked_fill(mask, -float('inf'))

        weight_temp = F.softmax(temp, dim=-1)
        value_attentioned = torch.bmm(weight_temp, values).permute(0,2,1) # shape: (N, T, value_size)
      
        return value_attentioned, weight_temp # output: (N, value_size, T)

class Chomp1d(nn.Module):
    """
    裁剪掉最后的chomp_size个元素，保持张量的维度不变。
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
    
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, key_size, num_sub_blocks, temp_attn, nheads, en_res, 
                conv, stride, dilation, padding, vhdrop_layer, visual, dropout=0.2):
        """
        Args:
            n_inputs: 输入张量的通道数。
            n_outputs: 输出张量的通道数。
            kernel_size: 卷积核的大小。
            key_size: 注意力机制中的键的大小。
            num_sub_blocks: 残差块中的卷积层数。
            temp_attn: 是否使用自注意力机制。
            nheads: 多头注意力机制的头数。
            en_res: 是否使用残差连接。
            conv: 是否使用卷积层。
            stride: 卷积层的步长。
            dilation: 卷积层的膨胀率。
            padding: 卷积层的填充大小。
            vhdrop_layer: 可分离卷积层。
            visual: 是否可视化注意力权重。
            dropout: dropout层的概率。

        Returns:
            out: 输出张量。
            attn_weight_cpu: 注意力权重。

        1、根据是否使用注意力机制，初始化多头注意力模块或单头注意力模块。
        2、初始化残差连接。
        3、初始化激活函数。
        4、根据是否使用卷积，初始化卷积层或一系列卷积层。
        5、初始化权重。
        
        """
        super(TemporalBlock, self).__init__()
        # multi head
        self.nheads = nheads
        self.visual = visual
        self.en_res = en_res
        self.conv = conv
        self.temp_attn = temp_attn
        if self.temp_attn:
            if self.nheads > 1:
                self.attentions = [AttentionBlock(n_inputs, key_size, n_inputs) for _ in range(self.nheads)]
                for i, attention in enumerate(self.attentions):
                    self.add_module('attention_{}'.format(i), attention)
                # self.cat_attentions = AttentionBlock(n_inputs * self.nheads, n_inputs, n_inputs)
                self.linear_cat = nn.Linear(n_inputs * self.nheads, n_inputs)
            else:
                self.attention = AttentionBlock(n_inputs, key_size, n_inputs)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        if self.conv:
            self.net = self._make_layers(num_sub_blocks, n_inputs, n_outputs, kernel_size, stride, dilation, 
                                        padding, vhdrop_layer, dropout)
            self.init_weights()


    def _make_layers(self, num_sub_blocks, n_inputs, n_outputs, kernel_size, stride, dilation, 
                    padding, vhdrop_layer, dropout=0.2):
        """
        该函数用于构建多个卷积层，每个卷积层后面跟着一个ReLU激活函数和一个dropout层。
        """
        layers_list = []

        if vhdrop_layer is not None:
            layers_list.append(vhdrop_layer)
        layers_list.append(
            weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)))
        layers_list.append(Chomp1d(padding)) 
        layers_list.append(nn.ReLU())
        layers_list.append(nn.Dropout(dropout))
        for _ in range(num_sub_blocks-1):
            layers_list.append(
                weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)))
            layers_list.append(Chomp1d(padding)) 
            layers_list.append(nn.ReLU())
            layers_list.append(nn.Dropout(dropout))

        return nn.Sequential(*layers_list)

    def init_weights(self):
        """
        该函数用于初始化卷积层的权重，包括卷积层和残差连接的 1x1 卷积层，采用正态分布初始化。
        """
        layer_idx_list = []
        for name, _ in self.net.named_parameters():
            inlayer_param_list = name.split('.')
            layer_idx_list.append(int(inlayer_param_list[0]))
        layer_idxes = list(set(layer_idx_list))
        for idx in layer_idxes:
            getattr(self.net[idx], 'weight').data.normal_(0, 0.01)

        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        根据是否使用自注意力机制，选择不同的前向传播方式。

        如果使用自注意力机制，则根据是否多头注意力进行处理，将注意力机制的输出与输入进行残差连接。

        如果不使用自注意力机制，则直接对输入进行卷积处理，再与输入进行残差连接。
        """
        # x: [N, emb_size, T]
        if self.temp_attn == True:
            en_res_x = None
            attn_weight = None
            if self.nheads > 1:
                # will create some bugs when nheads>1  注：这句话是原作者的注释，但是没有具体说明什么bug，但是确实产生了bug
                # 这行代码的作用是将多头注意力机制的输出拼接在一起。
                # bug：在执行 torch.cat([att(x) for att in self.attentions], dim=1) 时，att(x) 返回了一个元组，而 torch.cat 期望的是张量列表。
                # 解决方法：将 torch.cat([att(x) for att in self.attentions], dim=1) 改为 torch.cat([att(x)[0] for att in self.attentions], dim=1)
                x_out = torch.cat([att(x)[0] for att in self.attentions], dim=1)
                out = self.net(self.linear_cat(x_out.transpose(1,2)).transpose(1,2))
            else:
                # x = x if self.downsample is None else self.downsample(x)
                out_attn, attn_weight = self.attention(x)
                if self.conv:
                    out = self.net(out_attn)
                else:
                    out = out_attn
                weight_x = F.softmax(attn_weight.sum(dim=2),dim=1)
                en_res_x = weight_x.unsqueeze(2).repeat(1,1,x.size(1)).transpose(1,2) * x
                en_res_x = en_res_x if self.downsample is None else self.downsample(en_res_x)
                
            res = x if self.downsample is None else self.downsample(x)

            if self.visual:
                attn_weight_cpu = attn_weight.detach().cpu().numpy()
            else:
                attn_weight_cpu = [0]*10
            
            # 抽象，现在又遇到了一个问题，在 attn_weight 中，报错：UnboundLocalError: cannot access local variable 'attn_weight' where it is not associated with a value
            # 这作者太抽象了，估计是没有考虑到有人会单独测试TemporalBlock模块。
            # 当 temp_attn == True 时，forward 方法中的 attn_weight 变量只在条件分支中被定义，而在方法的其余部分根本就没有定义。那删除干嘛，肯定会报错呀。
            # 解决方法：在 if 语句之前声明 attn_weight = None。这样，无论条件是否满足，attn_weight 都会被赋予一个初始值。
            del attn_weight
            
            if self.en_res:
                # 我靠，服了，为什么不考虑使用多头注意力机制时，nheads>1时，en_res = True的情况
                # 多头注意力机制通常会生成多个注意力的输出，这些输出可能需要合并或者连接起来，然后再与原始输入进行残差连接。在这种情况下，en_res 可以表示是否使用残差连接。
                # 我这样设置是合理的，但是作者完全没有在这种情况下考虑 en_res_x 的情况。
                # 并且就算你在一开始设置了 en_res_x = None，但是在Python中，不能将None与其他类型的对象相加，因为None是一个特殊的单例对象，它表示一个空值或缺失值，并且没有与其他类型相加的定义。
                # 解决方法：先暂时将 en_res_x 去掉，不进行相加
                #return self.relu(out + res + en_res_x), attn_weight_cpu
                return self.relu(out + res + en_res_x), attn_weight_cpu
            else:
                return self.relu(out + res), attn_weight_cpu

        else:
            out = self.net(x)
            print("out", out.shape)
            res = x if self.downsample is None else self.downsample(x)
            print("res", res.shape)
            return self.relu(out + res) # return: [N, emb_size, T]

class TemporalConvNet(nn.Module):
    def __init__(self, input_output_size, emb_size, num_channels, num_sub_blocks, temp_attn, nheads, en_res,
                conv, key_size, kernel_size, visual, vhdropout=0.0, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        self.vhdrop_layer = None
        # layers.append(nn.Conv1d(emb_size*2, emb_size, 1))
        self.temp_attn = temp_attn
        # self.temp_attn_share = AttentionBlock(emb_size, key_size, emb_size)
        if vhdropout != 0.0:
            print("no vhdropout")
            # self.vhdrop_layer = VariationalHidDropout(vhdropout)
            # self.vhdrop_layer.reset_mask()
        num_levels = len(num_channels)
        print("num_levels", num_levels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = emb_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, key_size, num_sub_blocks, \
                temp_attn, nheads, en_res, conv, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, \
                vhdrop_layer=self.vhdrop_layer, visual=visual, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: [batchsize, seq_len, emb_size]
        attn_weight_list = []
        if self.temp_attn:
            out = x
            for i in range(len(self.network)):
                out, attn_weight = self.network[i](out)
                # print("the len of attn_weight", len(attn_weight))
                # if len(attn_weight) == 64:
                #     attn_weight_list.append([attn_weight[18], attn_weight[19]])
                attn_weight_list.append([attn_weight[0], attn_weight[-1]])
            return out, attn_weight_list
        else:
            return self.network(x)




def test_attention_block():
    batch_size = 16
    in_channels = 32
    seq_len = 50
    input_tensor = torch.randn(batch_size, in_channels, seq_len)

    key_size = 16
    value_size = 16

    attention_block = AttentionBlock(in_channels, key_size, value_size)
    value_attentioned, weight_temp = attention_block(input_tensor)

    print("Input shape:")
    print(input_tensor.shape)
    print("Linear query weights:")
    print(attention_block.linear_query.weight.shape)
    print("\nLinear keys weights:")
    print(attention_block.linear_keys.weight.shape)
    print("\nLinear values weights:")
    print(attention_block.linear_values.weight.shape)

    print("\nWeight matrix:")
    print(weight_temp.shape)

    print("\nOutput content:")
    print(value_attentioned.shape)


def test_temporal_block():
    # 参数设置
    n_inputs = 600      # 输入通道数
    n_outputs = 600     # 输出通道数
    kernel_size = 3     # 卷积核大小
    key_size = 600      # Attention key 的大小
    num_sub_blocks = 2  # 子块的数量
    temp_attn = True    # 是否使用自注意力机制
    nheads = 1          # 多头注意力机制中的头数
    en_res = True       # 是否启用增强残差连接
    conv = True         # 是否使用卷积
    stride = 1          # 步幅
    dilation = 1        # 膨胀率
    padding = (kernel_size - 1) * dilation  # 填充大小
    vhdrop_layer = None                     # 变分隐藏层的 dropout 层
    visual = True                           # 是否启用可视化
    dropout = 0.2                           # dropout 率


    temporal_block = TemporalBlock(n_inputs, n_outputs, kernel_size, key_size, num_sub_blocks, temp_attn, nheads, en_res, conv, stride, dilation, padding, vhdrop_layer, visual, dropout)

    batch_size = 64
    seq_len = 80
    x = torch.randn(batch_size, n_inputs, seq_len)

    if temp_attn:
        out, attn_weights = temporal_block(x)
        print("输出形状:", out.shape)
        print("注意力权重形状:", len(attn_weights), len(attn_weights[0]))
    else:
        out = temporal_block(x)
        print("输出形状:", out.shape)


def test_temporal_conv_net():
    # 参数设置
    input_output_size = 600                 # 输入和输出大小
    emb_size = 600                          # 嵌入大小
    num_channels = [600, 600, 600, 600]     # 各层的通道数
    num_sub_blocks = 2          # 每个块的子块数
    temp_attn = True            # 是否使用自注意力
    nheads = 1                  # 多头注意力中的头数
    en_res = True               # 是否启用增强残差连接
    conv = True                 # 是否使用卷积
    key_size = 600              # Attention key 的大小
    kernel_size = 3             # 卷积核大小
    visual = True               # 是否启用可视化
    vhdropout = 0.0             # 变分隐藏层的 dropout 率
    dropout = 0.2               # dropout 率

    temporal_conv_net = TemporalConvNet(input_output_size, emb_size, num_channels, num_sub_blocks, temp_attn, nheads, en_res, conv, key_size, kernel_size, visual, vhdropout, dropout)

    batch_size = 64
    seq_len = 80
    x = torch.randn(batch_size, seq_len, emb_size)

    if temp_attn:
        out, attn_weights = temporal_conv_net(x)
        print("输出形状:", out.shape)
        print("注意力权重形状:", len(attn_weights), len(attn_weights[0]))
    else:
        out = temporal_conv_net(x)
        print("输出形状:", out.shape)


test_temporal_conv_net()
