import torch
import torch.nn as nn
from torchinfo import summary

import numpy as np
import utility_functions as uf
from dual_quaternion.dual_quaternion_layers import * 
from quaternion.quaternion_layers import *

torch.backends.cudnn.enabled = False

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_size % num_heads == 0, "Embedding size must be divisible by number of heads"
        
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        self.values = nn.Conv1d(embed_size, embed_size, kernel_size=1, bias=False)
        self.keys = nn.Conv1d(embed_size, embed_size, kernel_size=1, bias=False)
        self.queries = nn.Conv1d(embed_size, embed_size, kernel_size=1, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)
        
    def forward(self, v, k, q, mask=None):
        N = q.shape[0]
        value_len, key_len, query_len = v.shape[1], k.shape[1], q.shape[1]
        
        # Transpose to (batch_size, embed_size, seq_length) for Conv1d
        v = v.permute(0, 2, 1)
        k = k.permute(0, 2, 1)
        q = q.permute(0, 2, 1)
        
        # Apply convolution
        v = self.values(v).view(N, self.head_dim * self.num_heads, value_len).permute(0, 2, 1).view(N, value_len, self.num_heads, self.head_dim)
        k = self.keys(k).view(N, self.head_dim * self.num_heads, key_len).permute(0, 2, 1).view(N, key_len, self.num_heads, self.head_dim)
        q = self.queries(q).view(N, self.head_dim * self.num_heads, query_len).permute(0, 2, 1).view(N, query_len, self.num_heads, self.head_dim)
        
        # Scaled dot-product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [q, k])
        
        if mask is not None:
            # Apply mask to the energy tensor
            energy = energy.masked_fill(mask == 0, float("-1e9"))
        
        attention = torch.softmax(energy / (self.head_dim ** (1 / 2)), dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, v]).reshape(N, query_len, -1)
        out = self.fc_out(out)
        
        return out

class ResBlock(nn.Module):
    def __init__(self, in_channels, 
        domain='DQ',
        G=128,U=128, kernel_size_dilated_conv=3, dilation=1, stride=1,
        spatial_dropout_rate=0.5, use_bias_conv=True,batch_norm='BN',verbose=False):

        super(ResBlock, self).__init__()
        self.verbose = verbose
        self.batch_norm=batch_norm
        self.spatial_dropout_rate=spatial_dropout_rate
        self.domain = domain
        padding=int(((kernel_size_dilated_conv-1) * dilation)/2)
        L=in_channels
        if self.domain=='Q':
            self.conv1_filter = QuaternionConv(L, G, kernel_size=kernel_size_dilated_conv,
                                    stride=stride, padding=padding,
                                    dilatation=dilation, bias=use_bias_conv, operation='convolution1d')
            self.conv1_gate = QuaternionConv(L, G, kernel_size=kernel_size_dilated_conv,
                                    stride=stride, padding=padding,
                                    dilatation=dilation, bias=use_bias_conv, operation='convolution1d')
        elif self.domain=='DQ':
            self.conv1_filter = DualQuaternionConv(L,G, kernel_size=kernel_size_dilated_conv,
                                    stride=stride, padding=padding,
                                    dilatation=dilation, bias=use_bias_conv, operation='convolution1d')
            self.conv1_gate = DualQuaternionConv(L, G, kernel_size=kernel_size_dilated_conv,
                                    stride=stride, padding=padding,
                                    dilatation=dilation, bias=use_bias_conv, operation='convolution1d')
        else:
            self.conv1_filter = nn.Conv1d(L,G, kernel_size=kernel_size_dilated_conv,
                                    stride=stride, padding=padding,
                                    dilation=dilation,bias=use_bias_conv)
            self.conv1_gate = nn.Conv1d(L,G, kernel_size=kernel_size_dilated_conv,
                                    stride=stride, padding=padding,
                                    dilation=dilation,bias=use_bias_conv)

        if batch_norm in {'BN', 'BN_on_TCN', 'BNonTCN'}:
            self.batch_filter1 = nn.BatchNorm1d(L)
            self.batch_gate1 = nn.BatchNorm1d(L)
            self.batch_filter2 = nn.BatchNorm1d(G)
            self.batch_gate2 = nn.BatchNorm1d(G)
                
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        if(not spatial_dropout_rate==0):
            self.dropout = nn.Dropout1d(p=spatial_dropout_rate)

        if self.domain=='Q':
            self.conv2_skip = QuaternionConv(G,U, kernel_size=1, stride=1, bias=use_bias_conv, operation='convolution1d')
            self.conv2_residual= QuaternionConv(G,L, kernel_size=1, stride=1, bias=use_bias_conv, operation='convolution1d')
        elif self.domain=='DQ':
            self.conv2_skip = DualQuaternionConv(G,U, kernel_size=1, stride=1, bias=use_bias_conv, operation='convolution1d')
            self.conv2_residual= DualQuaternionConv(G,L, kernel_size=1, stride=1, bias=use_bias_conv, operation='convolution1d')
        else:
            self.conv2_skip = nn.Conv1d(G,U, kernel_size=1, stride=1,bias=use_bias_conv)
            self.conv2_residual= nn.Conv1d(G,L, kernel_size=1, stride=1,bias=use_bias_conv)
            
    def forward(self, x):
        """
        修改现有的残差块为预激活残差块
        预处理残差块通过将批归一化和激活函数提前应用于卷积之前，改善了梯度传播，提升了训练稳定性，提供了有效的正则化效果，并在实际任务中带来了性能提升。
        """
        if self.batch_norm in {'BN', 'BN_on_TCN', 'BNonTCN'}:
            x = self.batch_filter1(x)
            x = self.tanh(x)
        
        y_f=self.conv1_filter(x)
        y_g=self.conv1_gate(x)

        if self.batch_norm in {'BN', 'BN_on_TCN', 'BNonTCN'}:
            y_f = self.batch_filter2(y_f)
            y_g = self.batch_gate2(y_g)
        
        y=self.tanh(y_f)*self.sigmoid(y_g)

        if(not self.spatial_dropout_rate==0):
            y=self.dropout(y)
            
        y_skip=self.conv2_skip(y)
        y_residual=self.conv2_residual(y)
        return x+y_residual,y_skip

class TC_Block(nn.Module):
    def __init__(self, in_channels, domain='DQ', 
                G=128,U=128, V=[128,128], V_kernel_size=3,pool_size=[[8,2],[8,2],[2,2]], D=[10], 
                spatial_dropout_rate=0.5, use_bias_conv=True,dilation_mode='fibonacci', pool_time='TCN',batch_norm='BN',
                kernel_size_dilated_conv=3,verbose=False, attention_type=None, key_size=None, value_size=None):
        super(TC_Block, self).__init__()
        self.verbose = verbose
        self.ResBlocks = nn.ModuleList()
        self.D=D
        self.pool_time=pool_time
        self.domain = domain

        for n_resblock in D:
            dilation=1
            prec_1=1
            prec_2=0
            if type(n_resblock)==list:
                for d in (n_resblock):
                    dilation=d
                    
                    self.ResBlocks.append(ResBlock(in_channels=in_channels, 
                                                    domain=domain,
                                                    G=G,U=U,kernel_size_dilated_conv=kernel_size_dilated_conv, 
                                                    dilation=dilation, spatial_dropout_rate=spatial_dropout_rate, 
                                                    use_bias_conv=use_bias_conv,batch_norm=batch_norm,verbose=verbose))
            else:
                for d in range(n_resblock):
                    if dilation_mode=='fibonacci':
                        if d==0:
                            dilation=1
                        else:
                            dilation=prec_1+prec_2
                            prec_2=prec_1
                            prec_1=dilation
                    else:
                        dilation=2**d
                    self.ResBlocks.append(ResBlock(in_channels=in_channels, 
                                                    domain=domain,
                                                    G=G,U=U,kernel_size_dilated_conv=kernel_size_dilated_conv, 
                                                    dilation=dilation, spatial_dropout_rate=spatial_dropout_rate, 
                                                    use_bias_conv=use_bias_conv,batch_norm=batch_norm,verbose=verbose))
        self.relu1=nn.ReLU()

        if self.pool_time=='TCN':
            self.maxpool1=nn.MaxPool1d(pool_size[0][1])

        if self.domain=='Q':
            self.conv1 = QuaternionConv(in_channels, V[0], kernel_size=V_kernel_size, stride=1,padding=1, bias=use_bias_conv, operation='convolution1d')
        elif self.domain=='DQ':
            self.conv1 = DualQuaternionConv(in_channels, V[0], kernel_size=V_kernel_size, stride=1,padding=1, bias=use_bias_conv, operation='convolution1d')
        else:
            self.conv1 = nn.Conv1d(in_channels,V[0], kernel_size=V_kernel_size, stride=1,padding=1,bias=use_bias_conv)
        
        self.attention = MultiHeadAttention(embed_size=V[0], num_heads=8)
        
        self.relu2=nn.ReLU()

        if self.pool_time=='TCN':
            self.maxpool2=nn.MaxPool1d(pool_size[1][1])
        if self.domain=='Q':
            self.conv2 = QuaternionConv(V[0], V[1], kernel_size=V_kernel_size, stride=1,padding=1, bias=use_bias_conv, operation='convolution1d')
        elif self.domain=='DQ':
            self.conv2 = DualQuaternionConv(V[0],V[1], kernel_size=V_kernel_size, stride=1,padding=1, bias=use_bias_conv, operation='convolution1d')
        else:
            self.conv2 = nn.Conv1d(V[0],V[1], kernel_size=V_kernel_size, stride=1,padding=1,bias=use_bias_conv)

        self.tanh=nn.Tanh()
        if self.pool_time=='TCN':
            self.maxpool3=nn.MaxPool1d(pool_size[2][1])
            
    def forward(self, residual):
        skip_connections=[]
        for i in range(len(self.ResBlocks)):
            residual,skip=self.ResBlocks[i](residual)
            skip_connections.append(skip)

        sum_skip=skip_connections[0]
        for i in range(1,len(skip_connections)):
            sum_skip+=skip_connections[i]

        out= self.relu1(sum_skip)

        if self.pool_time=='TCN':
            out=self.maxpool1(out)
        out= self.conv1(out)

        out = out.permute(0,2,1)
        out = self.attention(out, out, out, mask=None)
        out = out.permute(0,2,1)

        out= self.relu2(out)
        if self.pool_time=='TCN':
            out=self.maxpool2(out)
        out= self.conv2(out)

        out= self.tanh(out)
        if self.pool_time=='TCN':
            out=self.maxpool3(out)
        return out

class ConvTC_Block(nn.Module):
    def __init__(self, time_dim, freq_dim=256, input_channels=4, 
                 domain='DQ',
                 cnn_filters=[64,64,64], kernel_size_cnn_blocks=3, pool_size=[[8,2],[8,2],[2,2]], pool_time='TCN',
                 D=[10], dilation_mode='fibonacci',G=128, U=128, kernel_size_dilated_conv=3,spatial_dropout_rate=0.5,
                 V=[128,128], V_kernel_size=3,
                 dropout_perc=0.3, use_bias_conv=True,batch_norm='noBN',
                 attention_type=None,key_size=None,value_size=None,verbose=False):
        super(ConvTC_Block, self).__init__()
        self.time_dim = time_dim
        self.freq_dim = freq_dim
        self.domain = domain
        self.verbose = verbose
        self.D=D
        self.kernel_size_dilated_conv=kernel_size_dilated_conv
        self.dilation_mode=dilation_mode
        self.attenyion_type=attention_type

        if pool_time=='CNN':
            self.time_pooled_size = int(time_dim / np.prod(np.array(pool_size), axis=0)[-1])
        else:
            self.time_pooled_size = time_dim
        #building CNN feature extractor
        conv_layers = []
        layers_list=[]
        in_chans = input_channels
        
        for i, (p,c) in enumerate(zip(pool_size, np.array(cnn_filters))):
            curr_chans = c

            if pool_time=='CNN':
                pool = [p[0],p[1]]
            else:
                pool = [p[0],1]

            if self.domain=='Q':
                layers_list.append(QuaternionConv(in_chans, out_channels=curr_chans, kernel_size=kernel_size_cnn_blocks,
                                        stride=1, padding=1, operation='convolution2d', bias=use_bias_conv))
            elif self.domain=='DQ':
                layers_list.append(DualQuaternionConv(in_chans, out_channels=curr_chans, kernel_size=kernel_size_cnn_blocks,
                                        stride=1, padding=1, operation='convolution2d', bias=use_bias_conv))
            else:
                layers_list.append(nn.Conv2d(in_chans, out_channels=curr_chans, kernel_size=kernel_size_cnn_blocks,stride=1, padding=1,bias=use_bias_conv))
            
            if batch_norm=='BN'or batch_norm=='BN_on_CNN'or batch_norm=='BNonCNN':
                layers_list.append(nn.BatchNorm2d(c))
            layers_list.append(nn.ReLU())
            layers_list.append(nn.MaxPool2d(pool))
            layers_list.append(nn.Dropout(dropout_perc))
            conv_layers.append(nn.Sequential(*layers_list))
            layers_list=[]
            in_chans = curr_chans

        self.cnn = nn.Sequential(*conv_layers)
        L = int(freq_dim / np.prod(np.array(pool_size), axis=0)[0]*cnn_filters[-1])#input dimension for QTCN Block

        self.tcn=TC_Block(in_channels=L, domain=domain,
                G=G,U=U,V=V,V_kernel_size=V_kernel_size, pool_size=pool_size, D=D, 
                spatial_dropout_rate=spatial_dropout_rate, use_bias_conv=use_bias_conv,
                dilation_mode=dilation_mode, pool_time=pool_time,batch_norm=batch_norm,
                kernel_size_dilated_conv=kernel_size_dilated_conv,verbose=verbose,attention_type=attention_type,key_size=key_size,value_size=value_size)

        
    def forward(self, x):
        x = self.cnn(x)
        if self.verbose:
            print ('cnn out ', x.shape)    

        x = x.permute(0,3,1,2) 
        if self.verbose:
            print ('permuted: ', x.shape)   

        x = x.reshape(x.shape[0], self.time_pooled_size, -1)
        if self.verbose:
                print ('reshaped: ', x.shape)   

        x = x.permute(0,2,1)
        if self.verbose:
            print ('permute2: ', x.shape)   
        
        x= self.tcn(x)
        
        if self.verbose:
            print ('tcn out:  ', x.shape)    
        x = x.permute(0,2,1) 

        if self.verbose:
            print ('permute3: ', x.shape)   
        return x

class SELD_Model(nn.Module):
    def __init__(self, time_dim, freq_dim=256, input_channels=4, output_classes=14,
                 domain='DQ',domain_classifier='same', 
                 cnn_filters=[64,64,64], kernel_size_cnn_blocks=3, pool_size=[[8,2],[8,2],[2,2]], pool_time='TCN',
                 D=[10], dilation_mode='fibonacci',G=128, U=128, kernel_size_dilated_conv=3,spatial_dropout_rate=0.5,V=[128,128], V_kernel_size=3,
                 fc_layers=[128], fc_activations='Linear', fc_dropout='all', dropout_perc=0.3, 
                 class_overlaps=3.,
                 use_bias_conv=False,use_bias_linear=True,batch_norm='BN',parallel_ConvTC_block='False',parallel_magphase=False,
                 extra_name='', attention_type=None, key_size=None, value_size=None, verbose=False):
        super(SELD_Model, self).__init__()
        self.input_channels=input_channels
        self.time_dim = time_dim
        self.freq_dim = freq_dim
        self.domain = domain
        self.verbose = verbose
        self.D=D
        self.kernel_size_dilated_conv=kernel_size_dilated_conv
        self.dilation_mode=dilation_mode
        self.parallel_magphase=parallel_magphase
        self.domain_classifier=domain if domain_classifier=='same' else domain_classifier
        self.receptive_field, self.total_n_resblocks=self.calculate_receptive_field()
        self.parallel_ConvTC_block=parallel_ConvTC_block

        if domain in{'q','Q','quaternion','Quaternion'}:
            self.model_name='Q'
        elif domain in{'dq','dQ','DQ','dual_quaternion','Dual_Quaternion'}:
            self.model_name='DualQ'
        else:
            self.model_name=''
        self.model_name+='SELD'
        self.model_name+='-TCN'
        if dilation_mode=='fibonacci':
            self.model_name+='-PHI'
        self.model_name+='-'
        if len(D)>1:
            if D[0]<D[1]:
                self.model_name+='I'
        self.model_name+='S'+str(len(D))
        #self.model_name+='_D_'+str(D)
        if parallel_ConvTC_block not in {'False','false','None','none'}:
            self.model_name+='_'+parallel_ConvTC_block
        self.model_name+='_'+batch_norm
        #if pool_time=='TCN':
        #    self.model_name+='_ptTCN'
        if pool_time=='CNN':
            self.model_name+='_pooltCNN'
        self.model_name+='_RF{}_{}RB'.format(self.receptive_field,self.total_n_resblocks)
        
        self.model_name+=extra_name

        """ if domain in {'q', 'Q', 'quaternion', 'Quaternion'}:
            self.model_name = '四元数'
        elif domain in {'dq', 'dQ', 'DQ', 'dual_quaternion', 'Dual_Quaternion'}:
            self.model_name = '双四元数'
        else:
            self.model_name = ''
        self.model_name += 'SELD'
        self.model_name += '_使用的结构为时域卷积网络(TCN)'
        if dilation_mode == 'fibonacci':
            self.model_name += '_扩张模式为斐波那契'
        self.model_name += '_'
        if len(D) > 1:
            if D[0] < D[1]:
                self.model_name += 'I'
        self.model_name += '模型中堆栈的数量' + str(len(D))
        self.model_name += '_扩张卷积中堆栈层数' + str(D)
        if parallel_ConvTC_block not in {'False', 'false', 'None', 'none'}:
            self.model_name += '_是否使用并行卷积时间池化块' + parallel_ConvTC_block
        self.model_name += '_批归一化的类型为' + batch_norm
        if pool_time == 'TCN':
            self.model_name += '_池化时间维度TCN'
        if pool_time == 'CNN':
            self.model_name += '_池化时间维度CNN'
        self.model_name += '_感受野为{}_残差块个数为{}'.format(self.receptive_field, self.total_n_resblocks)

        self.model_name += extra_name """
        #self.print_model_name(self.model_name)

        sed_output_size = int(output_classes * class_overlaps)    #here 3 is the max number of simultaneus sounds from the same class
        doa_output_size = sed_output_size * 3   #here 3 is the number of spatial dimensions xyz

        if parallel_ConvTC_block in {'2Parallel','2BParallel','2ParallelBranches','2PB'}:
            self.branch_A=ConvTC_Block(time_dim=time_dim, freq_dim=freq_dim, input_channels=input_channels//2, 
                    domain=domain,
                    cnn_filters=cnn_filters, kernel_size_cnn_blocks=kernel_size_cnn_blocks, pool_size=pool_size, pool_time=pool_time,
                    D=D, dilation_mode=dilation_mode,G=G, U=U, kernel_size_dilated_conv=kernel_size_dilated_conv,spatial_dropout_rate=spatial_dropout_rate,
                    V=V, V_kernel_size=V_kernel_size,
                    dropout_perc=dropout_perc,use_bias_conv=use_bias_conv,batch_norm=batch_norm,verbose=False)
            self.branch_B=ConvTC_Block(time_dim=time_dim, freq_dim=freq_dim, input_channels=input_channels//2, 
                    domain=domain,
                    cnn_filters=cnn_filters, kernel_size_cnn_blocks=kernel_size_cnn_blocks, pool_size=pool_size, pool_time=pool_time,
                    D=D, dilation_mode=dilation_mode,G=G, U=U, kernel_size_dilated_conv=kernel_size_dilated_conv,spatial_dropout_rate=spatial_dropout_rate,
                    V=V, V_kernel_size=V_kernel_size,
                    dropout_perc=dropout_perc,use_bias_conv=use_bias_conv,batch_norm=batch_norm,verbose=False)
            fc_input_size=V[-1]*2
        else:
            self.seld_block=ConvTC_Block(time_dim=time_dim, freq_dim=freq_dim, input_channels=input_channels, 
                    domain=domain, 
                    cnn_filters=cnn_filters, kernel_size_cnn_blocks=kernel_size_cnn_blocks, pool_size=pool_size, pool_time=pool_time,
                    D=D, dilation_mode=dilation_mode,G=G, U=U, kernel_size_dilated_conv=kernel_size_dilated_conv,spatial_dropout_rate=spatial_dropout_rate,
                    V=V, V_kernel_size=V_kernel_size,
                    dropout_perc=dropout_perc,use_bias_conv=use_bias_conv,batch_norm=batch_norm,attention_type=attention_type,key_size=key_size,value_size=value_size,verbose=False)
            fc_input_size=V[-1]
        fc_sed_list = []
        fc_doa_list = []

        for fc_layer in fc_layers:
            
            if self.domain_classifier=='Q':
                fc_sed_list.append(QuaternionLinear(fc_input_size, fc_layer, bias=use_bias_linear))
                fc_doa_list.append(QuaternionLinear(fc_input_size, fc_layer, bias=use_bias_linear))
            elif self.domain_classifier=='DQ':
                fc_sed_list.append(DualQuaternionLinear(fc_input_size, fc_layer, bias=use_bias_linear))
                fc_doa_list.append(DualQuaternionLinear(fc_input_size, fc_layer, bias=use_bias_linear))
            else:
                fc_sed_list.append(nn.Linear(fc_input_size, fc_layer,bias=use_bias_linear))
                fc_doa_list.append(nn.Linear(fc_input_size, fc_layer,bias=use_bias_linear))
            
            if fc_activations in  {'relu','ReLU','RELU'}:
                fc_sed_list.append(nn.ReLU())
                fc_doa_list.append(nn.ReLU())
            if fc_dropout in {'all','ALL','True'}:
                fc_sed_list.append(nn.Dropout(dropout_perc))
                fc_doa_list.append(nn.Dropout(dropout_perc))
            fc_input_size=fc_layer
        if fc_dropout in {'last','Last','LAST'}:
                fc_sed_list.append(nn.Dropout(dropout_perc))
                fc_doa_list.append(nn.Dropout(dropout_perc))
        
        self.sed =  nn.Sequential(*fc_sed_list,
                    nn.Linear(fc_layers[-1], sed_output_size, bias=use_bias_linear),
                    nn.Sigmoid())

        self.doa =  nn.Sequential(*fc_doa_list,
                    nn.Linear(fc_layers[-1], doa_output_size, bias=use_bias_linear),
                    nn.Tanh())

    def forward(self, x):
        if self.parallel_ConvTC_block in {'2Parallel','2BParallel','2ParallelBranches','2PB'}:
            if self.parallel_magphase:
                x_A=torch.cat((x[:,:4,:,:],x[:,8:12,:,:]),1)####X_A MicA mag-phase
                x_B=torch.cat((x[:,4:8,:,:],x[:,12:,:,:]),1)####X_B MicB mag-phase
            else:
                x_A=x[:,:self.input_channels//2,:,:]
                x_B=x[:,self.input_channels//2:,:,:]
            branch_A=self.branch_A(x_A)
            branch_B=self.branch_B(x_B)
            x=torch.cat((branch_A,branch_B), 2)
        else:
            x = self.seld_block(x)
        sed = self.sed(x)
        doa = self.doa(x)
        if self.verbose:
            print ('sed prediction:  ', sed.shape)  #target dim: [batch, time, sed_output_size]
            print ('doa prediction: ', doa.shape)  #target dim: [batch, time, doa_output_size]

        return sed, doa

    def calculate_receptive_field(self,verbose=0):
        receptive_field=1
        tcn_block=self.D
        kernel_size=self.kernel_size_dilated_conv
        total_n_resblocks=0
        for i, n_resblock in enumerate(tcn_block):
            dilation=1
            prec_1=1
            prec_2=0
            if type(n_resblock)==list:
                res_count=0
                for d in n_resblock:
                    res_count+=1
                    total_n_resblocks+=1
                    dilation=d
                    receptive_field+=(kernel_size-1)*(dilation)
                    if verbose==2:
                        print('stack ',i+1,' resblock ',res_count,': ',receptive_field)
            else:
                for d in range(n_resblock):
                    total_n_resblocks+=1
                    if self.dilation_mode=='fibonacci':
                        if d==0:
                            dilation=1
                        else:
                            dilation=prec_1+prec_2
                            prec_2=prec_1
                            prec_1=dilation
                    else:
                        dilation=2**d
                    receptive_field+=(kernel_size-1)*(dilation)
                    if verbose==2:
                        print('stack ',i+1,' resblock ',d+1,': ',receptive_field)
        if verbose==1 or verbose==2:
            print(tcn_block,'  Receptive field:',receptive_field,', Total number of Resblocks:',total_n_resblocks)
        return receptive_field, total_n_resblocks
        
    def print_model_name(self, model_name):
        print(f"模型参数信息如下:")
        name_parts = model_name.split('_')
        for part in name_parts:
            if part.startswith('四元数') or part.startswith('双四元数'):
                print(f"--域: {part}")
            elif part.startswith('SELD'):
                print(f"--模型类型: {part}")
            elif part.startswith('使用的结构为'):
                print(f"--使用的结构: {part[6:]}")
            elif part.startswith('扩张模式为'):
                print(f"--扩张模式: {part[5:]}")
            elif part.startswith('模型中堆栈的数量'):
                print(f"--堆栈数量: {part[8:]}")
            elif part.startswith('扩张卷积中堆栈层数'):
                print(f"--扩张卷积层数: {part[9:]}")
            elif part.startswith('是否使用并行卷积时间池化块'):
                print(f"--并行卷积时间池化块: {part[13:]}")
            elif part.startswith('批归一化的类型为'):
                print(f"--批归一化类型: {part[8:]}")
            elif part.startswith('池化时间维度'):
                print(f"--池化时间维度: {part[6:]}")
            elif part.startswith('感受野为'):
                print(f"--感受野信息: {part[4:]}")
            elif part.startswith('残差块个数为'):
                print(f"--残差块数量: {part[6:]}")
            else:
                print(f"额外信息: {part}")

if __name__ == '__main__':
    # 定义参数
    """
    in_chans    : 输入数据的通道数
    sample      : 一个包含 8 个通道的音频样本，每个通道有 60 秒的数据
    nperseg     : 计算频谱的窗口参数
    noverlap    : 计算频谱的窗口参数
    sp          : 计算频谱,将频谱数据转换为 PyTorch 的张量--(1, in_chans, freq_dim, time_dim)
    """
    in_chans = 8                            
    sample = np.ones((in_chans,32000*60))
    nperseg = 512
    noverlap = 112
    
    sp = uf.spectrum_fast(sample, nperseg=nperseg, noverlap=noverlap, output_phase=False)
    sp = torch.tensor(sp.reshape(1,sp.shape[0],sp.shape[1],sp.shape[2])).float()

    # 创建模型
    """
    模型的设计考虑了输入数据的维度（频谱维度），以及模型内部处理和池化操作的维度
    为每个100毫秒的标签帧（label frame）创建1个预测（SED和DOA）
    模型的设计目标是针对音频数据中的每个100毫秒的时间窗口进行事件检测（SED）和方向角（DOA）的预测
    SED 通常表示声音事件的发生与否，而 DOA 表示声音事件的方向
    """
    model = SELD_Model(
        time_dim=sp.shape[-1],  
        freq_dim=256, 
        input_channels=8, 
        output_classes=14,
        domain='DQ', 
        domain_classifier='same',
        cnn_filters=[64, 64, 64], 
        kernel_size_cnn_blocks=3, 
        pool_size=[[8, 2], [8, 2], [2, 2]], 
        pool_time='TCN',
        D=[10], 
        dilation_mode='fibonacci', 
        G=128, 
        U=128, 
        kernel_size_dilated_conv=3,
        spatial_dropout_rate=0.5, 
        V=[128, 128], 
        V_kernel_size=3,
        fc_layers=[128], 
        fc_activations='Linear', 
        fc_dropout='all', 
        dropout_perc=0.3, 
        class_overlaps=3.,
        use_bias_conv=False,
        use_bias_linear=True,
        batch_norm='BN',
        parallel_ConvTC_block='False',
        parallel_magphase=False,
        extra_name='',
        attention_type='self_attention',
        key_size=16,
        value_size=32,
        verbose=True
    )

    print ('\n输入形状: ', sp.shape)
    sed, doa = model(sp)

    #target shape sed=[batch,600(label frames),42] doa=[batch, 600(label frames),126](这个注释是原作者的，经验证没有问题)
    print('\n输出形状: ','SED shape: ', sed.shape, '| DOA shape: ', doa.shape)

    print("\n模型详细结构如下: ")
    summary(model, input_size=(1,in_chans,256,4800))