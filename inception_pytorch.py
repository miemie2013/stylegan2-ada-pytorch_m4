import paddle.nn.functional
import torch
import torch.nn as nn
import torch.nn.functional as F
import operator
import os
import numpy as np
import torch.Tensor as Tensor




model = torch.load('inception-2015-12-05.pt', map_location=torch.device('cpu'))
std1 = model.state_dict()
save_name = 'inception-2015-12-05.pdparams'




class InceptionE(nn.Module):
    def __init__(self,
        in_features,
    ):
        super().__init__()
        self.activation = in_features

    def forward(self,
                x: Tensor) -> Tensor:
        a = (self.tower_conv).forward(x, )
        b = (self.tower_1_conv_1).forward((self.tower_1_conv).forward(x, ), )
        _0 = torch.contiguous((self.conv).forward(x, ))
        _1 = torch.contiguous((self.tower_mixed_conv).forward(a, ))
        _2 = (self.tower_mixed_conv_1).forward(a, )
        _3 = torch.contiguous(_2)
        _4 = (self.tower_1_mixed_conv).forward(b, )
        _5 = torch.contiguous(_4)
        _6 = (self.tower_1_mixed_conv_1).forward(b, )
        _7 = torch.contiguous(_6)
        _8 = (self.tower_2_conv).forward((self.tower_2_pool).forward(x, ), )
        _9 = [_0, _1, _3, _5, _7, torch.contiguous(_8)]
        return torch.cat(_9, 1)




class InceptionD(nn.Module):
    def __init__(self,
        in_features,
    ):
        super().__init__()
        self.activation = in_features

    def forward(self,
                x: Tensor) -> Tensor:
        _0 = torch.contiguous((self.tower).forward(x, ))
        _1 = torch.contiguous((self.tower_1).forward(x, ))
        _2 = torch.contiguous((self.pool).forward(x, ))
        return torch.cat([_0, _1, _2], 1)




class InceptionC(nn.Module):
    def __init__(self,
        in_features,
    ):
        super().__init__()
        self.activation = in_features

    def forward(self,
                x: Tensor) -> Tensor:
        _0 = torch.contiguous((self.conv).forward(x, ))
        _1 = torch.contiguous((self.tower).forward(x, ))
        _2 = torch.contiguous((self.tower_1).forward(x, ))
        _3 = torch.contiguous((self.tower_2).forward(x, ))
        return torch.cat([_0, _1, _2, _3], 1)




class InceptionB(nn.Module):
    def __init__(self,
        in_features,
    ):
        super().__init__()
        self.activation = in_features

    def forward(self,
                x: Tensor) -> Tensor:
        _0 = torch.contiguous((self.conv).forward(x, ))
        _1 = torch.contiguous((self.tower).forward(x, ))
        _2 = torch.contiguous((self.pool).forward(x, ))
        return torch.cat([_0, _1, _2], 1)




class InceptionA(nn.Module):
    def __init__(self, num_channels, pool_features):
        super().__init__()
        self.conv = Conv2dLayer(num_channels, 64, 1)
        self.tower = nn.Sequential()
        self.tower.add_module('conv', Conv2dLayer(num_channels, 48, 1))
        self.tower.add_module('conv_1', Conv2dLayer(48, 64, 5, padding=2))
        self.tower_1 = nn.Sequential()
        self.tower_1.add_module('conv', Conv2dLayer(num_channels, 64, 1))
        self.tower_1.add_module('conv_1', Conv2dLayer(64, 96, 3, padding=1))
        self.tower_1.add_module('conv_2', Conv2dLayer(96, 96, 3, padding=1))
        self.tower_2 = nn.Sequential()
        self.tower_2.add_module('pool', nn.AvgPool2d(kernel_size=3, stride=1, padding=1))
        self.tower_2.add_module('conv', Conv2dLayer(num_channels, pool_features, 1))

    def forward(self, x):
        _0 = self.conv(x)
        _1 = self.tower(x)
        _2 = self.tower_1(x)
        _3 = self.tower_2(x)
        return torch.cat([_0, _1, _2, _3], 1)








class Conv2dLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False)
        # 打印pytorch的RecursiveScriptModule的code源代码发现eps是0.001，而且bn的weight是None，相当于weight是1
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x




class Layers_Sequential(nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.activation = activation

    def forward(self,
                input: Tensor) -> Tensor:
        _0 = self.conv
        _1 = self.conv_1
        _2 = self.conv_2
        _3 = self.pool0
        _4 = self.conv_3
        _5 = self.conv_4
        _6 = self.pool1
        _7 = self.mixed
        _8 = self.mixed_1
        _9 = self.mixed_2
        _10 = self.mixed_3
        _11 = self.mixed_4
        _12 = self.mixed_5
        _13 = self.mixed_6
        _14 = self.mixed_7
        _15 = self.mixed_8
        _16 = self.mixed_9
        _17 = self.mixed_10
        _18 = self.pool2
        input0 = (_0).forward(input, )
        input1 = (_1).forward(input0, )
        input2 = (_2).forward(input1, )
        input3 = (_3).forward(input2, )
        input4 = (_4).forward(input3, )
        input5 = (_5).forward(input4, )
        input6 = (_6).forward(input5, )
        input7 = (_7).forward(input6, )
        input8 = (_8).forward(input7, )
        input9 = (_9).forward(input8, )
        input10 = (_10).forward(input9, )
        input11 = (_11).forward(input10, )
        input12 = (_12).forward(input11, )
        input13 = (_13).forward(input12, )
        input14 = (_14).forward(input13, )
        input15 = (_15).forward(input14, )
        input16 = (_16).forward(input15, )
        input17 = (_17).forward(input16, )
        return (_18).forward(input17, )



class InceptionV3(nn.Module):
    def __init__(self, class_num=1008):
        super().__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('conv', Conv2dLayer(3, 32, 3, 2, 0))
        self.layers.add_module('conv_1', Conv2dLayer(32, 32, 3, 1, 0))
        self.layers.add_module('conv_2', Conv2dLayer(32, 64, 3, 1, 1))
        self.layers.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        self.layers.add_module('conv_3', Conv2dLayer(64, 80, 1, 1, 0))
        self.layers.add_module('conv_4', Conv2dLayer(80, 192, 3, 1, 0))
        self.layers.add_module('pool1', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        self.layers.add_module('mixed', InceptionA(192, 32))
        self.layers.add_module('mixed_1', InceptionA(256, 64))
        self.layers.add_module('mixed_2', InceptionA(288, 64))
        self.layers.add_module('mixed_3', InceptionB(xxxxxxx))
        self.layers.add_module('pool2', nn.AvgPool2d(kernel_size=8, stride=8, padding=0))
        self.output = nn.Linear(2048, class_num)

    def forward(self,
                img: Tensor,
                return_features: bool = False,
                use_fp16: bool = False,
                no_output_bias: bool = False) -> Tensor:
        batch_size, channels, height, width, = img.shape
        x = img.to(torch.float32)
        theta = torch.eye(2, 3, dtype=torch.float32)
        _3 = torch.select(torch.select(theta, 0, 0), 0, 2)
        _4 = torch.select(torch.select(theta, 0, 0), 0, 0)
        _5 = torch.div(_4, width)
        _6 = torch.select(torch.select(theta, 0, 0), 0, 0)
        _7 = _3 + torch.sub(_5, torch.div(_6, 299))
        _8 = torch.select(torch.select(theta, 0, 1), 0, 2)
        _9 = torch.select(torch.select(theta, 0, 1), 0, 1)
        _10 = torch.div(_9, height)
        _11 = torch.select(torch.select(theta, 0, 1), 0, 1)
        _12 = _8 + torch.sub(_10, torch.div(_11, 299))
        _13 = torch.unsqueeze(theta.to(x.dtype), 0)
        theta0 = _13.repeat([batch_size, 1, 1])
        grid = torch.nn.functional.affine_grid(theta0, [batch_size, channels, 299, 299], False, )
        x0 = torch.nn.functional.grid_sample(x, grid, "bilinear", "border", False, )
        x1 = x0 - 128.0
        x2 = x1 / 128.0

        _14 = torch.reshape(self.layers(x2), [-1, 2048])
        # features = torch.to(_14, 6)
        if return_features:
            # _15 = features
            _15 = _14
        # else:
        #     if no_output_bias:
        #         logits0 = __torch__.torch.nn.functional.linear(features, self.output.weight, None, )
        #         logits = logits0
        #     else:
        #         logits = (self.output).forward(features, )
        #     _16 = __torch__.torch.nn.functional.softmax(logits, 1, 3, None, )
        #     _15 = _16
        return _15



print()



