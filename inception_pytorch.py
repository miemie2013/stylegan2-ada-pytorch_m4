import torch
import torch.nn as nn
import torch.nn.functional as F
import operator
import os
import numpy as np



# model = torch.load('inception-2015-12-05.pt', map_location=torch.device('cpu'))
# std1 = model.state_dict()
# save_name = 'inception-2015-12-05.pdparams'



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





class InceptionE(nn.Module):
    def __init__(self, num_channels, pool='avg'):
        super().__init__()
        assert pool in ['avg', 'max']
        self.tower_conv = Conv2dLayer(num_channels, 384, 1)
        self.tower_1_conv = Conv2dLayer(num_channels, 448, 1)
        self.tower_1_conv_1 = Conv2dLayer(448, 384, 3, padding=1)
        self.conv = Conv2dLayer(num_channels, 320, 1)
        self.tower_mixed_conv = Conv2dLayer(384, 384, (1, 3), padding=(0, 1))
        self.tower_mixed_conv_1 = Conv2dLayer(384, 384, (3, 1), padding=(1, 0))
        self.tower_1_mixed_conv = Conv2dLayer(384, 384, (1, 3), padding=(0, 1))
        self.tower_1_mixed_conv_1 = Conv2dLayer(384, 384, (3, 1), padding=(1, 0))
        if pool == 'avg':
            self.tower_2_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1, count_include_pad=False)
        elif pool == 'max':
            self.tower_2_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.tower_2_conv = Conv2dLayer(num_channels, 192, 1)

    def forward(self, x):
        a = self.tower_conv(x)
        b = self.tower_1_conv_1(self.tower_1_conv(x))
        _0 = self.conv(x)
        _1 = self.tower_mixed_conv(a)
        _2 = self.tower_mixed_conv_1(a)

        _3 = _2
        _4 = self.tower_1_mixed_conv(b)
        _5 = _4
        _6 = self.tower_1_mixed_conv_1(b)
        _7 = _6
        _8 = self.tower_2_conv(self.tower_2_pool(x))

        _9 = [_0, _1, _3, _5, _7, _8]
        return torch.cat(_9, 1)




class InceptionD(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.tower = nn.Sequential()
        self.tower.add_module('conv', Conv2dLayer(num_channels, 192, 1))
        self.tower.add_module('conv_1', Conv2dLayer(192, 320, 3, stride=2))
        self.tower_1 = nn.Sequential()
        self.tower_1.add_module('conv', Conv2dLayer(num_channels, 192, 1))
        self.tower_1.add_module('conv_1', Conv2dLayer(192, 192, (1, 7), padding=(0, 3)))
        self.tower_1.add_module('conv_2', Conv2dLayer(192, 192, (7, 1), padding=(3, 0)))
        self.tower_1.add_module('conv_3', Conv2dLayer(192, 192, 3, stride=2))
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        _0 = self.tower(x)
        _1 = self.tower_1(x)
        _2 = self.pool(x)
        return torch.cat([_0, _1, _2], 1)




class InceptionC(nn.Module):
    def __init__(self, num_channels, channels_7x7):
        super().__init__()
        self.conv = Conv2dLayer(num_channels, 192, 1)
        self.tower = nn.Sequential()
        self.tower.add_module('conv', Conv2dLayer(num_channels, channels_7x7, 1, stride=1))
        self.tower.add_module('conv_1', Conv2dLayer(channels_7x7, channels_7x7, (1, 7), stride=1, padding=(0, 3)))
        self.tower.add_module('conv_2', Conv2dLayer(channels_7x7, 192, (7, 1), stride=1, padding=(3, 0)))
        self.tower_1 = nn.Sequential()
        self.tower_1.add_module('conv', Conv2dLayer(num_channels, channels_7x7, 1))
        self.tower_1.add_module('conv_1', Conv2dLayer(channels_7x7, channels_7x7, (7, 1), padding=(3, 0)))
        self.tower_1.add_module('conv_2', Conv2dLayer(channels_7x7, channels_7x7, (1, 7), padding=(0, 3)))
        self.tower_1.add_module('conv_3', Conv2dLayer(channels_7x7, channels_7x7, (7, 1), padding=(3, 0)))
        self.tower_1.add_module('conv_4', Conv2dLayer(channels_7x7, 192, (1, 7), padding=(0, 3)))
        self.tower_2 = nn.Sequential()
        self.tower_2.add_module('pool', nn.AvgPool2d(kernel_size=3, stride=1, padding=1, count_include_pad=False))
        self.tower_2.add_module('conv', Conv2dLayer(num_channels, 192, 1))

    def forward(self, x):
        _0 = self.conv(x)
        _1 = self.tower(x)
        _2 = self.tower_1(x)
        _3 = self.tower_2(x)
        return torch.cat([_0, _1, _2, _3], 1)


class InceptionB(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv = Conv2dLayer(num_channels, 384, 3, stride=2)
        self.tower = nn.Sequential()
        self.tower.add_module('conv', Conv2dLayer(num_channels, 64, 1))
        self.tower.add_module('conv_1', Conv2dLayer(64, 96, 3, padding=1))
        self.tower.add_module('conv_2', Conv2dLayer(96, 96, 3, stride=2))
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        _0 = self.conv(x)
        _1 = self.tower(x)
        _2 = self.pool(x)
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
        self.tower_2.add_module('pool', nn.AvgPool2d(kernel_size=3, stride=1, padding=1, count_include_pad=False))
        self.tower_2.add_module('conv', Conv2dLayer(num_channels, pool_features, 1))

    def forward(self, x):
        _0 = self.conv(x)
        _1 = self.tower(x)
        _2 = self.tower_1(x)
        _3 = self.tower_2(x)
        return torch.cat([_0, _1, _2, _3], 1)




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
        self.layers.add_module('mixed_3', InceptionB(288))
        self.layers.add_module('mixed_4', InceptionC(768, 128))
        self.layers.add_module('mixed_5', InceptionC(768, 160))
        self.layers.add_module('mixed_6', InceptionC(768, 160))
        self.layers.add_module('mixed_7', InceptionC(768, 192))
        self.layers.add_module('mixed_8', InceptionD(768))
        self.layers.add_module('mixed_9', InceptionE(1280, pool='avg'))
        self.layers.add_module('mixed_10', InceptionE(2048, pool='max'))
        self.layers.add_module('pool2', nn.AvgPool2d(kernel_size=8, stride=8, padding=0))
        self.output = nn.Linear(2048, class_num)

    def forward(self, img, return_features=False, use_fp16=False, no_output_bias=False):
        batch_size, channels, height, width, = img.shape
        if use_fp16:
            type_ = torch.float16
        else:
            type_ = torch.float32
        x = img.to(type_)
        theta = torch.eye(2, 3, dtype=None, device=x.device)
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
        grid = torch.nn.functional.affine_grid(theta0, [batch_size, channels, 299, 299], False)
        x0 = torch.nn.functional.grid_sample(x, grid, "bilinear", "border", False)
        x1 = x0 - 127.5
        x2 = x1 / 127.5

        _14 = torch.reshape(self.layers(x2), [-1, 2048])
        features = _14.to(torch.float32)
        if return_features:
            return _14
        else:
            if no_output_bias:
                logits0 = torch.nn.functional.linear(features, self.output.weight, None, )
                logits = logits0
            else:
                logits = self.output(features)
            _16 = torch.nn.functional.softmax(logits, 1, 3, None, )
            _15 = _16
        return _15



print()



