

import torch
import operator
# import paddle
import os
import numpy as np
# from inception_paddle import InceptionV3




model = torch.load('inception-2015-12-05.pt', map_location=torch.device('cpu'))
std1 = model.state_dict()
save_name = 'inception-2015-12-05.pdparams'


# code = model.code
# code = model.layers.code
# code = model.output.code

sub_layer = model
original_name = sub_layer.original_name
code = sub_layer.code
print(original_name)
print(code)
print('================================')
print('================================')



sub_layer = model.layers
original_name = sub_layer.original_name
code = sub_layer.code
print(original_name)
print(code)
print('================================')
print('================================')



sub_layer = model.output
original_name = sub_layer.original_name
code = sub_layer.code
print(original_name)
print(code)
print('================================')
print('================================')



sub_layer = model.layers.conv  # .conv_1、.conv_2、.conv_3、.conv_4
original_name = sub_layer.original_name
code = sub_layer.code
print(original_name)
print(code)
print('================================')
print('================================')



sub_layer = model.layers.mixed  # .mixed_1、.mixed_2
original_name = sub_layer.original_name
code = sub_layer.code
print(original_name)
print(code)
print('================================')
print('================================')


sub_layer = model.layers.mixed_3  #
original_name = sub_layer.original_name
code = sub_layer.code
print(original_name)
print(code)
print('================================')
print('================================')


sub_layer = model.layers.mixed_4  # .mixed_5、.mixed_6、.mixed_7
original_name = sub_layer.original_name
code = sub_layer.code
print(original_name)
print(code)
print('================================')
print('================================')


sub_layer = model.layers.mixed_8  #
original_name = sub_layer.original_name
code = sub_layer.code
print(original_name)
print(code)
print('================================')
print('================================')


sub_layer = model.layers.mixed_9  # .mixed_10
original_name = sub_layer.original_name
code = sub_layer.code
print(original_name)
print(code)
print('================================')
print('================================')


sub_layer = model.layers.pool0  # .pool1
original_name = sub_layer.original_name
code = sub_layer.code
print(original_name)
print(code)
print('================================')
print('================================')


sub_layer = model.layers.pool2  #
original_name = sub_layer.original_name
code = sub_layer.code
print(original_name)
print(code)
print('================================')
print('================================')


print('================ InceptionXXX ================')
print('================ InceptionXXX ================')
print('================ InceptionXXX ================')






print()



