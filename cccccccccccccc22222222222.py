

import torch
import operator
import os
import numpy as np

torch.backends.cudnn.benchmark = True  # Improves training speed.
torch.backends.cuda.matmul.allow_tf32 = False  # Allow PyTorch to internally use tf32 for matmul
torch.backends.cudnn.allow_tf32 = False  # Allow PyTorch to internally use tf32 for convolutions

import inception_pytorch




model = torch.load('inception-2015-12-05.pt', map_location=torch.device('cpu'))
std1 = model.state_dict()
save_name = 'inception-2015-12-05.pdparams'
model.eval()


model2 = inception_pytorch.InceptionV3()
std2 = model2.state_dict()
model2.eval()


map = {}
already_used = []



for key2, value2 in std2.items():
    if '.bn.num_batches_tracked' in key2:
        continue
    if '.bn.weight' in key2:
        continue
    if '.conv.weight' in key2:
        key1 = key2.replace('.conv.weight', '.weight')
        map[key2] = key1
        std2[key2] = std1[key1]
    if '.bn.bias' in key2:
        key1 = key2.replace('.bn.bias', '.beta')
        map[key2] = key1
        std2[key2] = std1[key1]
    if '.bn.running_mean' in key2:
        key1 = key2.replace('.bn.running_mean', '.mean')
        map[key2] = key1
        std2[key2] = std1[key1]
    if '.bn.running_var' in key2:
        key1 = key2.replace('.bn.running_var', '.var')
        map[key2] = key1
        std2[key2] = std1[key1]
map['output.weight'] = 'output.weight'
map['output.bias'] = 'output.bias'

std2['output.weight'] = std1['output.weight']
std2['output.bias'] = std1['output.bias']

model2.load_state_dict(std2)


# x_shape = [4, 3, 512, 512]
# images = torch.randn(x_shape)

import cv2
images = cv2.imread('../data/data42681/afhq/train/cat/flickr_cat_000005.jpg')
images = torch.Tensor(images)
images = images.unsqueeze(0)
images = images.permute(0, 3, 1, 2)



def transform(img):
    batch_size, channels, height, width, = img.shape
    x = paddle.cast(img, paddle.float32)
    theta = paddle.eye(2, 3, dtype=paddle.float32)
    _3 = theta[0][2]
    _4 = theta[0][0]
    _5 = _4 / width
    _6 = theta[0][0]
    _7 = _3 + (_5 - _6 / 299.0)
    _8 = theta[1][2]
    _9 = theta[1][1]
    _10 = (_9 / height)
    _11 = theta[1][1]
    _12 = _8 + (_10 - (_11 / 299.0))
    _13 = paddle.unsqueeze(paddle.cast(theta, x.dtype), 0)
    theta0 = paddle.tile(_13, [batch_size, 1, 1])
    grid = paddle.nn.functional.affine_grid(theta0, [batch_size, channels, 299, 299], False, )
    x0 = paddle.nn.functional.grid_sample(x, grid, "bilinear", "border", False, )
    x1 = x0 - 128.0
    x2 = x1 / 128.0
    return x2


return_features = False
return_features = True

use_fp16 = False
# use_fp16 = True

no_output_bias = False
# no_output_bias = True

code = model.code
print(code)
code = code.replace('_15 = features', '_15 = theta')
print(code)
# model.code = code

features = model(images, return_features=return_features, use_fp16=use_fp16, no_output_bias=no_output_bias)
features2 = model2(images, return_features=return_features, use_fp16=use_fp16, no_output_bias=no_output_bias)

# features = model.layers(images)
# features2 = model2.layers(images)


ddd = np.sum((features2.cpu().detach().numpy() - features.cpu().detach().numpy()) ** 2)
print('diff=%.6f (%s)' % (ddd, 'features'))


print()



