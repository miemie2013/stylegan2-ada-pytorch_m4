

import torch
import operator
import paddle
import os
import numpy as np
from inception_paddle import InceptionV3




model = torch.load('inception-2015-12-05.pt', map_location=torch.device('cpu'))
std1 = model.state_dict()
save_name = 'inception-2015-12-05.pdparams'


model2 = InceptionV3(class_num=1008)
std2 = model2.state_dict()


map = {}
already_used = []


def match(conv_w_name, bn_param_name):
    ss1 = conv_w_name.split('.')
    ss2 = bn_param_name.split('.')
    if len(ss1) != len(ss2):
        return False
    else:
        len_ = len(ss1)
        count = 0
        for i in range(len_):
            if ss1[i] == ss2[i]:
                count += 1
        if count >= len_ - 1:
            return True
        else:
            return False



for key2, value2 in std2.items():
    shape2 = value2.shape
    if 'inception_stem.' in key2:
        if key2 not in map.keys():
            map[key2] = []
        for key1, value1 in std1.items():
            value1 = value1.cpu().detach().numpy()
            shape1 = list(value1.shape)
            if operator.eq(shape1, shape2):
                if len(shape1) == 4:
                    if '.weight' in key1 and '.conv.weight' in key2 and key1 not in already_used:
                        map[key2].append(key1)
                        already_used.append(key1)

                        key2_bn_name = key2.replace('.conv.', '.bn.')
                        key2_bn_b_name = key2_bn_name.replace('.weight', '.bias')
                        if key2_bn_b_name in std2.keys():
                            key1_bn_w_name = key1.replace('.weight', '.beta')
                            if key1_bn_w_name in std1.keys():
                                map[key2_bn_b_name] = key1_bn_w_name
                                already_used.append(key1_bn_w_name)
                        key2_bn_m_name = key2_bn_name.replace('.weight', '._mean')
                        if key2_bn_m_name in std2.keys():
                            key1_bn_m_name = key1.replace('.weight', '.mean')
                            if key1_bn_m_name in std1.keys():
                                map[key2_bn_m_name] = key1_bn_m_name
                                already_used.append(key1_bn_m_name)
                        key2_bn_v_name = key2_bn_name.replace('.weight', '._variance')
                        if key2_bn_v_name in std2.keys():
                            key1_bn_v_name = key1.replace('.weight', '.var')
                            if key1_bn_v_name in std1.keys():
                                map[key2_bn_v_name] = key1_bn_v_name
                                already_used.append(key1_bn_v_name)
                if len(shape1) == 1:
                    pass
    else:
        if key2 not in map.keys():
            map[key2] = []
        for key1, value1 in std1.items():
            value1 = value1.cpu().detach().numpy()
            shape1 = list(value1.shape)
            if operator.eq(shape1, shape2):
                if len(shape1) == 4:
                    if '.weight' in key1 and '.conv.weight' in key2 and key1 not in already_used and len(map[key2]) == 0:
                        map[key2].append(key1)
                        already_used.append(key1)

                        key2_bn_name = key2.replace('.conv.', '.bn.')
                        key2_bn_b_name = key2_bn_name.replace('.weight', '.bias')
                        if key2_bn_b_name in std2.keys():
                            key1_bn_w_name = key1.replace('.weight', '.beta')
                            if key1_bn_w_name in std1.keys():
                                map[key2_bn_b_name] = key1_bn_w_name
                                already_used.append(key1_bn_w_name)
                        key2_bn_m_name = key2_bn_name.replace('.weight', '._mean')
                        if key2_bn_m_name in std2.keys():
                            key1_bn_m_name = key1.replace('.weight', '.mean')
                            if key1_bn_m_name in std1.keys():
                                map[key2_bn_m_name] = key1_bn_m_name
                                already_used.append(key1_bn_m_name)
                        key2_bn_v_name = key2_bn_name.replace('.weight', '._variance')
                        if key2_bn_v_name in std2.keys():
                            key1_bn_v_name = key1.replace('.weight', '.var')
                            if key1_bn_v_name in std1.keys():
                                map[key2_bn_v_name] = key1_bn_v_name
                                already_used.append(key1_bn_v_name)
                if len(shape1) == 1:
                    pass


map['fc.weight'] = 'output.weight'
map['fc.bias'] = 'output.bias'


use_gpu = True
gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
place = paddle.CUDAPlace(gpu_id) if use_gpu else paddle.CPUPlace()

def copy(name, w, std):
    value2 = paddle.to_tensor(w, place=place)
    value = std[name]
    value = value * 0 + value2
    std[name] = value




# fullyConnectedLayer_dic = {}
# for key, value in state_dict.items():
#     fullyConnectedLayer_dic[key] = value.data.numpy()

for key2 in map.keys():
    key1 = map[key2]
    bn_weight = False
    if isinstance(key1, list):
        assert len(key1) <= 1
        if len(key1) == 0:
            bn_weight = True  # bn.weight = 0
        else:
            key1 = key1[0]

    if bn_weight:
        v2 = std2[key2]
        w = np.ones(v2.shape).astype(np.float32)  # bn.weight = 1
    else:
        w = std1[key1].data.numpy()

    if 'output.weight' in key1:
        w = w.transpose(1, 0)  # pytorch的nn.Linear()的weight权重要转置才能赋值给paddle的nn.Linear()
    copy(key2, w, std2)
model2.set_state_dict(std2)

paddle.save(std2, save_name)




kkk = 0
for key2, value2 in std2.items():
    if 'bn._mean' in key2:
        kkk += 1


x_shape = [4, 3, 512, 512]
images = torch.randn(x_shape)
images2 = images.cpu().detach().numpy()
images2 = paddle.to_tensor(images2)
images2 = paddle.cast(images2, dtype=paddle.float32)
features = model(images, return_features=True)


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

# features2 = model2(images2, return_features=True)

features2 = model2(transform(images2), return_features=True)
ddd = np.sum((features2.numpy() - features.cpu().detach().numpy()) ** 2)
print('diff=%.6f (%s)' % (ddd, 'features'))

print()



