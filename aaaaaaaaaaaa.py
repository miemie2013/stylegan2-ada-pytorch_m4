
import torch
import paddle
import numpy as np



def pre(img):
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
    return x2



def pre2(img):
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





x_shape = [4, 3, 512, 512]
images = torch.randn(x_shape)
images2 = images.cpu().detach().numpy()
images2 = paddle.to_tensor(images2)
images2 = paddle.cast(images2, dtype=paddle.float32)
features = pre(images)

features2 = pre2(images2)


ddd = np.sum((features2.numpy() - features.cpu().detach().numpy()) ** 2)
print('diff=%.6f (%s)' % (ddd, 'features'))
print()





