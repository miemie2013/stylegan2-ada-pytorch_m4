from __future__ import absolute_import, division, print_function
import math
import paddle
from paddle import ParamAttr
import paddle.nn as nn
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
from paddle.nn.initializer import Uniform

from typing import Tuple, List, Dict, Union, Callable, Any

from paddle import nn


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs


class TheseusLayer(nn.Layer):
    def __init__(self, *args, **kwargs):
        super(TheseusLayer, self).__init__()
        self.res_dict = {}
        self.res_name = self.full_name()
        self.pruner = None
        self.quanter = None

    def _return_dict_hook(self, layer, input, output):
        res_dict = {"output": output}
        # 'list' is needed to avoid error raised by popping self.res_dict
        for res_key in list(self.res_dict):
            # clear the res_dict because the forward process may change according to input
            res_dict[res_key] = self.res_dict.pop(res_key)
        return res_dict

    def init_res(self,
                 stages_pattern,
                 return_patterns=None,
                 return_stages=None):
        if return_patterns and return_stages:
            msg = f"The 'return_patterns' would be ignored when 'return_stages' is set."
            print(msg)
            return_stages = None

        if return_stages is True:
            return_patterns = stages_pattern
        # return_stages is int or bool
        if type(return_stages) is int:
            return_stages = [return_stages]
        if isinstance(return_stages, list):
            if max(return_stages) > len(stages_pattern) or min(
                    return_stages) < 0:
                msg = f"The 'return_stages' set error. Illegal value(s) have been ignored. The stages' pattern list is {stages_pattern}."
                print(msg)
                return_stages = [
                    val for val in return_stages
                    if val >= 0 and val < len(stages_pattern)
                ]
            return_patterns = [stages_pattern[i] for i in return_stages]

        if return_patterns:
            self.update_res(return_patterns)

    def replace_sub(self, *args, **kwargs) -> None:
        msg = "The function 'replace_sub()' is deprecated, please use 'upgrade_sublayer()' instead."
        print(DeprecationWarning(msg))
        raise DeprecationWarning(msg)

    def upgrade_sublayer(self,
                         layer_name_pattern: Union[str, List[str]],
                         handle_func: Callable[[nn.Layer, str], nn.Layer]
                         ) -> Dict[str, nn.Layer]:
        """use 'handle_func' to modify the sub-layer(s) specified by 'layer_name_pattern'.
        Args:
            layer_name_pattern (Union[str, List[str]]): The name of layer to be modified by 'handle_func'.
            handle_func (Callable[[nn.Layer, str], nn.Layer]): The function to modify target layer specified by 'layer_name_pattern'. The formal params are the layer(nn.Layer) and pattern(str) that is (a member of) layer_name_pattern (when layer_name_pattern is List type). And the return is the layer processed.
        Returns:
            Dict[str, nn.Layer]: The key is the pattern and corresponding value is the result returned by 'handle_func()'.
        Examples:
            from paddle import nn
            import paddleclas
            def rep_func(layer: nn.Layer, pattern: str):
                new_layer = nn.Conv2D(
                    in_channels=layer._in_channels,
                    out_channels=layer._out_channels,
                    kernel_size=5,
                    padding=2
                )
                return new_layer
            net = paddleclas.MobileNetV1()
            res = net.replace_sub(layer_name_pattern=["blocks[11].depthwise_conv.conv", "blocks[12].depthwise_conv.conv"], handle_func=rep_func)
            print(res)
            # {'blocks[11].depthwise_conv.conv': the corresponding new_layer, 'blocks[12].depthwise_conv.conv': the corresponding new_layer}
        """

        if not isinstance(layer_name_pattern, list):
            layer_name_pattern = [layer_name_pattern]

        hit_layer_pattern_list = []
        for pattern in layer_name_pattern:
            # parse pattern to find target layer and its parent
            layer_list = parse_pattern_str(pattern=pattern, parent_layer=self)
            if not layer_list:
                continue
            sub_layer_parent = layer_list[-2]["layer"] if len(
                layer_list) > 1 else self

            sub_layer = layer_list[-1]["layer"]
            sub_layer_name = layer_list[-1]["name"]
            sub_layer_index = layer_list[-1]["index"]

            new_sub_layer = handle_func(sub_layer, pattern)

            if sub_layer_index:
                getattr(sub_layer_parent,
                        sub_layer_name)[sub_layer_index] = new_sub_layer
            else:
                setattr(sub_layer_parent, sub_layer_name, new_sub_layer)

            hit_layer_pattern_list.append(pattern)
        return hit_layer_pattern_list

    def stop_after(self, stop_layer_name: str) -> bool:
        """stop forward and backward after 'stop_layer_name'.
        Args:
            stop_layer_name (str): The name of layer that stop forward and backward after this layer.
        Returns:
            bool: 'True' if successful, 'False' otherwise.
        """

        layer_list = parse_pattern_str(stop_layer_name, self)
        if not layer_list:
            return False

        parent_layer = self
        for layer_dict in layer_list:
            name, index = layer_dict["name"], layer_dict["index"]
            if not set_identity(parent_layer, name, index):
                msg = f"Failed to set the layers that after stop_layer_name('{stop_layer_name}') to IdentityLayer. The error layer's name is '{name}'."
                print(msg)
                return False
            parent_layer = layer_dict["layer"]

        return True

    def update_res(
            self,
            return_patterns: Union[str, List[str]]) -> Dict[str, nn.Layer]:
        """update the result(s) to be returned.
        Args:
            return_patterns (Union[str, List[str]]): The name of layer to return output.
        Returns:
            Dict[str, nn.Layer]: The pattern(str) and corresponding layer(nn.Layer) that have been set successfully.
        """

        # clear res_dict that could have been set
        self.res_dict = {}

        class Handler(object):
            def __init__(self, res_dict):
                # res_dict is a reference
                self.res_dict = res_dict

            def __call__(self, layer, pattern):
                layer.res_dict = self.res_dict
                layer.res_name = pattern
                if hasattr(layer, "hook_remove_helper"):
                    layer.hook_remove_helper.remove()
                layer.hook_remove_helper = layer.register_forward_post_hook(
                    save_sub_res_hook)
                return layer

        handle_func = Handler(self.res_dict)

        hit_layer_pattern_list = self.upgrade_sublayer(
            return_patterns, handle_func=handle_func)

        if hasattr(self, "hook_remove_helper"):
            self.hook_remove_helper.remove()
        self.hook_remove_helper = self.register_forward_post_hook(
            self._return_dict_hook)

        return hit_layer_pattern_list


def save_sub_res_hook(layer, input, output):
    layer.res_dict[layer.res_name] = output


def set_identity(parent_layer: nn.Layer,
                 layer_name: str,
                 layer_index: str=None) -> bool:
    """set the layer specified by layer_name and layer_index to Indentity.
    Args:
        parent_layer (nn.Layer): The parent layer of target layer specified by layer_name and layer_index.
        layer_name (str): The name of target layer to be set to Indentity.
        layer_index (str, optional): The index of target layer to be set to Indentity in parent_layer. Defaults to None.
    Returns:
        bool: True if successfully, False otherwise.
    """

    stop_after = False
    for sub_layer_name in parent_layer._sub_layers:
        if stop_after:
            parent_layer._sub_layers[sub_layer_name] = Identity()
            continue
        if sub_layer_name == layer_name:
            stop_after = True

    if layer_index and stop_after:
        stop_after = False
        for sub_layer_index in parent_layer._sub_layers[
                layer_name]._sub_layers:
            if stop_after:
                parent_layer._sub_layers[layer_name][
                    sub_layer_index] = Identity()
                continue
            if layer_index == sub_layer_index:
                stop_after = True

    return stop_after


def parse_pattern_str(pattern: str, parent_layer: nn.Layer) -> Union[
        None, List[Dict[str, Union[nn.Layer, str, None]]]]:
    """parse the string type pattern.
    Args:
        pattern (str): The pattern to discribe layer.
        parent_layer (nn.Layer): The root layer relative to the pattern.
    Returns:
        Union[None, List[Dict[str, Union[nn.Layer, str, None]]]]: None if failed. If successfully, the members are layers parsed in order:
                                                                [
                                                                    {"layer": first layer, "name": first layer's name parsed, "index": first layer's index parsed if exist},
                                                                    {"layer": second layer, "name": second layer's name parsed, "index": second layer's index parsed if exist},
                                                                    ...
                                                                ]
    """

    pattern_list = pattern.split(".")
    if not pattern_list:
        msg = f"The pattern('{pattern}') is illegal. Please check and retry."
        print(msg)
        return None

    layer_list = []
    while len(pattern_list) > 0:
        if '[' in pattern_list[0]:
            target_layer_name = pattern_list[0].split('[')[0]
            target_layer_index = pattern_list[0].split('[')[1].split(']')[0]
        else:
            target_layer_name = pattern_list[0]
            target_layer_index = None

        target_layer = getattr(parent_layer, target_layer_name, None)

        if target_layer is None:
            msg = f"Not found layer named('{target_layer_name}') specifed in pattern('{pattern}')."
            print(msg)
            return None

        if target_layer_index and target_layer:
            if int(target_layer_index) < 0 or int(target_layer_index) >= len(
                    target_layer):
                msg = f"Not found layer by index('{target_layer_index}') specifed in pattern('{pattern}'). The index should < {len(target_layer)} and > 0."
                print(msg)
                return None

            target_layer = target_layer[target_layer_index]

        layer_list.append({
            "layer": target_layer,
            "name": target_layer_name,
            "index": target_layer_index
        })

        pattern_list = pattern_list[1:]
        parent_layer = target_layer
    return layer_list




MODEL_URLS = {
    "InceptionV3":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/InceptionV3_pretrained.pdparams"
}

MODEL_STAGES_PATTERN = {
    "InceptionV3": [
        "inception_block_list[2]", "inception_block_list[3]",
        "inception_block_list[7]", "inception_block_list[8]",
        "inception_block_list[10]"
    ]
}

__all__ = MODEL_URLS.keys()
'''
InceptionV3 config: dict.
    key: inception blocks of InceptionV3.
    values: conv num in different blocks.
'''
NET_CONFIG = {
    "inception_a": [[192, 256, 288], [32, 64, 64]],
    "inception_b": [288],
    "inception_c": [[768, 768, 768, 768], [128, 160, 160, 192]],
    "inception_d": [768],
    "inception_e": [1280, 2048]
}


class ConvBNLayer(TheseusLayer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 padding=0,
                 groups=1,
                 act="relu"):
        super().__init__()
        self.act = act
        self.conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias_attr=False)
        self.bn = BatchNorm(num_filters, epsilon=0.001)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act:
            x = self.relu(x)
        return x


class InceptionStem(TheseusLayer):
    def __init__(self):
        super().__init__()
        self.conv_1a_3x3 = ConvBNLayer(
            num_channels=3,
            num_filters=32,
            filter_size=3,
            stride=2,
            act="relu")
        self.conv_2a_3x3 = ConvBNLayer(
            num_channels=32,
            num_filters=32,
            filter_size=3,
            stride=1,
            act="relu")
        self.conv_2b_3x3 = ConvBNLayer(
            num_channels=32,
            num_filters=64,
            filter_size=3,
            padding=1,
            act="relu")

        self.max_pool = MaxPool2D(kernel_size=3, stride=2, padding=0)
        self.conv_3b_1x1 = ConvBNLayer(
            num_channels=64, num_filters=80, filter_size=1, act="relu")
        self.conv_4a_3x3 = ConvBNLayer(
            num_channels=80, num_filters=192, filter_size=3, act="relu")

    def forward(self, x):
        x = self.conv_1a_3x3(x)
        x = self.conv_2a_3x3(x)
        x = self.conv_2b_3x3(x)
        x = self.max_pool(x)
        x = self.conv_3b_1x1(x)
        x = self.conv_4a_3x3(x)
        x = self.max_pool(x)
        return x


class InceptionA(TheseusLayer):
    def __init__(self, num_channels, pool_features):
        super().__init__()
        self.branch1x1 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=64,
            filter_size=1,
            act="relu")
        self.branch5x5_1 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=48,
            filter_size=1,
            act="relu")
        self.branch5x5_2 = ConvBNLayer(
            num_channels=48,
            num_filters=64,
            filter_size=5,
            padding=2,
            act="relu")

        self.branch3x3dbl_1 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=64,
            filter_size=1,
            act="relu")
        self.branch3x3dbl_2 = ConvBNLayer(
            num_channels=64,
            num_filters=96,
            filter_size=3,
            padding=1,
            act="relu")
        self.branch3x3dbl_3 = ConvBNLayer(
            num_channels=96,
            num_filters=96,
            filter_size=3,
            padding=1,
            act="relu")
        self.branch_pool = AvgPool2D(
            kernel_size=3, stride=1, padding=1, exclusive=False)
        self.branch_pool_conv = ConvBNLayer(
            num_channels=num_channels,
            num_filters=pool_features,
            filter_size=1,
            act="relu")

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.branch_pool(x)
        branch_pool = self.branch_pool_conv(branch_pool)
        x = paddle.concat(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=1)
        return x


class InceptionB(TheseusLayer):
    def __init__(self, num_channels):
        super().__init__()
        self.branch3x3 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=384,
            filter_size=3,
            stride=2,
            act="relu")
        self.branch3x3dbl_1 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=64,
            filter_size=1,
            act="relu")
        self.branch3x3dbl_2 = ConvBNLayer(
            num_channels=64,
            num_filters=96,
            filter_size=3,
            padding=1,
            act="relu")
        self.branch3x3dbl_3 = ConvBNLayer(
            num_channels=96,
            num_filters=96,
            filter_size=3,
            stride=2,
            act="relu")
        self.branch_pool = MaxPool2D(kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.branch_pool(x)

        x = paddle.concat([branch3x3, branch3x3dbl, branch_pool], axis=1)

        return x


class InceptionC(TheseusLayer):
    def __init__(self, num_channels, channels_7x7):
        super().__init__()
        self.branch1x1 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=192,
            filter_size=1,
            act="relu")

        self.branch7x7_1 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=channels_7x7,
            filter_size=1,
            stride=1,
            act="relu")
        self.branch7x7_2 = ConvBNLayer(
            num_channels=channels_7x7,
            num_filters=channels_7x7,
            filter_size=(1, 7),
            stride=1,
            padding=(0, 3),
            act="relu")
        self.branch7x7_3 = ConvBNLayer(
            num_channels=channels_7x7,
            num_filters=192,
            filter_size=(7, 1),
            stride=1,
            padding=(3, 0),
            act="relu")

        self.branch7x7dbl_1 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=channels_7x7,
            filter_size=1,
            act="relu")
        self.branch7x7dbl_2 = ConvBNLayer(
            num_channels=channels_7x7,
            num_filters=channels_7x7,
            filter_size=(7, 1),
            padding=(3, 0),
            act="relu")
        self.branch7x7dbl_3 = ConvBNLayer(
            num_channels=channels_7x7,
            num_filters=channels_7x7,
            filter_size=(1, 7),
            padding=(0, 3),
            act="relu")
        self.branch7x7dbl_4 = ConvBNLayer(
            num_channels=channels_7x7,
            num_filters=channels_7x7,
            filter_size=(7, 1),
            padding=(3, 0),
            act="relu")
        self.branch7x7dbl_5 = ConvBNLayer(
            num_channels=channels_7x7,
            num_filters=192,
            filter_size=(1, 7),
            padding=(0, 3),
            act="relu")

        self.branch_pool = AvgPool2D(
            kernel_size=3, stride=1, padding=1, exclusive=False)
        self.branch_pool_conv = ConvBNLayer(
            num_channels=num_channels,
            num_filters=192,
            filter_size=1,
            act="relu")

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = self.branch_pool(x)
        branch_pool = self.branch_pool_conv(branch_pool)

        x = paddle.concat(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=1)

        return x


class InceptionD(TheseusLayer):
    def __init__(self, num_channels):
        super().__init__()
        self.branch3x3_1 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=192,
            filter_size=1,
            act="relu")
        self.branch3x3_2 = ConvBNLayer(
            num_channels=192,
            num_filters=320,
            filter_size=3,
            stride=2,
            act="relu")
        self.branch7x7x3_1 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=192,
            filter_size=1,
            act="relu")
        self.branch7x7x3_2 = ConvBNLayer(
            num_channels=192,
            num_filters=192,
            filter_size=(1, 7),
            padding=(0, 3),
            act="relu")
        self.branch7x7x3_3 = ConvBNLayer(
            num_channels=192,
            num_filters=192,
            filter_size=(7, 1),
            padding=(3, 0),
            act="relu")
        self.branch7x7x3_4 = ConvBNLayer(
            num_channels=192,
            num_filters=192,
            filter_size=3,
            stride=2,
            act="relu")
        self.branch_pool = MaxPool2D(kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = self.branch_pool(x)

        x = paddle.concat([branch3x3, branch7x7x3, branch_pool], axis=1)
        return x


class InceptionE(TheseusLayer):
    def __init__(self, num_channels):
        super().__init__()
        self.branch1x1 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=320,
            filter_size=1,
            act="relu")
        self.branch3x3_1 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=384,
            filter_size=1,
            act="relu")
        self.branch3x3_2a = ConvBNLayer(
            num_channels=384,
            num_filters=384,
            filter_size=(1, 3),
            padding=(0, 1),
            act="relu")
        self.branch3x3_2b = ConvBNLayer(
            num_channels=384,
            num_filters=384,
            filter_size=(3, 1),
            padding=(1, 0),
            act="relu")

        self.branch3x3dbl_1 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=448,
            filter_size=1,
            act="relu")
        self.branch3x3dbl_2 = ConvBNLayer(
            num_channels=448,
            num_filters=384,
            filter_size=3,
            padding=1,
            act="relu")
        self.branch3x3dbl_3a = ConvBNLayer(
            num_channels=384,
            num_filters=384,
            filter_size=(1, 3),
            padding=(0, 1),
            act="relu")
        self.branch3x3dbl_3b = ConvBNLayer(
            num_channels=384,
            num_filters=384,
            filter_size=(3, 1),
            padding=(1, 0),
            act="relu")
        self.branch_pool = AvgPool2D(
            kernel_size=3, stride=1, padding=1, exclusive=False)
        self.branch_pool_conv = ConvBNLayer(
            num_channels=num_channels,
            num_filters=192,
            filter_size=1,
            act="relu")

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = paddle.concat(branch3x3, axis=1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = paddle.concat(branch3x3dbl, axis=1)

        branch_pool = self.branch_pool(x)
        branch_pool = self.branch_pool_conv(branch_pool)

        x = paddle.concat(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=1)
        return x


class Inception_V3(TheseusLayer):
    """
    Inception_V3
    Args:
        config: dict. config of Inception_V3.
        class_num: int=1000. The number of classes.
        pretrained: (True or False) or path of pretrained_model. Whether to load the pretrained model.
    Returns:
        model: nn.Layer. Specific Inception_V3 model depends on args.
    """

    def __init__(self,
                 config,
                 stages_pattern,
                 class_num=1000,
                 return_patterns=None,
                 return_stages=None):
        super().__init__()

        self.inception_a_list = config["inception_a"]
        self.inception_c_list = config["inception_c"]
        self.inception_b_list = config["inception_b"]
        self.inception_d_list = config["inception_d"]
        self.inception_e_list = config["inception_e"]

        self.inception_stem = InceptionStem()

        self.inception_block_list = nn.LayerList()
        for i in range(len(self.inception_a_list[0])):
            inception_a = InceptionA(self.inception_a_list[0][i],
                                     self.inception_a_list[1][i])
            self.inception_block_list.append(inception_a)

        for i in range(len(self.inception_b_list)):
            inception_b = InceptionB(self.inception_b_list[i])
            self.inception_block_list.append(inception_b)

        for i in range(len(self.inception_c_list[0])):
            inception_c = InceptionC(self.inception_c_list[0][i],
                                     self.inception_c_list[1][i])
            self.inception_block_list.append(inception_c)

        for i in range(len(self.inception_d_list)):
            inception_d = InceptionD(self.inception_d_list[i])
            self.inception_block_list.append(inception_d)

        for i in range(len(self.inception_e_list)):
            inception_e = InceptionE(self.inception_e_list[i])
            self.inception_block_list.append(inception_e)

        self.avg_pool = AdaptiveAvgPool2D(1)
        self.dropout = Dropout(p=0.2, mode="downscale_in_infer")
        stdv = 1.0 / math.sqrt(2048 * 1.0)
        self.fc = Linear(
            2048,
            class_num,
            weight_attr=ParamAttr(initializer=Uniform(-stdv, stdv)),
            bias_attr=ParamAttr())

        super().init_res(
            stages_pattern,
            return_patterns=return_patterns,
            return_stages=return_stages)

    def forward(self, x, return_features=False):
        x = self.inception_stem(x)
        for inception_block in self.inception_block_list:
            x = inception_block(x)
        x = self.avg_pool(x)
        x = paddle.reshape(x, shape=[-1, 2048])
        if return_features:
            return x
        x = self.dropout(x)
        x = self.fc(x)
        return x


def InceptionV3(pretrained=False, use_ssld=False, **kwargs):
    """
    InceptionV3
    Args:
        pretrained: bool=false or str. if `true` load pretrained parameters, `false` otherwise.
                    if str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `InceptionV3` model
    """
    model = Inception_V3(
        NET_CONFIG,
        stages_pattern=MODEL_STAGES_PATTERN["InceptionV3"],
        **kwargs)
    return model





