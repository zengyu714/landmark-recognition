"""
Reference
https://github.com/tengshaofeng/ResidualAttentionNetwork-pytorch/blob/master/Residual-Attention-Network/model/attention_module.py
"""
from collections import OrderedDict

import torch.nn as nn

from .att_module import *
from .senet import SEBottleneck, SEResNeXtBottleneck, SEResNetBottleneck
from utils.util import load_and_modify_pretrained_num_classes

__all__ = ['SeAtt', 'seatt154', 'seatt_base56', 'seatt_base92', 'seatt_resnext50_32x4d',
           'seatt_resnet50', 'seatt_resnext50_base', 'seatt_resnext50_in224']

model_urls = {
    'se_resnext50_32x4d': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth',
    'seatt_resnet50'    : 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth',
    'se154'             : 'http://data.lip6.fr/cadene/pretrainedmodels/senet154-c7b49a05.pth',
}


class SeAtt(nn.Module):

    def __init__(self, block=SEBottleneck, layers=[3, 8, 36, 3], att_layers=[1, 2, 5],
                 att_input_size=(48, 48), groups=64, reduction=16, dropout_p=0.2, inplanes=128,
                 input_3x3=True, downsample_kernel_size=3, downsample_padding=1, num_classes=1000):
        """
        Args:
            - block (nn.Module): Bottleneck class.
            - layers (list of ints): Number of residual blocks for 4 layers of the network (layer1...layer4).
            - att_layers (list of ints): Number of attention blocks for 3 layers of the network.
            - groups (int): Number of groups for the 3x3 convolution in each bottleneck block.
            - reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - dropout_p (float or None): Drop probability for the Dropout layer.
            - inplanes (int):  Number of input channels for layer1.
            - input_3x3 (bool): If `True`, use three 3x3 convolutions instead of a single 7x7 convolution in layer0.
            - downsample_kernel_size (int): Kernel size for downsampling convolutions in layer2, layer3 and layer4.
            - downsample_padding (int): Padding for downsampling convolutions in layer2, layer3 and layer4.
            - num_classes (int): Number of outputs in `last_linear` layer.
        """
        super(SeAtt, self).__init__()
        self.inplanes = inplanes
        self.mask_setting = gen_mask(att_input_size)

        if input_3x3:
            # CHANGE: use three 3x3 convolutions instead of a single 7x7 convolution in layer0
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2, ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))

        self.layer1 = self._make_se_block(
                block,
                planes=64,
                blocks=layers[0],
                groups=groups,
                reduction=reduction,
                downsample_kernel_size=1,
                downsample_padding=0
        )
        self.att_block1 = nn.Sequential(
                *[AttentionLayer1(256, planes=64, groups=groups, reduction=reduction,
                                  mask_setting=self.mask_setting)] * att_layers[0])
        self.layer2 = self._make_se_block(
                block,
                planes=128,
                blocks=layers[1],
                stride=2,
                groups=groups,
                reduction=reduction,
                downsample_kernel_size=downsample_kernel_size,
                downsample_padding=downsample_padding
        )
        self.att_block2 = nn.Sequential(
                *[AttentionLayer2(512, planes=128, groups=groups, reduction=reduction,
                                  mask_setting=self.mask_setting)] * att_layers[1])

        self.layer3 = self._make_se_block(
                block,
                planes=256,
                blocks=layers[2],
                stride=2,
                groups=groups,
                reduction=reduction,
                downsample_kernel_size=downsample_kernel_size,
                downsample_padding=downsample_padding
        )
        self.att_block3 = nn.Sequential(
                *[AttentionLayer3(1024, planes=256, groups=groups, reduction=reduction,
                                  mask_setting=self.mask_setting)] * att_layers[2])

        self.layer4 = self._make_se_block(
                block,
                planes=512,
                blocks=layers[3],
                stride=2,
                groups=groups,
                reduction=reduction,
                downsample_kernel_size=downsample_kernel_size,
                downsample_padding=downsample_padding
        )
        # CHANGE: pool size 7 -> 3
        layer1_size = att_input_size[0]
        self.avg_pool = nn.AvgPool2d(int(layer1_size / (2 ** 4)), stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_se_block(self, block, planes, blocks, groups, reduction, stride=1,
                       downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=downsample_kernel_size, stride=stride,
                              padding=downsample_padding, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        # N x 256 x (N / 4) x (N / 4)
        x = self.layer1(x)
        x = self.att_block1(x)
        # N x 512 x (N / 8) x (N / 8)
        x = self.layer2(x)
        x = self.att_block2(x)
        # N x 1024 x (N / 16) x (N / 16)
        x = self.layer3(x)
        x = self.att_block3(x)
        # N x 2048 x (N / 32) x (N / 32)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


def seatt154(pretrained=True, **kwargs):
    model = SeAtt(block=SEBottleneck, layers=[3, 8, 36, 3], att_layers=[1, 2, 5],
                  groups=64, reduction=16, dropout_p=0.2, inplanes=128, input_3x3=True,
                  downsample_kernel_size=3, downsample_padding=1)
    if pretrained:
        model = load_and_modify_pretrained_num_classes(model, model_urls['se154'], kwargs['num_classes'])
    return model


def seatt_resnext50_32x4d(pretrained=True, **kwargs):
    model = SeAtt(block=SEResNeXtBottleneck, layers=[3, 4, 6, 3], att_layers=[1, 2, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0)
    if pretrained:
        model = load_and_modify_pretrained_num_classes(model, model_urls['se_resnext50_32x4d'], kwargs['num_classes'])
    return model


def seatt_resnext50_base(pretrained=True, **kwargs):
    """
    The difference with `seatt_resnext50_32x4d` is the number of attention module layer
    """
    model = SeAtt(block=SEResNeXtBottleneck, layers=[3, 4, 6, 3], att_layers=[1, 1, 1], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0)
    if pretrained:
        model = load_and_modify_pretrained_num_classes(model, model_urls['se_resnext50_32x4d'], kwargs['num_classes'])
    return model


def seatt_resnext50_in224(pretrained=True, **kwargs):
    """
    The difference with `seatt_resnext50_32x4d` is the number of attention module layer
    """
    model = SeAtt(block=SEResNeXtBottleneck, layers=[3, 4, 6, 3], att_layers=[1, 1, 1], groups=32, reduction=16,
                  att_input_size=(112, 112), dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0)
    if pretrained:
        model = load_and_modify_pretrained_num_classes(model, model_urls['se_resnext50_32x4d'], kwargs['num_classes'])
    return model


def seatt_resnet50(pretrained=True, **kwargs):
    model = SeAtt(SEResNetBottleneck, layers=[3, 4, 6, 3], att_layers=[1, 2, 5], groups=1, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0)
    if pretrained:
        model = load_and_modify_pretrained_num_classes(model, model_urls['seatt_resnet50'], kwargs['num_classes'])
    return model


def seatt_base56(pretrained=False, **kwargs):
    model = SeAtt(SEResNetBottleneck, layers=[1, 1, 1, 3], att_layers=[1, 1, 1], groups=1, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0, num_classes=kwargs['num_classes'])
    if pretrained:
        model = load_and_modify_pretrained_num_classes(model, model_urls['seatt_resnet50'], kwargs['num_classes'])
    return model


def seatt_base92(pretrained=False, **kwargs):
    model = SeAtt(SEResNetBottleneck, layers=[1, 1, 1, 3], att_layers=[1, 2, 3], groups=1, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0, num_classes=kwargs['num_classes'])
    if pretrained:
        model = load_and_modify_pretrained_num_classes(model, model_urls['seatt_resnet50'], kwargs['num_classes'])
    return model
