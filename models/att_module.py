"""
    Suppose the input of image is N x N
    - Input of AttentionLayer0: (N / 2) x (N / 2) ==> Maxpool x 4 ==> Output: (N / 32) x (N / 32)
    - Input of AttentionLayer1: (N / 4) x (N / 4) ==> Maxpool x 3 ==> Output: (N / 32) x (N / 32)
    - Input of AttentionLayer2: (N / 8) x (N / 8) ==> Maxpool x 2 ==> Output: (N / 32) x (N / 32)
    - Input of AttentionLayer3: (N /16) x (N /16) ==> Maxpool x 1 ==> Output: (N / 32) x (N / 32)

"""
import torch.nn as nn
from .senet import SEBottleneck
import torch.nn.functional as F

__all__ = ['AttentionLayer0', 'AttentionLayer1', 'AttentionLayer2', 'AttentionLayer3']


def _make_pre_last_layer(channels):
    return nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
    )


mask_setting = {
    'layer0': {'1': (48, 48), '2': (24, 24), '3': (12, 12), '4': (6, 6)},
    'layer1': {'1': (24, 24), '2': (12, 12), '3': (6, 6)},
    'layer2': {'1': (12, 12), '2': (6, 6)},
    'layer3': {'1': (6, 6)}
}


class AttentionLayer0(nn.Module):

    def __init__(self, inplanes, planes, groups, reduction):
        super(AttentionLayer0, self).__init__()

        size = mask_setting['layer0']
        self.first_layer = SEBottleneck(inplanes, planes, groups, reduction)

        self.trunk = nn.Sequential(SEBottleneck(inplanes, planes, groups, reduction),
                                   SEBottleneck(inplanes, planes, groups, reduction))

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.se1_down = SEBottleneck(inplanes, planes, groups, reduction)
        self.se1_add = SEBottleneck(inplanes, planes, groups, reduction)

        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.se2_down = SEBottleneck(inplanes, planes, groups, reduction)
        self.se2_add = SEBottleneck(inplanes, planes, groups, reduction)

        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.se3_down = SEBottleneck(inplanes, planes, groups, reduction)
        self.se3_add = SEBottleneck(inplanes, planes, groups, reduction)

        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.se4_bottom = nn.Sequential(SEBottleneck(inplanes, planes, groups, reduction),
                                        SEBottleneck(inplanes, planes, groups, reduction))

        self.up4 = nn.UpsamplingBilinear2d(size=size['4'])
        self.se4_up = SEBottleneck(inplanes, planes, groups, reduction)

        self.up3 = nn.UpsamplingBilinear2d(size=size['3'])
        self.se3_up = SEBottleneck(inplanes, planes, groups, reduction)

        self.up2 = nn.UpsamplingBilinear2d(size=size['2'])
        self.se2_up = SEBottleneck(inplanes, planes, groups, reduction)

        self.up1 = nn.UpsamplingBilinear2d(size=size['1'])
        self.se1_up = _make_pre_last_layer(inplanes)

        self.last_layer = SEBottleneck(inplanes, planes, groups, reduction)

    def forward(self, x):
        x = self.first_layer(x)
        trunk = self.trunk(x)

        pool1 = self.pool1(x)
        se1_down = self.se1_down(pool1)
        se1_add = self.se1_add(se1_down)

        pool2 = self.pool2(se1_down)
        se2_down = self.se2_down(pool2)
        se2_add = self.se2_add(se2_down)

        pool3 = self.pool3(se2_down)
        se3_down = self.se3_down(pool3)
        se3_add = self.se3_add(se3_down)

        pool4 = self.pool4(se3_down)
        se4_bottom = self.se4_bottom(pool4)

        up4 = self.up4(se4_bottom) + se3_add + se3_down
        se4_up = self.se4_up(up4)

        up3 = self.up3(se4_up) + se2_add + se2_down
        se3_up = self.se3_up(up3)

        up2 = self.up2(se3_up) + se1_add + se1_down
        se2_up = self.se2_up(up2)

        up1 = self.up1(se2_up) + trunk
        se1_up = self.se1_up(up1)

        out = (1 + se1_up) * trunk
        out = self.last_layer(out)

        return out


class AttentionLayer1(nn.Module):

    def __init__(self, inplanes, planes, groups, reduction):
        super(AttentionLayer1, self).__init__()
        self.size = mask_setting['layer1']

        self.first_layer = SEBottleneck(inplanes, planes, groups, reduction)

        self.trunk = nn.Sequential(SEBottleneck(inplanes, planes, groups, reduction),
                                   SEBottleneck(inplanes, planes, groups, reduction))

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.se1_down = SEBottleneck(inplanes, planes, groups, reduction)
        self.se1_add = SEBottleneck(inplanes, planes, groups, reduction)

        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.se2_down = SEBottleneck(inplanes, planes, groups, reduction)
        self.se2_add = SEBottleneck(inplanes, planes, groups, reduction)

        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.se3_bottom = nn.Sequential(SEBottleneck(inplanes, planes, groups, reduction),
                                        SEBottleneck(inplanes, planes, groups, reduction))

        self.se3_up = SEBottleneck(inplanes, planes, groups, reduction)
        self.se2_up = SEBottleneck(inplanes, planes, groups, reduction)
        self.se1_up = _make_pre_last_layer(inplanes)

        self.last_layer = SEBottleneck(inplanes, planes, groups, reduction)

    def forward(self, x):
        x = self.first_layer(x)
        trunk = self.trunk(x)

        pool1 = self.pool1(x)
        se1_down = self.se1_down(pool1)
        se1_add = self.se1_add(se1_down)

        pool2 = self.pool2(se1_down)
        se2_down = self.se2_down(pool2)
        se2_add = self.se2_add(se2_down)

        pool3 = self.pool3(se2_down)
        se3_bottom = self.se3_bottom(pool3)

        up3 = F.interpolate(se3_bottom, self.size['3']) + se2_add + se2_down
        se3_up = self.se3_up(up3)

        up2 = F.interpolate(se3_up, self.size['2']) + se1_add + se1_down
        se2_up = self.se2_up(up2)

        up1 = F.interpolate(se2_up, self.size['1']) + trunk
        se1_up = self.se1_up(up1)

        out = (1 + se1_up) * trunk
        out = self.last_layer(out)

        return out


class AttentionLayer2(nn.Module):

    def __init__(self, inplanes, planes, groups, reduction):
        super(AttentionLayer2, self).__init__()
        self.size = mask_setting['layer2']

        self.first_layer = SEBottleneck(inplanes, planes, groups, reduction)

        self.trunk = nn.Sequential(SEBottleneck(inplanes, planes, groups, reduction),
                                   SEBottleneck(inplanes, planes, groups, reduction))

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.se1_down = SEBottleneck(inplanes, planes, groups, reduction)
        self.se1_add = SEBottleneck(inplanes, planes, groups, reduction)

        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.se2_bottom = nn.Sequential(SEBottleneck(inplanes, planes, groups, reduction),
                                        SEBottleneck(inplanes, planes, groups, reduction))

        self.se2_up = SEBottleneck(inplanes, planes, groups, reduction)
        self.se1_up = _make_pre_last_layer(inplanes)

        self.last_layer = SEBottleneck(inplanes, planes, groups, reduction)

    def forward(self, x):
        x = self.first_layer(x)
        trunk = self.trunk(x)

        pool1 = self.pool1(x)
        se1_down = self.se1_down(pool1)
        se1_add = self.se1_add(se1_down)

        pool2 = self.pool2(se1_down)
        se2_bottom = self.se2_bottom(pool2)

        up2 = F.interpolate(se2_bottom, self.size['2']) + se1_add + se1_down
        se2_up = self.se2_up(up2)

        up1 = F.interpolate(se2_up, self.size['1']) + trunk
        se1_up = self.se1_up(up1)

        out = (1 + se1_up) * trunk
        out = self.last_layer(out)

        return out


class AttentionLayer3(nn.Module):

    def __init__(self, inplanes, planes, groups, reduction):
        super(AttentionLayer3, self).__init__()
        self.size = mask_setting['layer3']

        self.first_layer = SEBottleneck(inplanes, planes, groups, reduction)

        self.trunk = nn.Sequential(SEBottleneck(inplanes, planes, groups, reduction),
                                   SEBottleneck(inplanes, planes, groups, reduction))

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.se1_bottom = nn.Sequential(SEBottleneck(inplanes, planes, groups, reduction),
                                        SEBottleneck(inplanes, planes, groups, reduction))

        self.se1_up = _make_pre_last_layer(inplanes)

        self.last_layer = SEBottleneck(inplanes, planes, groups, reduction)

    def forward(self, x):
        x = self.first_layer(x)
        trunk = self.trunk(x)

        pool1 = self.pool1(x)
        se1_bottom = self.se1_bottom(pool1)

        up1 = F.interpolate(se1_bottom, self.size['1']) + trunk
        se1_up = self.se1_up(up1)

        out = (1 + se1_up) * trunk
        out = self.last_layer(out)

        return out
