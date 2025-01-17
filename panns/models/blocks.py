import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.functional import mask_along_axis_iid

__all__ = ['init_layer',
           'init_bn',
           '_ConvBlock',
           '_ConvBlock5x5',
           '_ConvPreWavBlock',
           '_AttBlock',
           '_ResnetBasicBlock',
           '_ResnetBottleneck',
           '_ResNet',
           '_ResnetBasicBlockWav1d',
           '_ResNetWav1d',
           '_ConvBnV1',
           '_ConvDw',
           '_ConvBnV2',
           '_Conv1x1Bn',
           '_InvertedResidual',
           '_LeeNetConvBlock',
           '_LeeNetConvBlock2',
           '_DaiNetResBlock',
           '_SpecAugmentation',
           '_interpolate',
           '_pad_framewise_output',
           '_count_parameters',
           '_count_flops']


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, x, pool_size=(2, 2), pool_type='avg'):

        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2

        return x


class _ConvBlock5x5(nn.Module):
    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)

        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, x, pool_size=(2, 2), pool_type='avg'):

        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2

        return x


class _ConvPreWavBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.conv2 = nn.Conv1d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3, stride=1, dilation=2,
                               padding=2, bias=False)

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, x, pool_size):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=pool_size)

        return x


class _AttBlock(nn.Module):
    def __init__(self, n_in, n_out, activation='linear', temperature=1.):

        super().__init__()

        assert activation in ['linear', 'sigmoid']

        self.activation = activation
        self.temperature = temperature
        self.att = nn.Conv1d(in_channels=n_in, out_channels=n_out,
                             kernel_size=1, stride=1, padding=0, bias=True)
        self.cla = nn.Conv1d(in_channels=n_in, out_channels=n_out,
                             kernel_size=1, stride=1, padding=0, bias=True)

        self.bn_att = nn.BatchNorm1d(n_out)

        init_layer(self.att)
        init_layer(self.cla)
        init_bn(self.bn_att)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.clamp(self.att(x), -10, 10), dim=-1)
        cla = self.cla(x)
        if self.activation == 'sigmoid': cla = torch.sigmoid(cla)
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla


class _ResnetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, *, in_planes, planes, stride=1, downsample=None,
                 norm_layer=None, **kwargs):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.stride = stride
        self.downsample = downsample

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1,
                               padding=1, groups=1, bias=False, dilation=1)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1,
                               padding=1, groups=1, bias=False, dilation=1)
        self.bn2 = norm_layer(planes)

        init_layer(self.conv1)
        init_bn(self.bn1)
        init_layer(self.conv2)
        init_bn(self.bn2)
        nn.init.constant_(self.bn2.weight, 0)

    def forward(self, x):
        identity = x

        if self.stride == 2:
            out = F.avg_pool2d(x, kernel_size=(2, 2))
        else:
            out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = F.dropout(out, p=0.1, training=self.training)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class _ResnetBottleneck(nn.Module):
    expansion = 4

    def __init__(self, *, in_planes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, norm_layer=None, **kwargs):

        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.stride = stride
        self.downsample = downsample

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv2d(in_planes, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=1,
                               padding=1, groups=1, bias=False, dilation=1)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes*self.expansion, kernel_size=1,
                               stride=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        init_layer(self.conv1)
        init_bn(self.bn1)
        init_layer(self.conv2)
        init_bn(self.bn2)
        init_layer(self.conv3)
        init_bn(self.bn3)
        nn.init.constant_(self.bn3.weight, 0)

    def forward(self, x):
        identity = x

        if self.stride == 2:
            x = F.avg_pool2d(x, kernel_size=(2, 2))

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = F.dropout(out, p=0.1, training=self.training)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class _ResNet(nn.Module):
    def __init__(self, *, block, layers,
                 groups=1, width_per_group=64,
                 replace_stride_with_dilation=(False, False, False),
                 norm_layer=None):
        """

        Args:
            block:
            layers:
            groups:
            width_per_group:
            replace_stride_with_dilation: 3-tuple of bools;
                indicates whether to replace stride with dilation in
                each of the three layers (default (False, False, False))
            norm_layer:
        """
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.in_planes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_planes != planes * block.expansion:
            if stride == 1:
                downsample = nn.Sequential(
                        nn.Conv2d(self.in_planes, planes*block.expansion,
                                  kernel_size=1, stride=1, bias=False),
                        norm_layer(planes * block.expansion),
                )
                init_layer(downsample[0])
                init_bn(downsample[1])
            elif stride == 2:
                downsample = nn.Sequential(
                        nn.AvgPool2d(kernel_size=2),
                        nn.Conv2d(self.in_planes, planes * block.expansion,
                                  kernel_size=1, stride=1, bias=False),
                        norm_layer(planes * block.expansion),
                )
                init_layer(downsample[1])
                init_bn(downsample[2])

        layers = [block(in_planes=self.in_planes, planes=planes, stride=stride,
                        downsample=downsample, groups=self.groups,
                        base_width=self.base_width, dilation=previous_dilation,
                        norm_layer=norm_layer)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(in_planes=self.in_planes, planes=planes,
                                groups=self.groups,
                                base_width=self.base_width,
                                dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class _ResnetBasicBlockWav1d(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None,
                 dilation=1, norm_layer=None, **kwargs):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.stride = stride

        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=1,
                               padding=dilation, groups=1, bias=False,
                               dilation=1)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1,
                               padding=dilation, groups=1, bias=False,
                               dilation=2)

        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        init_layer(self.conv1)
        init_bn(self.bn1)
        init_layer(self.conv2)
        init_bn(self.bn2)
        nn.init.constant_(self.bn2.weight, 0)

    def forward(self, x):
        identity = x

        if self.stride != 1:
            out = F.max_pool1d(x, kernel_size=self.stride)
        else:
            out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = F.dropout(out, p=0.1, training=self.training)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class _ResNetWav1d(nn.Module):
    def __init__(self, *, block, layers,
                 groups=1, width_per_group=64,
                 norm_layer=None):

        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.in_planes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=4)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=4)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=4)
        self.layer5 = self._make_layer(block, 1024, layers[4], stride=4)
        self.layer6 = self._make_layer(block, 1024, layers[5], stride=4)
        self.layer7 = self._make_layer(block, 2048, layers[6], stride=4)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_planes != planes * block.expansion:
            if stride == 1:
                downsample = nn.Sequential(
                        nn.Conv1d(self.in_planes, planes * block.expansion,
                                  kernel_size=1, stride=1, bias=False),
                        norm_layer(planes * block.expansion),
                )
                init_layer(downsample[0])
                init_bn(downsample[1])
            else:
                downsample = nn.Sequential(
                        nn.AvgPool1d(kernel_size=stride),
                        nn.Conv1d(self.in_planes, planes * block.expansion,
                                  kernel_size=1, stride=1, bias=False),
                        norm_layer(planes * block.expansion),
                )
                init_layer(downsample[1])
                init_bn(downsample[2])

        layers = [block(in_planes=self.in_planes, planes=planes, stride=stride,
                        downsample=downsample, groups=self.groups,
                        base_width=self.base_width, dilation=previous_dilation,
                        norm_layer=norm_layer)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(in_planes=self.in_planes, planes=planes,
                                groups=self.groups,
                                base_width=self.base_width,
                                dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        return x


def _ConvBnV1(inp, oup, stride):
    _layers = [
        nn.Conv2d(inp, oup, 3, 1, 1, bias=False),
        nn.AvgPool2d(stride),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    ]
    _layers = nn.Sequential(*_layers)
    init_layer(_layers[0])
    init_bn(_layers[2])
    return _layers


def _ConvDw(inp, oup, stride):
    _layers = [
        nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
        nn.AvgPool2d(stride),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    ]
    _layers = nn.Sequential(*_layers)
    init_layer(_layers[0])
    init_bn(_layers[2])
    init_layer(_layers[4])
    init_bn(_layers[5])
    return _layers


def _ConvBnV2(inp, oup, stride):
    _layers = [
        nn.Conv2d(inp, oup, 3, 1, 1, bias=False),
        nn.AvgPool2d(stride),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    ]
    _layers = nn.Sequential(*_layers)
    init_layer(_layers[0])
    init_bn(_layers[2])
    return _layers


def _Conv1x1Bn(inp, oup):
    _layers = nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
    )
    init_layer(_layers[0])
    init_bn(_layers[1])
    return _layers


class _InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = stride == 1 and inp == oup

        if expand_ratio == 1:
            _layers = [
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim,
                          bias=False),
                nn.AvgPool2d(stride),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
            ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[2])
            init_layer(_layers[4])
            init_bn(_layers[5])
            self.conv = _layers
        else:
            _layers = [
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim,
                          bias=False),
                nn.AvgPool2d(stride),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
            ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[1])
            init_layer(_layers[3])
            init_bn(_layers[5])
            init_layer(_layers[7])
            init_bn(_layers[8])
            self.conv = _layers

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class _LeeNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, bias=False)

        self.bn1 = nn.BatchNorm1d(out_channels)

        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, x, pool_size=1):
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_size != 1:
            x = F.max_pool1d(x, kernel_size=pool_size, padding=pool_size // 2)
        return x


class _LeeNetConvBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, bias=False)

        self.conv2 = nn.Conv1d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=1,
                               padding=kernel_size // 2, bias=False)

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, x, pool_size=1):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_size != 1:
            x = F.max_pool1d(x, kernel_size=pool_size, padding=pool_size // 2)
        return x


class _DaiNetResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):

        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=1,
                               padding=kernel_size // 2, bias=False)

        self.conv2 = nn.Conv1d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=1,
                               padding=kernel_size // 2, bias=False)

        self.conv3 = nn.Conv1d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=1,
                               padding=kernel_size // 2, bias=False)

        self.conv4 = nn.Conv1d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=1,
                               padding=kernel_size // 2, bias=False)

        self.downsample = nn.Conv1d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=1, stride=1,
                                    padding=0, bias=False)

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.bn4 = nn.BatchNorm1d(out_channels)
        self.bn_downsample = nn.BatchNorm1d(out_channels)

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.conv4)
        init_layer(self.downsample)
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)
        init_bn(self.bn4)
        nn.init.constant_(self.bn4.weight, 0)
        init_bn(self.bn_downsample)

    def forward(self, data, pool_size=1):
        x = F.relu_(self.bn1(self.conv1(data)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.relu_(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        if data.shape == x.shape:
            x = F.relu_(x + data)
        else:
            x = F.relu(x + self.bn_downsample(self.downsample(data)))

        if pool_size != 1:
            x = F.max_pool1d(x, kernel_size=pool_size, padding=pool_size // 2)
        return x


class _SpecAugmentation(nn.Module):
    def __init__(self, stripes_num, mask_param, axis, mask_value=0.0):
        """ Spectrogram augmentation.

        Masks will be applied from indices ``[v_0, v_0 + v)``,
        where ``v`` is sampled from ``uniform(0, max_v)`` and
        ``v_0`` from ``uniform(0, specgrams.size(axis) - v)``,
        with ``max_v = mask_param`` when ``p = 1.0`` and
        ``max_v = min(mask_param, floor(specgrams.size(axis) * p))`` otherwise.
        Note: masks are sampled independently and so may overlap

        [ref] Park, D.S., Chan, W., Zhang, Y., Chiu, C.C., Zoph, B., Cubuk, E.D.
        and Le, Q.V., 2019. Specaugment: A simple data augmentation method
        for automatic speech recognition. arXiv preprint arXiv:1904.08779.

        Args:
            stripes_num: int, amount of masks to apply
                (Note: masks are sampled independently
            mask_param: int, Number of columns to be masked will be uniformly
                sampled from [0, mask_param]
            axis: int, Axis to apply masking on (2 -> frequency, 3 -> time)
            mask_value: float, Value to assign to the masked columns
                (default 0.0)
        """

        super().__init__()

        self.stripes_num = stripes_num
        self.mask_param = mask_param
        self.axis = axis
        self.mask_value = mask_value

    def forward(self, specgrams):
        for i in range(self.stripes_num):
            specgrams = mask_along_axis_iid(specgrams, self.mask_param,
                                            self.mask_value, self.axis)
        return specgrams


def _interpolate(x, ratio):
    """Interpolate data in time domain. This is used to compensate the 
    resolution reduction in downsampling of a CNN.
    
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate

    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def _pad_framewise_output(framewise_output, frames_num):
    """Pad framewise_output to the same length as input frames. The pad value 
    is the same as the value of the last frame.

    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad

    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    pad = framewise_output[:, -1:, :].repeat(1,
                                             frames_num -
                                             framewise_output.shape[1], 1)
    """tensor for padding"""

    output = torch.cat((framewise_output, pad), dim=1)
    """(batch_size, frames_num, classes_num)"""

    return output


def _count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _count_flops(model, audio_length):
    """Count flops. Code modified from others' implementation.
    """
    multiply_adds = True
    list_conv2d = []

    def conv2d_hook(self, x, output):
        batch_size, input_channels, input_height, input_width = x[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (
                self.in_channels / self.groups) * (
                         2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_conv2d.append(flops)

    list_conv1d = []

    def conv1d_hook(self, x, output):
        batch_size, input_channels, input_length = x[0].size()
        output_channels, output_length = output[0].size()

        kernel_ops = self.kernel_size[0] * (self.in_channels / self.groups) * (
            2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_length

        list_conv1d.append(flops)

    list_linear = []

    def linear_hook(self, x, output):
        batch_size = x[0].size(0) if x[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn = []

    def bn_hook(self, x, output):
        list_bn.append(x[0].nelement() * 2)

    list_relu = []

    def relu_hook(self, x, output):
        list_relu.append(x[0].nelement() * 2)

    list_pooling2d = []

    def pooling2d_hook(self, x, output):
        batch_size, input_channels, input_height, input_width = x[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_pooling2d.append(flops)

    list_pooling1d = []

    def pooling1d_hook(self, x, output):
        batch_size, input_channels, input_length = x[0].size()
        output_channels, output_length = output[0].size()

        kernel_ops = self.kernel_size[0]
        bias_ops = 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_length

        list_pooling2d.append(flops)

    def foo(net):
        children = list(net.children())
        if not children:
            if isinstance(net, nn.Conv2d):
                net.register_forward_hook(conv2d_hook)
            elif isinstance(net, nn.Conv1d):
                net.register_forward_hook(conv1d_hook)
            elif isinstance(net, nn.Linear):
                net.register_forward_hook(linear_hook)
            elif isinstance(net, nn.BatchNorm2d) or isinstance(net,
                                                               nn.BatchNorm1d):
                net.register_forward_hook(bn_hook)
            elif isinstance(net, nn.ReLU):
                net.register_forward_hook(relu_hook)
            elif isinstance(net, nn.AvgPool2d) or isinstance(net, nn.MaxPool2d):
                net.register_forward_hook(pooling2d_hook)
            elif isinstance(net, nn.AvgPool1d) or isinstance(net, nn.MaxPool1d):
                net.register_forward_hook(pooling1d_hook)
            else:
                print('Warning: flop of module {} is not counted!'.format(net))
            return
        for c in children:
            foo(c)

    # Register hook
    foo(model)

    device = next(model.parameters()).device
    data = torch.rand(1, audio_length).to(device)

    model(data)

    total_flops = sum(list_conv2d) + sum(list_conv1d) + sum(list_linear) + \
                  sum(list_bn) + sum(list_relu) + sum(list_pooling2d) +\
                  sum(list_pooling1d)

    return total_flops
