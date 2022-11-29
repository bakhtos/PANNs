import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['init_layer',
           'init_bn',
           'ConvBlock',
           'ConvBlock5x5',
           'AttBlock',
           '_resnet_conv3x3',
           '_resnet_conv1x1',
           '_ResnetBasicBlock',
           '_ResnetBottleneck',
           '_ResNet',
           '_InvertedResidual',
           '_LeeNetConvBlock',
           '_LeeNetConvBlock2',
           '_DaiNetResBlock',
           '_resnet_conv3x1_wav1d',
           '_resnet_conv1x1_wav1d',
           '_ResnetBasicBlockWav1d',
           '_ResNetWav1d',
           '_ConvPreWavBlock',
           '_DropStripes',
           '_SpecAugmentation',
           '_interpolate',
           '_pad_framewise_output',
           '_count_parameters',
           '_count_flops']


def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

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

        self.init_weight()

    def init_weight(self):
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
        else:
            raise Exception('Incorrect argument!')

        return x


class ConvBlock5x5(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock5x5, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
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
        else:
            raise Exception('Incorrect argument!')

        return x


class AttBlock(nn.Module):
    def __init__(self, n_in, n_out, activation='linear', temperature=1.):
        super(AttBlock, self).__init__()

        self.activation = activation
        self.temperature = temperature
        self.att = nn.Conv1d(in_channels=n_in, out_channels=n_out,
                             kernel_size=1, stride=1, padding=0, bias=True)
        self.cla = nn.Conv1d(in_channels=n_in, out_channels=n_out,
                             kernel_size=1, stride=1, padding=0, bias=True)

        self.bn_att = nn.BatchNorm1d(n_out)
        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)
        init_bn(self.bn_att)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.clamp(self.att(x), -10, 10), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)


def _resnet_conv3x3(in_planes, out_planes):
    #3x3 convolution with padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, groups=1, bias=False, dilation=1)


def _resnet_conv1x1(in_planes, out_planes):
    #1x1 convolution
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


class _ResnetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(_ResnetBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('_ResnetBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in _ResnetBasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.stride = stride

        self.conv1 = _resnet_conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _resnet_conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.init_weights()

        self._sed_model = False

    @property
    def sed_model(self):
        return self._sed_model

    def init_weights(self):
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

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(_ResnetBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.stride = stride
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = _resnet_conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = _resnet_conv3x3(width, width)
        self.bn2 = norm_layer(width)
        self.conv3 = _resnet_conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.init_weights()

        self._sed_model = False

    @property
    def sed_model(self):
        return self._sed_model

    def init_weights(self):
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
    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(_ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
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
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1:
                downsample = nn.Sequential(
                        _resnet_conv1x1(self.inplanes, planes * block.expansion),
                        norm_layer(planes * block.expansion),
                )
                init_layer(downsample[0])
                init_bn(downsample[1])
            elif stride == 2:
                downsample = nn.Sequential(
                        nn.AvgPool2d(kernel_size=2),
                        _resnet_conv1x1(self.inplanes, planes * block.expansion),
                        norm_layer(planes * block.expansion),
                )
                init_layer(downsample[1])
                init_bn(downsample[2])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class _InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(_InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            _layers = [
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
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
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
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

        super(_LeeNetConvBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, bias=False)

        self.bn1 = nn.BatchNorm1d(out_channels)

        self.init_weight()

        self._sed_model = False

    @property
    def sed_model(self):
        return self._sed_model

    def init_weight(self):
        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, x, pool_size=1):
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_size != 1:
            x = F.max_pool1d(x, kernel_size=pool_size, padding=pool_size // 2)
        return x


class _LeeNetConvBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):

        super(_LeeNetConvBlock2, self).__init__()

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

        self.init_weight()

        self._sed_model = False

    @property
    def sed_model(self):
        return self._sed_model

    def init_weight(self):
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

        super(_DaiNetResBlock, self).__init__()

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

        self.init_weight()

        self._sed_model = False

    @property
    def sed_model(self):
        return self._sed_model

    def init_weight(self):
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

    def forward(self, input, pool_size=1):
        x = F.relu_(self.bn1(self.conv1(input)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.relu_(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        if input.shape == x.shape:
            x = F.relu_(x + input)
        else:
            x = F.relu(x + self.bn_downsample(self.downsample(input)))

        if pool_size != 1:
            x = F.max_pool1d(x, kernel_size=pool_size, padding=pool_size // 2)
        return x


def _resnet_conv3x1_wav1d(in_planes, out_planes, dilation):
    #3x3 convolution with padding
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=dilation, groups=1, bias=False, dilation=dilation)


def _resnet_conv1x1_wav1d(in_planes, out_planes):
    #1x1 convolution
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


class _ResnetBasicBlockWav1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(_ResnetBasicBlockWav1d, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if groups != 1 or base_width != 64:
            raise ValueError('_ResnetBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in _ResnetBasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.stride = stride

        self.conv1 = _resnet_conv3x1_wav1d(inplanes, planes, dilation=1)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _resnet_conv3x1_wav1d(planes, planes, dilation=2)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.init_weights()

        self._sed_model = False

    @property
    def sed_model(self):
        return self._sed_model

    def init_weights(self):
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
    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(_ResNetWav1d, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
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
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1:
                downsample = nn.Sequential(
                        _resnet_conv1x1_wav1d(self.inplanes, planes * block.expansion),
                        norm_layer(planes * block.expansion),
                )
                init_layer(downsample[0])
                init_bn(downsample[1])
            else:
                downsample = nn.Sequential(
                        nn.AvgPool1d(kernel_size=stride),
                        _resnet_conv1x1_wav1d(self.inplanes, planes * block.expansion),
                        norm_layer(planes * block.expansion),
                )
                init_layer(downsample[1])
                init_bn(downsample[2])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
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


class _ConvPreWavBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(_ConvPreWavBlock, self).__init__()

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

        self.init_weight()

        self._sed_model = False

    @property
    def sed_model(self):
        return self._sed_model

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)


    def forward(self, input, pool_size):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=pool_size)

        return x


# Class taken from torchlibrosa package
# https://github.com/qiuqiangkong/torchlibrosa/blob/master/torchlibrosa/augmentation.py
class _DropStripes(nn.Module):
    def __init__(self, dim, drop_width, stripes_num):
        """Drop stripes.

          :param int dim: Dimension along which to drop
          :param int drop_width: Maximum width of stripes to drop
          :param int stripes_num: How many stripes to drop
        """
        super(_DropStripes, self).__init__()

        assert dim in [2, 3]  # dim 2: time; dim 3: frequency

        self.dim = dim
        self.drop_width = drop_width
        self.stripes_num = stripes_num

    def forward(self, x):
        """Drop stripes of input.

        Input must be of shape (batch_size, channels, time_steps, freq_bins).
        """

        assert x.ndimension() == 4

        if self.training is False:
            return x

        else:
            batch_size = x.shape[0]
            total_width = x.shape[self.dim]

            for n in range(batch_size):
                self.transform_slice(x[n], total_width)

            return x

    def transform_slice(self, e, total_width):
        """e: (channels, time_steps, freq_bins)"""

        for _ in range(self.stripes_num):
            distance = torch.randint(low=0, high=self.drop_width, size=(1,))[0]
            bgn = torch.randint(low=0, high=total_width - distance, size=(1,))[
                0]

            if self.dim == 2:
                e[:, bgn: bgn + distance, :] = 0
            elif self.dim == 3:
                e[:, :, bgn: bgn + distance] = 0


# Class taken from torchlibrosa package
# https://github.com/qiuqiangkong/torchlibrosa/blob/master/torchlibrosa/augmentation.py
class _SpecAugmentation(nn.Module):
    def __init__(self, time_drop_width, time_stripes_num, freq_drop_width,
                 freq_stripes_num):
        """Spec augmetation.
        [ref] Park, D.S., Chan, W., Zhang, Y., Chiu, C.C., Zoph, B., Cubuk, E.D.
        and Le, Q.V., 2019. Specaugment: A simple data augmentation method
        for automatic speech recognition. arXiv preprint arXiv:1904.08779.
        Args:
          time_drop_width: int
          time_stripes_num: int
          freq_drop_width: int
          freq_stripes_num: int
        """

        super(_SpecAugmentation, self).__init__()

        self.time_dropper = _DropStripes(dim=2, drop_width=time_drop_width,
                                        stripes_num=time_stripes_num)

        self.freq_dropper = _DropStripes(dim=3, drop_width=freq_drop_width,
                                        stripes_num=freq_stripes_num)

    def forward(self, x):
        x = self.time_dropper(x)
        x = self.freq_dropper(x)
        return x


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
                  sum(list_bn) + sum(list_relu) + sum(list_pooling2d) + sum(
            list_pooling1d)

    return total_flops
