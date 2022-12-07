import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank

from panns.data.mixup import mixup
from .blocks import *

__all__ = ['Cnn6',
           'Cnn10',
           'Cnn14',
           'Cnn14DecisionLevel',
           'Cnn14DecisionLevelAtt',
           'Cnn14Wavegram',
           'Cnn14WavegramLogmel',
           'ResNet22',
           'ResNet38',
           'ResNet54',
           'Res1dNet31',
           'Res1dNet51',
           'MobileNetV1',
           'MobileNetV2',
           'LeeNet11',
           'LeeNet24',
           'DaiNet19',
           'TransferModel'
           ]


class Cnn6(nn.Module):
    def __init__(self, *, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num, **kwargs):
        """

        Args:
            sample_rate:
            window_size:
            hop_size:
            mel_bins:
            fmin:
            fmax:
            classes_num:
            **kwargs: 'window', 'center', 'pad_mode' for Spectrogram,
                'ref', 'amin', 'top_db' for LogmelFilterbank
        """

        super().__init__()

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size,
                                                 hop_length=hop_size,
                                                 win_length=window_size,
                                                 window=kwargs.get('window',
                                                                   'hann'),
                                                 center=kwargs.get('center',
                                                                   True),
                                                 pad_mode=kwargs.get(
                                                         'pad_mode', 'reflect'),
                                                 freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate,
                                                 n_fft=window_size,
                                                 n_mels=mel_bins, fmin=fmin,
                                                 fmax=fmax,
                                                 ref=kwargs.get('ref', 1.0),
                                                 amin=kwargs.get('amin', 1e-10),
                                                 top_db=kwargs.get('top_db',
                                                                   None),
                                                 freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = _SpecAugmentation(time_drop_width=64,
                                                time_stripes_num=2,
                                                freq_drop_width=8,
                                                freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = _ConvBlock5x5(in_channels=1, out_channels=64)
        self.conv_block2 = _ConvBlock5x5(in_channels=64, out_channels=128)
        self.conv_block3 = _ConvBlock5x5(in_channels=128, out_channels=256)
        self.conv_block4 = _ConvBlock5x5(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)

        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, batch, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(batch)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        return clipwise_output, embedding


class Cnn10(nn.Module):
    def __init__(self, *, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num, **kwargs):
        """

        Args:
            sample_rate:
            window_size:
            hop_size:
            mel_bins:
            fmin:
            fmax:
            classes_num:
            **kwargs: 'window', 'center', 'pad_mode' for Spectrogram,
                'ref', 'amin', 'top_db' for LogmelFilterbank
        """

        super().__init__()

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size,
                                                 hop_length=hop_size,
                                                 win_length=window_size,
                                                 window=kwargs.get('window',
                                                                   'hann'),
                                                 center=kwargs.get('center',
                                                                   True),
                                                 pad_mode=kwargs.get(
                                                         'pad_mode', 'reflect'),
                                                 freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate,
                                                 n_fft=window_size,
                                                 n_mels=mel_bins, fmin=fmin,
                                                 fmax=fmax,
                                                 ref=kwargs.get('ref', 1.0),
                                                 amin=kwargs.get('amin', 1e-10),
                                                 top_db=kwargs.get('top_db',
                                                                   None),
                                                 freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = _SpecAugmentation(time_drop_width=64,
                                                time_stripes_num=2,
                                                freq_drop_width=8,
                                                freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = _ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = _ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = _ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = _ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)

        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, batch, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(
            batch)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        return clipwise_output, embedding


class Cnn14(nn.Module):
    def __init__(self, *, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num, spec_aug=True, mixup_time=False,
                 mixup_freq=True, dropout=True, **kwargs):
        """

        Args:
            sample_rate:
            window_size:
            hop_size:
            mel_bins:
            fmin:
            fmax:
            classes_num:
            **kwargs: 'window', 'center', 'pad_mode' for Spectrogram
                    (defaults 'hann', 'center', 'reflect'),
                'ref', 'amin', 'top_db' for LogmelFilterbank
                    (defaults 1.0, 1e-10, None),
                'num_features' for BatchNorm2d (default 64).
                'embedding_size' for the amount of neurons connecting the
                    last two fully connected layers (default 2048)
        """
        super().__init__()

        self.spec_aug = spec_aug
        self.mixup_time = mixup_time
        self.mixup_freq = mixup_freq
        self.dropout = dropout

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size,
                                                 hop_length=hop_size,
                                                 win_length=window_size,
                                                 window=kwargs.get('window',
                                                                   'hann'),
                                                 center=kwargs.get('center',
                                                                   True),
                                                 pad_mode=kwargs.get(
                                                         'pad_mode', 'reflect'),
                                                 freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate,
                                                 n_fft=window_size,
                                                 n_mels=mel_bins, fmin=fmin,
                                                 fmax=fmax,
                                                 ref=kwargs.get('ref', 1.0),
                                                 amin=kwargs.get('amin', 1e-10),
                                                 top_db=kwargs.get('top_db',
                                                                   None),
                                                 freeze_parameters=True)
        # Spec augmenter
        if spec_aug:
            self.spec_augmenter = _SpecAugmentation(time_drop_width=64,
                                                    time_stripes_num=2,
                                                    freq_drop_width=8,
                                                    freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(kwargs.get('num_features', 64))

        self.conv_block1 = _ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = _ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = _ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = _ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = _ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = _ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, kwargs.get('embedding_size', 2048),
                             bias=True)
        self.fc_audioset = nn.Linear(kwargs.get('embedding_size', 2048),
                                     classes_num, bias=True)

        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, batch, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = batch

        # Mixup in time domain
        if self.mixup_time and self.training and mixup_lambda is not None:
            x = mixup(x, mixup_lambda)

        x = self.spectrogram_extractor(x)  # (batch_size,1,time_steps,freq_bins)
        x = self.logmel_extractor(x)  # (batch_size,1,time_steps,mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            if self.spec_aug:
                x = self.spec_augmenter(x)
            # Mixup on spectrogram
            if self.mixup_freq and mixup_lambda is not None:
                x = mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training) if self.dropout else x
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training) if self.dropout else x
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training) if self.dropout else x
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training) if self.dropout else x
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training) if self.dropout else x
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training) if self.dropout else x
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        return clipwise_output, embedding


class Cnn14DecisionLevel(nn.Module):
    def __init__(self, *, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num, decision_level='max', **kwargs):
        """

        Args:
            sample_rate:
            window_size:
            hop_size:
            mel_bins:
            fmin:
            fmax:
            classes_num:
            **kwargs: 'window', 'center', 'pad_mode' for Spectrogram,
                'ref', 'amin', 'top_db' for LogmelFilterbank
        """

        super().__init__()

        self.interpolate_ratio = 32  # Downsampled ratio
        self.decision_level = decision_level

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size,
                                                 hop_length=hop_size,
                                                 win_length=window_size,
                                                 window=kwargs.get('window',
                                                                   'hann'),
                                                 center=kwargs.get('center',
                                                                   True),
                                                 pad_mode=kwargs.get(
                                                         'pad_mode', 'reflect'),
                                                 freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate,
                                                 n_fft=window_size,
                                                 n_mels=mel_bins, fmin=fmin,
                                                 fmax=fmax,
                                                 ref=kwargs.get('ref', 1.0),
                                                 amin=kwargs.get('amin', 1e-10),
                                                 top_db=kwargs.get('top_db',
                                                                   None),
                                                 freeze_parameters=True)
        # Spec augmenter
        self.spec_augmenter = _SpecAugmentation(time_drop_width=64,
                                                time_stripes_num=2,
                                                freq_drop_width=8,
                                                freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = _ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = _ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = _ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = _ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = _ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = _ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, batch, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(
            batch)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        frames_num = x.shape[2]

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        segmentwise_output = torch.sigmoid(self.fc_audioset(x))
        if self.decision_level == 'max':
            (clipwise_output, _) = torch.max(segmentwise_output, dim=1)
        elif self.decision_level == 'avg':
            clipwise_output = torch.mean(segmentwise_output, dim=1)

        # Get framewise output
        framewise_output = _interpolate(segmentwise_output,
                                        self.interpolate_ratio)
        framewise_output = _pad_framewise_output(framewise_output, frames_num)

        return clipwise_output, framewise_output


class Cnn14DecisionLevelAtt(nn.Module):
    def __init__(self, *, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num, **kwargs):
        """

        Args:
            sample_rate:
            window_size:
            hop_size:
            mel_bins:
            fmin:
            fmax:
            classes_num:
            **kwargs: 'window', 'center', 'pad_mode' for Spectrogram,
                'ref', 'amin', 'top_db' for LogmelFilterbank
        """

        super().__init__()

        self.interpolate_ratio = 32  # Downsampled ratio

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size,
                                                 hop_length=hop_size,
                                                 win_length=window_size,
                                                 window=kwargs.get('window',
                                                                   'hann'),
                                                 center=kwargs.get('center',
                                                                   True),
                                                 pad_mode=kwargs.get(
                                                         'pad_mode', 'reflect'),
                                                 freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate,
                                                 n_fft=window_size,
                                                 n_mels=mel_bins, fmin=fmin,
                                                 fmax=fmax,
                                                 ref=kwargs.get('ref', 1.0),
                                                 amin=kwargs.get('amin', 1e-10),
                                                 top_db=kwargs.get('top_db',
                                                                   None),
                                                 freeze_parameters=True)
        # Spec augmenter
        self.spec_augmenter = _SpecAugmentation(time_drop_width=64,
                                                time_stripes_num=2,
                                                freq_drop_width=8,
                                                freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = _ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = _ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = _ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = _ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = _ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = _ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.att_block = _AttBlock(2048, classes_num, activation='sigmoid')

        init_bn(self.bn0)
        init_layer(self.fc1)

    def forward(self, batch, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(
            batch)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        frames_num = x.shape[2]

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)
        (clipwise_output, _, segmentwise_output) = self.att_block(x)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        # Get framewise output
        framewise_output = _interpolate(segmentwise_output,
                                        self.interpolate_ratio)
        framewise_output = _pad_framewise_output(framewise_output, frames_num)

        return clipwise_output, framewise_output


class Cnn14Wavegram(nn.Module):
    def __init__(self, *, classes_num, **kwargs):
        super().__init__()

        self.pre_conv0 = nn.Conv1d(in_channels=1, out_channels=64,
                                   kernel_size=11, stride=5, padding=5,
                                   bias=False)
        self.pre_bn0 = nn.BatchNorm1d(64)
        self.pre_block1 = _ConvPreWavBlock(64, 64)
        self.pre_block2 = _ConvPreWavBlock(64, 128)
        self.pre_block3 = _ConvPreWavBlock(128, 128)
        self.pre_block4 = _ConvBlock(in_channels=4, out_channels=64)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = _ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = _ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = _ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = _ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = _ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = _ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        init_layer(self.pre_conv0)
        init_bn(self.pre_bn0)
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, batch, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        # Wavegram
        a1 = F.relu_(self.pre_bn0(self.pre_conv0(batch[:, None, :])))
        a1 = self.pre_block1(a1, pool_size=4)
        a1 = self.pre_block2(a1, pool_size=4)
        a1 = self.pre_block3(a1, pool_size=4)
        a1 = a1.reshape((a1.shape[0], -1, 32, a1.shape[-1])).transpose(2, 3)
        a1 = self.pre_block4(a1, pool_size=(2, 1))

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            a1 = mixup(a1, mixup_lambda)

        x = a1
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        return clipwise_output, embedding


class Cnn14WavegramLogmel(nn.Module):
    def __init__(self, *, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num, multiplier=1, **kwargs):
        """

        Args:
            sample_rate:
            window_size:
            hop_size:
            mel_bins:
            fmin:
            fmax:
            classes_num:
            **kwargs: 'window', 'center', 'pad_mode' for Spectrogram,
                'ref', 'amin', 'top_db' for LogmelFilterbank
        """

        super().__init__()

        self.multiplier = multiplier
        self.pre_conv0 = nn.Conv1d(in_channels=1, out_channels=64,
                                   kernel_size=11, stride=5, padding=5,
                                   bias=False)
        self.pre_bn0 = nn.BatchNorm1d(64)
        self.pre_block1 = _ConvPreWavBlock(64, 64)
        self.pre_block2 = _ConvPreWavBlock(64, 128)
        self.pre_block3 = _ConvPreWavBlock(128, 128*multiplier)
        self.pre_block4 = _ConvBlock(in_channels=4, out_channels=64)

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size,
                                                 hop_length=hop_size,
                                                 win_length=window_size,
                                                 window=kwargs.get('window',
                                                                   'hann'),
                                                 center=kwargs.get('center',
                                                                   True),
                                                 pad_mode=kwargs.get(
                                                         'pad_mode', 'reflect'),
                                                 freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate,
                                                 n_fft=window_size,
                                                 n_mels=mel_bins, fmin=fmin,
                                                 fmax=fmax,
                                                 ref=kwargs.get('ref', 1.0),
                                                 amin=kwargs.get('amin', 1e-10),
                                                 top_db=kwargs.get('top_db',
                                                                   None),
                                                 freeze_parameters=True)
        # Spec augmenter
        self.spec_augmenter = _SpecAugmentation(time_drop_width=64,
                                                time_stripes_num=2,
                                                freq_drop_width=8*multiplier,
                                                freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64*multiplier)

        self.conv_block1 = _ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = _ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = _ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = _ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = _ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = _ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        init_layer(self.pre_conv0)
        init_bn(self.pre_bn0)
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, batch, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        # Wavegram
        a1 = F.relu_(self.pre_bn0(self.pre_conv0(batch[:, None, :])))
        a1 = self.pre_block1(a1, pool_size=4)
        a1 = self.pre_block2(a1, pool_size=4)
        a1 = self.pre_block3(a1, pool_size=4)
        a1 = a1.reshape((a1.shape[0], -1, 32*self.multiplier,
                        a1.shape[-1])).transpose(2, 3)
        a1 = self.pre_block4(a1, pool_size=(2, 1))

        # Log mel spectrogram
        x = self.spectrogram_extractor(
            batch)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = mixup(x, mixup_lambda)
            a1 = mixup(a1, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')

        # Concatenate Wavegram and Log mel spectrogram along the channel dimension
        x = torch.cat((x, a1), dim=1)

        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        return clipwise_output, embedding


class ResNet22(nn.Module):
    def __init__(self, *, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num, **kwargs):
        """

        Args:
            sample_rate:
            window_size:
            hop_size:
            mel_bins:
            fmin:
            fmax:
            classes_num:
            **kwargs: 'window', 'center', 'pad_mode' for Spectrogram,
                'ref', 'amin', 'top_db' for LogmelFilterbank
        """

        super().__init__()

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size,
                                                 hop_length=hop_size,
                                                 win_length=window_size,
                                                 window=kwargs.get('window',
                                                                   'hann'),
                                                 center=kwargs.get('center',
                                                                   True),
                                                 pad_mode=kwargs.get(
                                                         'pad_mode', 'reflect'),
                                                 freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate,
                                                 n_fft=window_size,
                                                 n_mels=mel_bins, fmin=fmin,
                                                 fmax=fmax,
                                                 ref=kwargs.get('ref', 1.0),
                                                 amin=kwargs.get('amin', 1e-10),
                                                 top_db=kwargs.get('top_db',
                                                                   None),
                                                 freeze_parameters=True)
        # Spec augmenter
        self.spec_augmenter = _SpecAugmentation(time_drop_width=64,
                                                time_stripes_num=2,
                                                freq_drop_width=8,
                                                freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = _ConvBlock(in_channels=1, out_channels=64)
        # self.conv_block2 = ConvBlock(in_channels=64, out_channels=64)

        self.resnet = _ResNet(block=_ResnetBasicBlock, layers=[2, 2, 2, 2],
                              zero_init_residual=True)

        self.conv_block_after1 = _ConvBlock(in_channels=512, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, batch, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(
            batch)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.resnet(x)
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.conv_block_after1(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        return clipwise_output, embedding


class ResNet38(nn.Module):
    def __init__(self, *, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num, **kwargs):
        """

        Args:
            sample_rate:
            window_size:
            hop_size:
            mel_bins:
            fmin:
            fmax:
            classes_num:
            **kwargs: 'window', 'center', 'pad_mode' for Spectrogram,
                'ref', 'amin', 'top_db' for LogmelFilterbank
        """

        super().__init__()

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size,
                                                 hop_length=hop_size,
                                                 win_length=window_size,
                                                 window=kwargs.get('window',
                                                                   'hann'),
                                                 center=kwargs.get('center',
                                                                   True),
                                                 pad_mode=kwargs.get(
                                                         'pad_mode', 'reflect'),
                                                 freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate,
                                                 n_fft=window_size,
                                                 n_mels=mel_bins, fmin=fmin,
                                                 fmax=fmax,
                                                 ref=kwargs.get('ref', 1.0),
                                                 amin=kwargs.get('amin', 1e-10),
                                                 top_db=kwargs.get('top_db',
                                                                   None),
                                                 freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = _SpecAugmentation(time_drop_width=64,
                                                time_stripes_num=2,
                                                freq_drop_width=8,
                                                freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = _ConvBlock(in_channels=1, out_channels=64)
        # self.conv_block2 = ConvBlock(in_channels=64, out_channels=64)

        self.resnet = _ResNet(block=_ResnetBasicBlock, layers=[3, 4, 6, 3],
                              zero_init_residual=True)

        self.conv_block_after1 = _ConvBlock(in_channels=512, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, batch, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(
            batch)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.resnet(x)
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.conv_block_after1(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        return clipwise_output, embedding


class ResNet54(nn.Module):
    def __init__(self, *, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num, **kwargs):
        """

        Args:
            sample_rate:
            window_size:
            hop_size:
            mel_bins:
            fmin:
            fmax:
            classes_num:
            **kwargs: 'window', 'center', 'pad_mode' for Spectrogram,
                'ref', 'amin', 'top_db' for LogmelFilterbank
        """

        super().__init__()

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size,
                                                 hop_length=hop_size,
                                                 win_length=window_size,
                                                 window=kwargs.get('window',
                                                                   'hann'),
                                                 center=kwargs.get('center',
                                                                   True),
                                                 pad_mode=kwargs.get(
                                                         'pad_mode', 'reflect'),
                                                 freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate,
                                                 n_fft=window_size,
                                                 n_mels=mel_bins, fmin=fmin,
                                                 fmax=fmax,
                                                 ref=kwargs.get('ref', 1.0),
                                                 amin=kwargs.get('amin', 1e-10),
                                                 top_db=kwargs.get('top_db',
                                                                   None),
                                                 freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = _SpecAugmentation(time_drop_width=64,
                                                time_stripes_num=2,
                                                freq_drop_width=8,
                                                freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = _ConvBlock(in_channels=1, out_channels=64)
        # self.conv_block2 = ConvBlock(in_channels=64, out_channels=64)

        self.resnet = _ResNet(block=_ResnetBottleneck, layers=[3, 4, 6, 3],
                              zero_init_residual=True)

        self.conv_block_after1 = _ConvBlock(in_channels=2048, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, batch, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(
            batch)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.resnet(x)
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.conv_block_after1(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        return clipwise_output, embedding


class Res1dNet31(nn.Module):
    def __init__(self, *, classes_num, **kwargs):
        super().__init__()

        self.conv0 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=11,
                               stride=5, padding=5, bias=False)
        self.bn0 = nn.BatchNorm1d(64)

        self.resnet = _ResNetWav1d(_ResnetBasicBlockWav1d,
                                   [2, 2, 2, 2, 2, 2, 2])

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        init_layer(self.conv0)
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, batch, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = batch[:, None, :]

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = mixup(x, mixup_lambda)

        x = self.bn0(self.conv0(x))
        x = self.resnet(x)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        return clipwise_output, embedding


class Res1dNet51(nn.Module):
    def __init__(self, *, classes_num, **kwargs):
        super().__init__()

        self.conv0 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=11,
                               stride=5, padding=5, bias=False)
        self.bn0 = nn.BatchNorm1d(64)

        self.resnet = _ResNetWav1d(_ResnetBasicBlockWav1d,
                                   [2, 3, 4, 6, 4, 3, 2])

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        init_layer(self.conv0)
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, batch, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = batch[:, None, :]

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = mixup(x, mixup_lambda)

        x = self.bn0(self.conv0(x))
        x = self.resnet(x)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        return clipwise_output, embedding


class MobileNetV1(nn.Module):
    def __init__(self, *, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num, **kwargs):
        """

        Args:
            sample_rate:
            window_size:
            hop_size:
            mel_bins:
            fmin:
            fmax:
            classes_num:
            **kwargs: 'window', 'center', 'pad_mode' for Spectrogram,
                'ref', 'amin', 'top_db' for LogmelFilterbank
        """

        super().__init__()

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size,
                                                 hop_length=hop_size,
                                                 win_length=window_size,
                                                 window=kwargs.get('window',
                                                                   'hann'),
                                                 center=kwargs.get('center',
                                                                   True),
                                                 pad_mode=kwargs.get(
                                                         'pad_mode', 'reflect'),
                                                 freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate,
                                                 n_fft=window_size,
                                                 n_mels=mel_bins, fmin=fmin,
                                                 fmax=fmax,
                                                 ref=kwargs.get('ref', 1.0),
                                                 amin=kwargs.get('amin', 1e-10),
                                                 top_db=kwargs.get('top_db',
                                                                   None),
                                                 freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = _SpecAugmentation(time_drop_width=64,
                                                time_stripes_num=2,
                                                freq_drop_width=8,
                                                freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        def conv_bn(inp, oup, stride):
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

        def conv_dw(inp, oup, stride):
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

        self.features = nn.Sequential(
                conv_bn(1, 32, 2),
                conv_dw(32, 64, 1),
                conv_dw(64, 128, 2),
                conv_dw(128, 128, 1),
                conv_dw(128, 256, 2),
                conv_dw(256, 256, 1),
                conv_dw(256, 512, 2),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 1024, 2),
                conv_dw(1024, 1024, 1))

        self.fc1 = nn.Linear(1024, 1024, bias=True)
        self.fc_audioset = nn.Linear(1024, classes_num, bias=True)

        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, batch, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(
            batch)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = mixup(x, mixup_lambda)

        x = self.features(x)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        return clipwise_output, embedding


class MobileNetV2(nn.Module):
    def __init__(self, *, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num, **kwargs):
        """

        Args:
            sample_rate:
            window_size:
            hop_size:
            mel_bins:
            fmin:
            fmax:
            classes_num:
            **kwargs: 'window', 'center', 'pad_mode' for Spectrogram,
                'ref', 'amin', 'top_db' for LogmelFilterbank
        """

        super().__init__()

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size,
                                                 hop_length=hop_size,
                                                 win_length=window_size,
                                                 window=kwargs.get('window',
                                                                   'hann'),
                                                 center=kwargs.get('center',
                                                                   True),
                                                 pad_mode=kwargs.get(
                                                         'pad_mode', 'reflect'),
                                                 freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate,
                                                 n_fft=window_size,
                                                 n_mels=mel_bins, fmin=fmin,
                                                 fmax=fmax,
                                                 ref=kwargs.get('ref', 1.0),
                                                 amin=kwargs.get('amin', 1e-10),
                                                 top_db=kwargs.get('top_db',
                                                                   None),
                                                 freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = _SpecAugmentation(time_drop_width=64,
                                                time_stripes_num=2,
                                                freq_drop_width=8,
                                                freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        width_mult = 1.
        block = _InvertedResidual
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 2],
            [6, 160, 3, 1],
            [6, 320, 1, 1],
        ]

        def conv_bn(inp, oup, stride):
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

        def conv_1x1_bn(inp, oup):
            _layers = nn.Sequential(
                    nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                    nn.ReLU6(inplace=True)
            )
            init_layer(_layers[0])
            init_bn(_layers[1])
            return _layers

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(
            last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(1, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(
                        block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(
                        block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        self.fc1 = nn.Linear(1280, 1024, bias=True)
        self.fc_audioset = nn.Linear(1024, classes_num, bias=True)

        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, batch, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(
            batch)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = mixup(x, mixup_lambda)

        x = self.features(x)

        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        return clipwise_output, embedding


class LeeNet11(nn.Module):
    def __init__(self, *, classes_num, **kwargs):
        super().__init__()

        self.conv_block1 = _LeeNetConvBlock(1, 64, 3, 3)
        self.conv_block2 = _LeeNetConvBlock(64, 64, 3, 1)
        self.conv_block3 = _LeeNetConvBlock(64, 64, 3, 1)
        self.conv_block4 = _LeeNetConvBlock(64, 128, 3, 1)
        self.conv_block5 = _LeeNetConvBlock(128, 128, 3, 1)
        self.conv_block6 = _LeeNetConvBlock(128, 128, 3, 1)
        self.conv_block7 = _LeeNetConvBlock(128, 128, 3, 1)
        self.conv_block8 = _LeeNetConvBlock(128, 128, 3, 1)
        self.conv_block9 = _LeeNetConvBlock(128, 256, 3, 1)

        self.fc1 = nn.Linear(256, 512, bias=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)

        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, batch, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = batch[:, None, :]

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = mixup(x, mixup_lambda)

        x = self.conv_block1(x)
        x = self.conv_block2(x, pool_size=3)
        x = self.conv_block3(x, pool_size=3)
        x = self.conv_block4(x, pool_size=3)
        x = self.conv_block5(x, pool_size=3)
        x = self.conv_block6(x, pool_size=3)
        x = self.conv_block7(x, pool_size=3)
        x = self.conv_block8(x, pool_size=3)
        x = self.conv_block9(x, pool_size=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        return clipwise_output, embedding


class LeeNet24(nn.Module):
    def __init__(self, *, classes_num, **kwargs):
        super().__init__()

        self.conv_block1 = _LeeNetConvBlock2(1, 64, 3, 3)
        self.conv_block2 = _LeeNetConvBlock2(64, 96, 3, 1)
        self.conv_block3 = _LeeNetConvBlock2(96, 128, 3, 1)
        self.conv_block4 = _LeeNetConvBlock2(128, 128, 3, 1)
        self.conv_block5 = _LeeNetConvBlock2(128, 256, 3, 1)
        self.conv_block6 = _LeeNetConvBlock2(256, 256, 3, 1)
        self.conv_block7 = _LeeNetConvBlock2(256, 512, 3, 1)
        self.conv_block8 = _LeeNetConvBlock2(512, 512, 3, 1)
        self.conv_block9 = _LeeNetConvBlock2(512, 1024, 3, 1)

        self.fc1 = nn.Linear(1024, 1024, bias=True)
        self.fc_audioset = nn.Linear(1024, classes_num, bias=True)

        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, batch, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = batch[:, None, :]

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = mixup(x, mixup_lambda)

        x = self.conv_block1(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv_block2(x, pool_size=3)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv_block3(x, pool_size=3)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv_block4(x, pool_size=3)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv_block5(x, pool_size=3)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv_block6(x, pool_size=3)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv_block7(x, pool_size=3)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv_block8(x, pool_size=3)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv_block9(x, pool_size=1)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        return clipwise_output, embedding


class DaiNet19(nn.Module):
    def __init__(self, *, classes_num, **kwargs):
        super().__init__()

        self.conv0 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=80,
                               stride=4, padding=0, bias=False)
        self.bn0 = nn.BatchNorm1d(64)
        self.conv_block1 = _DaiNetResBlock(64, 64, 3)
        self.conv_block2 = _DaiNetResBlock(64, 128, 3)
        self.conv_block3 = _DaiNetResBlock(128, 256, 3)
        self.conv_block4 = _DaiNetResBlock(256, 512, 3)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)

        init_layer(self.conv0)
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, batch, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = batch[:, None, :]

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = mixup(x, mixup_lambda)

        x = self.bn0(self.conv0(x))
        x = self.conv_block1(x)
        x = F.max_pool1d(x, kernel_size=4)
        x = self.conv_block2(x)
        x = F.max_pool1d(x, kernel_size=4)
        x = self.conv_block3(x)
        x = F.max_pool1d(x, kernel_size=4)
        x = self.conv_block4(x)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        return clipwise_output, embedding


class TransferModel(nn.Module):
    def __init__(self, model, classes_num_new, freeze_base=True):

        """Classifier for a new task using pretrained model as a submodule. """
        super().__init__()

        self.base = model

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(2048, classes_num_new, bias=True)

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        init_layer(self.fc_transfer)

    def forward(self, data, mixup_lambda=None):
        """Data: (batch_size, data_length)
        """
        _, embedding = self.base(data, mixup_lambda)

        clipwise_output = torch.log_softmax(self.fc_transfer(embedding), dim=-1)

        return clipwise_output, embedding
