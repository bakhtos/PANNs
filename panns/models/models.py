import logging
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

from panns.data.mixup import mixup
from .blocks import *

__all__ = ['Cnn6',
           'Cnn10',
           'Cnn14',
           'Cnn14DecisionLevelMax',
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
           ]

ModelOutput = namedtuple('ModelOutput', ['clipwise_output',
                                         'segmentwise_output',
                                         'framewise_output', 'embedding'])


class Cnn6(nn.Module):
    def __init__(self, *, sample_rate, win_length, hop_length, n_mels, f_min,
                 f_max, classes_num, spec_aug=True, mixup_time=False,
                 mixup_freq=True, dropout=True, wavegram=False,
                 spectrogram=True, decision_level=None, **kwargs):
        """

        Args:
            sample_rate: Sample rate of audio signal.
            win_length: Window size for MelSpectrogram.
            hop_length: Length of hop between STFT windows in MelSpectrogram.
            n_mels: Number of mel filterbanks in MelSpectrogram.
            f_min: Minimum frequency for MelSpectrogram.
            f_max: Maximum frequency for MelSpectrogram.
            classes_num: Amount of classes used for training.
            spec_aug: If True, apply Spectrogram Augmentation (default True).
            mixup_time: If True, apply mixup in time domain (default False).
            mixup_freq: If True, apply mixup in frequency domain (default True).
            dropout: If True, apply dropout in dropout layers during training.
            wavegram: If True, use wavegram features (default False).
            spectrogram: If True, use spectrogram features (default True).
            decision_level: If not None, create framewise output using one of
                the options 'max', 'avg', 'att' (default None).
            **kwargs: 'window_fn', 'center', 'pad_mode' for MelSpectrogram
                      (defaults 'hann', 'center', 'reflect'),
                      'top_db' for AmplitudeToDB (default None),
                      'num_features' for BatchNorm2d (default 64),
                      'embedding_size' for the amount of neurons connecting the
                       last two fully connected layers (default 512)
        """

        super().__init__()

        logging.info("Initializing model Cnn6.")
        logging.info(f"Cnn6 - parameter 'classes_num' is: {classes_num}.")
        self.wavegram = wavegram
        logging.info(f"Cnn6 - parameter 'wavegram' is: {wavegram}.")
        self.spectrogram = spectrogram
        logging.info(f"Cnn6 - parameter 'spectrogram' is: {spectrogram}.")
        self.spec_aug = spec_aug
        logging.info(f"Cnn6 - parameter 'spec_aug' is: {spec_aug}.")
        if spectrogram is False and spec_aug is True:
            logging.warning("Cnn6 - 'spectrogram is False but spec_aug is "
                            "'True' - spec_aug is ignored.")
        self.mixup_time = mixup_time
        logging.info(f"Cnn6 - parameter 'mixup_time' is: {mixup_time}")
        self.mixup_freq = mixup_freq
        logging.info(f"Cnn6 - parameter 'mixup_freq' is: {mixup_freq}.")
        self.dropout = dropout
        logging.info(f"Cnn6 - parameter 'dropout' is: {dropout}.")
        self.decision_level = decision_level
        logging.info(f"Cnn6 - parameter 'decision_level' is: "
                     f"{decision_level}.")
        self.interpolate_ratio = 32
        window_fn = kwargs.get('window_fn', torch.hann_window)
        center = kwargs.get('center', True)
        pad_mode = kwargs.get('pad_mode', 'reflect')
        top_db = kwargs.get('top_db', None)
        num_features = kwargs.get('num_features', 64)
        logging.info(f"Cnn6 - parameter 'num_features' is: {num_features}.")
        embedding_size = kwargs.get('embedding_size', 512)
        logging.info(f"Cnn6 - parameter 'embedding_size' is: "
                     f"{embedding_size}.")

        if self.wavegram:
            self.pre_conv0 = nn.Conv1d(in_channels=1, out_channels=64,
                                       kernel_size=11, stride=5, padding=5,
                                       bias=False)
            self.pre_bn0 = nn.BatchNorm1d(64)
            self.pre_block1 = _ConvPreWavBlock(64, 64)
            self.pre_block2 = _ConvPreWavBlock(64, 128)
            self.pre_block3 = _ConvPreWavBlock(128, 128)
            self.pre_block4 = _ConvBlock(in_channels=4, out_channels=64)
            init_layer(self.pre_conv0)
            init_bn(self.pre_bn0)

        if self.spectrogram:
            # Spectrogram extractor
            logging.info(f"Cnn6 Spectrogram - parameter 'sample_rate' is: "
                         f"{sample_rate}.")
            logging.info(f"Cnn6 Spectrogram - parameter 'win_length' is: "
                         f"{win_length}.")
            logging.info(f"Cnn6 Spectrogram - parameter 'hop_length' is: "
                         f"{hop_length}.")
            logging.info(f"Cnn6 Spectrogram - parameter 'f_min' is: "
                         f"{f_min}.")
            logging.info(f"Cnn6 Spectrogram - parameter 'f_max' is: "
                         f"{f_max}.")
            logging.info(f"Cnn6 Spectrogram - parameter 'n_mels' is: "
                         f"{n_mels}.")
            logging.info(f"Cnn6 Spectrogram - parameter 'center' is:"
                         f" {center}.")
            logging.info(f"Cnn6 Spectrogram - parameter 'pad_mode' is:"
                         f" {pad_mode}.")
            logging.info(f"Cnn6 Spectrogram - parameter 'top_db' is: {top_db}.")
            self.mel_spectrogram = MelSpectrogram(sample_rate=sample_rate,
                                                  n_fft=win_length,
                                                  win_length=win_length,
                                                  hop_length=hop_length,
                                                  f_min=f_min, f_max=f_max,
                                                  n_mels=n_mels,
                                                  window_fn=window_fn,
                                                  power=2, onesided=True,
                                                  center=center,
                                                  pad_mode=pad_mode)
            self.amplitude_to_db = AmplitudeToDB(stype="power", top_db=top_db)

            # Spec augmenter
            if spec_aug:
                self.spec_aug_time = _SpecAugmentation(mask_param=64,
                                                       stripes_num=2,
                                                       axis=2)
                self.spec_aug_freq = _SpecAugmentation(mask_param=8,
                                                       stripes_num=2,
                                                       axis=3)

            self.bn0 = nn.BatchNorm2d(num_features)
            init_bn(self.bn0)

        self.conv_block1 = _ConvBlock5x5(in_channels=1, out_channels=64)
        self.conv_block2 = _ConvBlock5x5(in_channels=64, out_channels=128)
        self.conv_block3 = _ConvBlock5x5(in_channels=128, out_channels=256)
        self.conv_block4 = _ConvBlock5x5(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, embedding_size, bias=True)
        init_layer(self.fc1)

        if self.decision_level == 'att':
            self.audioset_layer = _AttBlock(embedding_size, classes_num,
                                            activation='sigmoid')
        else:
            self.audioset_layer = nn.Linear(embedding_size, classes_num,
                                            bias=True)
            init_layer(self.audioset_layer)

    def forward(self, batch, mixup_lambda=None):
        """

        Args:
            batch: Input tensor of shape (batch_size, time).
            mixup_lambda: If not None, apply mixup with given coefficients
                          (default None).

        Returns: clipwise_output, segmentwise_output, framewise_output,
                 embedding
        """

        # Mixup in time domain
        if self.mixup_time and self.training and mixup_lambda is not None:
            batch = mixup(batch, mixup_lambda)

        x = None
        wave = None

        if self.wavegram:
            # Wavegram
            wave = F.relu_(self.pre_bn0(self.pre_conv0(batch[:, None, :])))
            wave = self.pre_block1(wave, pool_size=4)
            wave = self.pre_block2(wave, pool_size=4)
            wave = self.pre_block3(wave, pool_size=4)
            wave = wave.reshape((wave.shape[0], -1, 32,
                                 wave.shape[-1])).transpose(2, 3)
            wave = self.pre_block4(wave, pool_size=(2, 1))

            if self.mixup_freq and self.training and mixup_lambda is not None:
                wave = mixup(wave, mixup_lambda)

        if self.spectrogram:
            x = batch[:, None, :]        # (batch_size, 1, time)
            x = self.mel_spectrogram(x)  # (batch_size, 1, n_mels, time)
            x = self.amplitude_to_db(x)  # (batch_size, 1, n_mels, time)

            x = torch.transpose(x, 1, 2)        # (batch_size, n_mels, 1, time)
            x = torch.transpose(x, 2, 3)        # (batch_size, n_mels, time, 1)

            frames_num = x.shape[2]

            x = self.bn0(x)

            x = torch.transpose(x, 1, 3)        # (batch_size, 1, time, n_mels)

            if self.training:
                if self.spec_aug:
                    x = self.spec_aug_time(x)
                    x = self.spec_aug_freq(x)
                # Mixup on spectrogram
                if self.mixup_freq and mixup_lambda is not None:
                    x = mixup(x, mixup_lambda)

            x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')

        if self.wavegram and self.spectrogram:
            x = torch.cat((x, wave), dim=1)
        elif self.wavegram:
            x = wave

        x = F.dropout(x, p=0.2, training=(self.training and
                                          self.dropout))
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=(self.training and
                                          self.dropout))
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=(self.training and
                                          self.dropout))
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=(self.training and
                                          self.dropout))
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=(self.training and
                                          self.dropout))
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=(self.training and
                                          self.dropout))
        x = torch.mean(x, dim=3)

        clipwise_output = None
        segmentwise_output = None
        framewise_output = None
        if self.decision_level is None:  # Weak labels
            (x1, _) = torch.max(x, dim=2)
            x2 = torch.mean(x, dim=2)
            x = x1 + x2
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu_(self.fc1(x))
            embedding = F.dropout(x, p=0.5, training=self.training)
            clipwise_output = torch.sigmoid(self.audioset_layer(x))
        else:  # Strong labels
            x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
            x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
            x = x1 + x2
            x = F.dropout(x, p=0.5, training=self.training)
            x = x.transpose(1, 2)
            x = F.relu_(self.fc1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            embedding = x

            if self.decision_level == 'att':
                x = x.transpose(1, 2)
                (clipwise_output, _, segmentwise_output) = self.audioset_layer(x)
                segmentwise_output = segmentwise_output.transpose(1, 2)
            else:
                segmentwise_output = torch.sigmoid(self.audioset_layer(x))
                if self.decision_level == 'max':
                    (clipwise_output, _) = torch.max(segmentwise_output, dim=1)
                elif self.decision_level == 'avg':
                    clipwise_output = torch.mean(segmentwise_output, dim=1)

            # Get framewise output
            framewise_output = _interpolate(segmentwise_output,
                                            self.interpolate_ratio)
            framewise_output = _pad_framewise_output(framewise_output,
                                                     frames_num)

        return ModelOutput(clipwise_output, segmentwise_output,
                           framewise_output, embedding)


class Cnn10(nn.Module):
    def __init__(self, *, sample_rate, win_length, hop_length, n_mels, f_min,
                 f_max, classes_num, spec_aug=True, mixup_time=False,
                 mixup_freq=True, dropout=True, wavegram=False,
                 spectrogram=True, decision_level=None, **kwargs):
        """

        Args:
            sample_rate: Sample rate of audio signal.
            win_length: Window size for MelSpectrogram.
            hop_length: Length of hop between STFT windows in MelSpectrogram.
            n_mels: Number of mel filterbanks in MelSpectrogram.
            f_min: Minimum frequency for MelSpectrogram.
            f_max: Maximum frequency for MelSpectrogram.
            classes_num: Amount of classes used for training.
            spec_aug: If True, apply Spectrogram Augmentation (default True).
            mixup_time: If True, apply mixup in time domain (default False).
            mixup_freq: If True, apply mixup in frequency domain (default True).
            dropout: If True, apply dropout in dropout layers during training.
            wavegram: If True, use wavegram features (default False).
            spectrogram: If True, use spectrogram features (default True).
            decision_level: If not None, create framewise output using one of
                the options 'max', 'avg', 'att' (default None).
            **kwargs: 'window_fn', 'center', 'pad_mode' for MelSpectrogram
                      (defaults 'hann', 'center', 'reflect'),
                      'top_db' for AmplitudeToDB (default None),
                      'num_features' for BatchNorm2d (default 64),
                      'embedding_size' for the amount of neurons connecting the
                       last two fully connected layers (default 512)
        """

        super().__init__()

        logging.info("Initializing model Cnn10.")
        logging.info(f"Cnn10 - parameter 'classes_num' is: {classes_num}.")
        self.wavegram = wavegram
        logging.info(f"Cnn10 - parameter 'wavegram' is: {wavegram}.")
        self.spectrogram = spectrogram
        logging.info(f"Cnn10 - parameter 'spectrogram' is: {spectrogram}.")
        self.spec_aug = spec_aug
        logging.info(f"Cnn10 - parameter 'spec_aug' is: {spec_aug}.")
        if spectrogram is False and spec_aug is True:
            logging.warning("Cnn10 - 'spectrogram is False but spec_aug is "
                            "'True' - spec_aug is ignored.")
        self.mixup_time = mixup_time
        logging.info(f"Cnn10 - parameter 'mixup_time' is: {mixup_time}")
        self.mixup_freq = mixup_freq
        logging.info(f"Cnn10 - parameter 'mixup_freq' is: {mixup_freq}.")
        self.dropout = dropout
        logging.info(f"Cnn10 - parameter 'dropout' is: {dropout}.")
        self.decision_level = decision_level
        logging.info(f"Cnn10 - parameter 'decision_level' is: "
                     f"{decision_level}.")
        self.interpolate_ratio = 32
        window_fn = kwargs.get('window_fn', torch.hann_window)
        center = kwargs.get('center', True)
        pad_mode = kwargs.get('pad_mode', 'reflect')
        top_db = kwargs.get('top_db', None)
        num_features = kwargs.get('num_features', 64)
        logging.info(f"Cnn10 - parameter 'num_features' is: {num_features}.")
        embedding_size = kwargs.get('embedding_size', 512)
        logging.info(f"Cnn10 - parameter 'embedding_size' is: "
                     f"{embedding_size}.")

        if self.wavegram:
            self.pre_conv0 = nn.Conv1d(in_channels=1, out_channels=64,
                                       kernel_size=11, stride=5, padding=5,
                                       bias=False)
            self.pre_bn0 = nn.BatchNorm1d(64)
            self.pre_block1 = _ConvPreWavBlock(64, 64)
            self.pre_block2 = _ConvPreWavBlock(64, 128)
            self.pre_block3 = _ConvPreWavBlock(128, 128)
            self.pre_block4 = _ConvBlock(in_channels=4, out_channels=64)
            init_layer(self.pre_conv0)
            init_bn(self.pre_bn0)

        if self.spectrogram:
            # Spectrogram extractor
            logging.info(f"Cnn10 Spectrogram - parameter 'sample_rate' is: "
                         f"{sample_rate}.")
            logging.info(f"Cnn10 Spectrogram - parameter 'win_length' is: "
                         f"{win_length}.")
            logging.info(f"Cnn10 Spectrogram - parameter 'hop_length' is: "
                         f"{hop_length}.")
            logging.info(f"Cnn10 Spectrogram - parameter 'f_min' is: "
                         f"{f_min}.")
            logging.info(f"Cnn10 Spectrogram - parameter 'f_max' is: "
                         f"{f_max}.")
            logging.info(f"Cnn10 Spectrogram - parameter 'n_mels' is: "
                         f"{n_mels}.")
            logging.info(f"Cnn10 Spectrogram - parameter 'center' is:"
                         f" {center}.")
            logging.info(f"Cnn10 Spectrogram - parameter 'pad_mode' is:"
                         f" {pad_mode}.")
            logging.info(f"Cnn10 Spectrogram - parameter 'top_db' is:"
                         f" {top_db}.")
            self.mel_spectrogram = MelSpectrogram(sample_rate=sample_rate,
                                                  n_fft=win_length,
                                                  win_length=win_length,
                                                  hop_length=hop_length,
                                                  f_min=f_min, f_max=f_max,
                                                  n_mels=n_mels,
                                                  window_fn=window_fn,
                                                  power=2, onesided=True,
                                                  center=center,
                                                  pad_mode=pad_mode)
            self.amplitude_to_db = AmplitudeToDB(stype="power", top_db=top_db)

            # Spec augmenter
            if spec_aug:
                self.spec_aug_time = _SpecAugmentation(mask_param=64,
                                                       stripes_num=2,
                                                       axis=2)
                self.spec_aug_freq = _SpecAugmentation(mask_param=8,
                                                       stripes_num=2,
                                                       axis=3)

            self.bn0 = nn.BatchNorm2d(num_features)
            init_bn(self.bn0)

        self.conv_block1 = _ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = _ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = _ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = _ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, embedding_size, bias=True)
        init_layer(self.fc1)

        if self.decision_level == 'att':
            self.audioset_layer = _AttBlock(embedding_size, classes_num,
                                            activation='sigmoid')
        else:
            self.audioset_layer = nn.Linear(embedding_size, classes_num,
                                            bias=True)
            init_layer(self.audioset_layer)

    def forward(self, batch, mixup_lambda=None):
        """

        Args:
            batch: Input tensor of shape (batch_size, time).
            mixup_lambda: If not None, apply mixup with given coefficients
                          (default None).

        Returns: clipwise_output, segmentwise_output, framewise_output,
                 embedding
        """

        # Mixup in time domain
        if self.mixup_time and self.training and mixup_lambda is not None:
            batch = mixup(batch, mixup_lambda)

        x = None
        wave = None

        if self.wavegram:
            # Wavegram
            wave = F.relu_(self.pre_bn0(self.pre_conv0(batch[:, None, :])))
            wave = self.pre_block1(wave, pool_size=4)
            wave = self.pre_block2(wave, pool_size=4)
            wave = self.pre_block3(wave, pool_size=4)
            wave = wave.reshape((wave.shape[0], -1, 32,
                                 wave.shape[-1])).transpose(2, 3)
            wave = self.pre_block4(wave, pool_size=(2, 1))

            if self.mixup_freq and self.training and mixup_lambda is not None:
                wave = mixup(wave, mixup_lambda)

        if self.spectrogram:
            x = batch[:, None, :]        # (batch_size, 1, time)
            x = self.mel_spectrogram(x)  # (batch_size, 1, n_mels, time)
            x = self.amplitude_to_db(x)  # (batch_size, 1, n_mels, time)

            x = torch.transpose(x, 1, 2)        # (batch_size, n_mels, 1, time)
            x = torch.transpose(x, 2, 3)        # (batch_size, n_mels, time, 1)

            frames_num = x.shape[2]

            x = self.bn0(x)

            x = torch.transpose(x, 1, 3)        # (batch_size, 1, time, n_mels)

            if self.training:
                if self.spec_aug:
                    x = self.spec_aug_time(x)
                    x = self.spec_aug_freq(x)
                # Mixup on spectrogram
                if self.mixup_freq and mixup_lambda is not None:
                    x = mixup(x, mixup_lambda)

            x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')

        if self.wavegram and self.spectrogram:
            x = torch.cat((x, wave), dim=1)
        elif self.wavegram:
            x = wave

        x = F.dropout(x, p=0.2, training=(self.training and
                                          self.dropout))
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=(self.training and
                                          self.dropout))
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=(self.training and
                                          self.dropout))
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=(self.training and
                                          self.dropout))
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=(self.training and
                                          self.dropout))
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=(self.training and
                                          self.dropout))
        x = torch.mean(x, dim=3)

        clipwise_output = None
        segmentwise_output = None
        framewise_output = None
        if self.decision_level is None:  # Weak labels
            (x1, _) = torch.max(x, dim=2)
            x2 = torch.mean(x, dim=2)
            x = x1 + x2
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu_(self.fc1(x))
            embedding = F.dropout(x, p=0.5, training=self.training)
            clipwise_output = torch.sigmoid(self.audioset_layer(x))
        else:  # Strong labels
            x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
            x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
            x = x1 + x2
            x = F.dropout(x, p=0.5, training=self.training)
            x = x.transpose(1, 2)
            x = F.relu_(self.fc1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            embedding = x

            if self.decision_level == 'att':
                x = x.transpose(1, 2)
                (clipwise_output, _, segmentwise_output) = self.audioset_layer(x)
                segmentwise_output = segmentwise_output.transpose(1, 2)
            else:
                segmentwise_output = torch.sigmoid(self.audioset_layer(x))
                if self.decision_level == 'max':
                    (clipwise_output, _) = torch.max(segmentwise_output, dim=1)
                elif self.decision_level == 'avg':
                    clipwise_output = torch.mean(segmentwise_output, dim=1)

            # Get framewise output
            framewise_output = _interpolate(segmentwise_output,
                                            self.interpolate_ratio)
            framewise_output = _pad_framewise_output(framewise_output,
                                                     frames_num)

        return ModelOutput(clipwise_output, segmentwise_output,
                           framewise_output, embedding)


class Cnn14(nn.Module):
    def __init__(self, *, sample_rate, win_length, hop_length, n_mels, f_min,
                 f_max, classes_num, spec_aug=True, mixup_time=False,
                 mixup_freq=True, dropout=True, wavegram=False,
                 spectrogram=True, decision_level=None, **kwargs):
        """

        Args:
            sample_rate: Sample rate of audio signal.
            win_length: Window size for MelSpectrogram.
            hop_length: Length of hop between STFT windows in MelSpectrogram.
            n_mels: Number of mel filterbanks in MelSpectrogram.
            f_min: Minimum frequency for MelSpectrogram.
            f_max: Maximum frequency for MelSpectrogram.
            classes_num: Amount of classes used for training.
            spec_aug: If True, apply Spectrogram Augmentation (default True).
            mixup_time: If True, apply mixup in time domain (default False).
            mixup_freq: If True, apply mixup in frequency domain (default True).
            dropout: If True, apply dropout in dropout layers during training.
            wavegram: If True, use wavegram features (default False).
            spectrogram: If True, use spectrogram features (default True).
            decision_level: If not None, create framewise output using one of
                the options 'max', 'avg', 'att' (default None).
            **kwargs: 'window_fn', 'center', 'pad_mode' for MelSpectrogram
                      (defaults 'hann', 'center', 'reflect'),
                      'top_db' for AmplitudeToDB (default None),
                      'num_features' for BatchNorm2d (default 64),
                      'embedding_size' for the amount of neurons connecting the
                       last two fully connected layers (default 2048)
        """

        super().__init__()

        logging.info("Initializing model Cnn14.")
        logging.info(f"Cnn14 - parameter 'classes_num' is: {classes_num}.")
        self.wavegram = wavegram
        logging.info(f"Cnn14 - parameter 'wavegram' is: {wavegram}.")
        self.spectrogram = spectrogram
        logging.info(f"Cnn14 - parameter 'spectrogram' is: {spectrogram}.")
        self.spec_aug = spec_aug
        logging.info(f"Cnn14 - parameter 'spec_aug' is: {spec_aug}.")
        if spectrogram is False and spec_aug is True:
            logging.warning("Cnn14 - 'spectrogram is False but spec_aug is "
                            "'True' - spec_aug is ignored.")
        self.mixup_time = mixup_time
        logging.info(f"Cnn14 - parameter 'mixup_time' is: {mixup_time}")
        self.mixup_freq = mixup_freq
        logging.info(f"Cnn14 - parameter 'mixup_freq' is: {mixup_freq}.")
        self.dropout = dropout
        logging.info(f"Cnn14 - parameter 'dropout' is: {dropout}.")
        self.decision_level = decision_level
        logging.info(f"Cnn14 - parameter 'decision_level' is: "
                     f"{decision_level}.")
        self.interpolate_ratio = 32
        window_fn = kwargs.get('window_fn', torch.hann_window)
        center = kwargs.get('center', True)
        pad_mode = kwargs.get('pad_mode', 'reflect')
        top_db = kwargs.get('top_db', None)
        num_features = kwargs.get('num_features', 64)
        logging.info(f"Cnn14 - parameter 'num_features' is: {num_features}.")
        embedding_size = kwargs.get('embedding_size', 2048)
        logging.info(f"Cnn14 - parameter 'embedding_size' is: "
                     f"{embedding_size}.")

        if self.wavegram:
            self.pre_conv0 = nn.Conv1d(in_channels=1, out_channels=64,
                                       kernel_size=11, stride=5, padding=5,
                                       bias=False)
            self.pre_bn0 = nn.BatchNorm1d(64)
            self.pre_block1 = _ConvPreWavBlock(64, 64)
            self.pre_block2 = _ConvPreWavBlock(64, 128)
            self.pre_block3 = _ConvPreWavBlock(128, 128)
            self.pre_block4 = _ConvBlock(in_channels=4, out_channels=64)
            init_layer(self.pre_conv0)
            init_bn(self.pre_bn0)

        if self.spectrogram:
            # Spectrogram extractor
            logging.info(f"Cnn14 Spectrogram - parameter 'sample_rate' is: "
                         f"{sample_rate}.")
            logging.info(f"Cnn14 Spectrogram - parameter 'win_length' is: "
                         f"{win_length}.")
            logging.info(f"Cnn14 Spectrogram - parameter 'hop_length' is: "
                         f"{hop_length}.")
            logging.info(f"Cnn14 Spectrogram - parameter 'f_min' is: "
                         f"{f_min}.")
            logging.info(f"Cnn14 Spectrogram - parameter 'f_max' is: "
                         f"{f_max}.")
            logging.info(f"Cnn14 Spectrogram - parameter 'n_mels' is: "
                         f"{n_mels}.")
            logging.info(f"Cnn14 Spectrogram - parameter 'center' is:"
                         f" {center}.")
            logging.info(f"Cnn14 Spectrogram - parameter 'pad_mode' is:"
                         f" {pad_mode}.")
            logging.info(f"Cnn14 Spectrogram - parameter 'top_db' is:"
                         f" {top_db}.")
            self.mel_spectrogram = MelSpectrogram(sample_rate=sample_rate,
                                                  n_fft=win_length,
                                                  win_length=win_length,
                                                  hop_length=hop_length,
                                                  f_min=f_min, f_max=f_max,
                                                  n_mels=n_mels,
                                                  window_fn=window_fn,
                                                  power=2, onesided=True,
                                                  center=center,
                                                  pad_mode=pad_mode)
            self.amplitude_to_db = AmplitudeToDB(stype="power", top_db=top_db)

            # Spec augmenter
            if spec_aug:
                self.spec_aug_time = _SpecAugmentation(mask_param=64,
                                                       stripes_num=2,
                                                       axis=2)
                self.spec_aug_freq = _SpecAugmentation(mask_param=8,
                                                       stripes_num=2,
                                                       axis=3)

            self.bn0 = nn.BatchNorm2d(num_features)
            init_bn(self.bn0)

        self.conv_block1 = _ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = _ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = _ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = _ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = _ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = _ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, embedding_size, bias=True)
        init_layer(self.fc1)

        if self.decision_level == 'att':
            self.audioset_layer = _AttBlock(embedding_size, classes_num,
                                            activation='sigmoid')
        else:
            self.audioset_layer = nn.Linear(embedding_size, classes_num,
                                            bias=True)
            init_layer(self.audioset_layer)

    def forward(self, batch, mixup_lambda=None):
        """

        Args:
            batch: Input tensor of shape (batch_size, time).
            mixup_lambda: If not None, apply mixup with given coefficients
                          (default None).

        Returns: clipwise_output, segmentwise_output, framewise_output,
                 embedding
        """

        # Mixup in time domain
        if self.mixup_time and self.training and mixup_lambda is not None:
            batch = mixup(batch, mixup_lambda)

        x = None
        wave = None

        if self.wavegram:
            # Wavegram
            wave = F.relu_(self.pre_bn0(self.pre_conv0(batch[:, None, :])))
            wave = self.pre_block1(wave, pool_size=4)
            wave = self.pre_block2(wave, pool_size=4)
            wave = self.pre_block3(wave, pool_size=4)
            wave = wave.reshape((wave.shape[0], -1, 32,
                                 wave.shape[-1])).transpose(2, 3)
            wave = self.pre_block4(wave, pool_size=(2, 1))

            if self.mixup_freq and self.training and mixup_lambda is not None:
                wave = mixup(wave, mixup_lambda)

        if self.spectrogram:
            x = batch[:, None, :]        # (batch_size, 1, time)
            x = self.mel_spectrogram(x)  # (batch_size, 1, n_mels, time)
            x = self.amplitude_to_db(x)  # (batch_size, 1, n_mels, time)

            x = torch.transpose(x, 1, 2)        # (batch_size, n_mels, 1, time)
            x = torch.transpose(x, 2, 3)        # (batch_size, n_mels, time, 1)

            frames_num = x.shape[2]

            x = self.bn0(x)

            x = torch.transpose(x, 1, 3)        # (batch_size, 1, time, n_mels)

            if self.training:
                if self.spec_aug:
                    x = self.spec_aug_time(x)
                    x = self.spec_aug_freq(x)
                # Mixup on spectrogram
                if self.mixup_freq and mixup_lambda is not None:
                    x = mixup(x, mixup_lambda)

            x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')

        if self.wavegram and self.spectrogram:
            x = torch.cat((x, wave), dim=1)
        elif self.wavegram:
            x = wave

        x = F.dropout(x, p=0.2, training=(self.training and
                                          self.dropout))
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=(self.training and
                                          self.dropout))
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=(self.training and
                                          self.dropout))
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=(self.training and
                                          self.dropout))
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=(self.training and
                                          self.dropout))
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=(self.training and
                                          self.dropout))
        x = torch.mean(x, dim=3)

        clipwise_output = None
        segmentwise_output = None
        framewise_output = None
        if self.decision_level is None:  # Weak labels
            (x1, _) = torch.max(x, dim=2)
            x2 = torch.mean(x, dim=2)
            x = x1 + x2
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu_(self.fc1(x))
            embedding = F.dropout(x, p=0.5, training=self.training)
            clipwise_output = torch.sigmoid(self.audioset_layer(x))
        else:  # Strong labels
            x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
            x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
            x = x1 + x2
            x = F.dropout(x, p=0.5, training=self.training)
            x = x.transpose(1, 2)
            x = F.relu_(self.fc1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            embedding = x

            if self.decision_level == 'att':
                x = x.transpose(1, 2)
                (clipwise_output, _, segmentwise_output) = self.audioset_layer(x)
                segmentwise_output = segmentwise_output.transpose(1, 2)
            else:
                segmentwise_output = torch.sigmoid(self.audioset_layer(x))
                if self.decision_level == 'max':
                    (clipwise_output, _) = torch.max(segmentwise_output, dim=1)
                elif self.decision_level == 'avg':
                    clipwise_output = torch.mean(segmentwise_output, dim=1)

            # Get framewise output
            framewise_output = _interpolate(segmentwise_output,
                                            self.interpolate_ratio)
            framewise_output = _pad_framewise_output(framewise_output,
                                                     frames_num)

        return ModelOutput(clipwise_output, segmentwise_output,
                           framewise_output, embedding)


class Cnn14DecisionLevelMax(nn.Module):
    def __init__(self, *, sample_rate, win_length, hop_length, n_mels, f_min,
                 f_max, classes_num, spec_aug=True, mixup_time=False,
                 mixup_freq=True, dropout=True, **kwargs):
        """

        Args:
            sample_rate: Sample rate of audio signal.
            win_length: Window size for MelSpectrogram.
            hop_length: Length of hop between STFT windows in MelSpectrogram.
            n_mels: Number of mel filterbanks in MelSpectrogram.
            f_min: Minimum frequency for MelSpectrogram.
            f_max: Maximum frequency for MelSpectrogram.
            classes_num: Amount of classes used for training.
            spec_aug: If True, apply Spectrogram Augmentation (default True).
            mixup_time: If True, apply mixup in time domain (default False).
            mixup_freq: If True, apply mixup in frequency domain (default True).
            dropout: If True, apply dropout in dropout layers during training.
            **kwargs: 'window_fn', 'center', 'pad_mode' for MelSpectrogram
                      (defaults 'hann', 'center', 'reflect'),
                      'top_db' for AmplitudeToDB (default None),
                      'num_features' for BatchNorm2d (default 64),
                      'embedding_size' for the amount of neurons connecting the
                       last two fully connected layers (default 2048)
        """

        super().__init__()

        self.spec_aug = spec_aug
        self.mixup_time = mixup_time
        self.mixup_freq = mixup_freq
        self.dropout = dropout
        self.interpolate_ratio = 32
        window_fn = kwargs.get('window_fn', torch.hann_window)
        center = kwargs.get('center', True)
        pad_mode = kwargs.get('pad_mode', 'reflect')
        top_db = kwargs.get('top_db', None)
        num_features = kwargs.get('num_features', 64)
        embedding_size = kwargs.get('embedding_size', 2048)

        self.mel_spectrogram = MelSpectrogram(sample_rate=sample_rate,
                                              n_fft=win_length,
                                              win_length=win_length,
                                              hop_length=hop_length,
                                              f_min=f_min, f_max=f_max,
                                              n_mels=n_mels,
                                              window_fn=window_fn,
                                              power=2, onesided=True,
                                              center=center,
                                              pad_mode=pad_mode)
        self.amplitude_to_db = AmplitudeToDB(stype="power", top_db=top_db)

        # Spec augmenter
        if spec_aug:
            self.spec_aug_time = _SpecAugmentation(mask_param=64,
                                                   stripes_num=2,
                                                   axis=2)
            self.spec_aug_freq = _SpecAugmentation(mask_param=8,
                                                   stripes_num=2,
                                                   axis=3)

        self.bn0 = nn.BatchNorm2d(num_features)
        init_bn(self.bn0)

        self.conv_block1 = _ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = _ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = _ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = _ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = _ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = _ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, embedding_size, bias=True)
        init_layer(self.fc1)

        self.audioset_layer = nn.Linear(embedding_size, classes_num,
                                        bias=True)
        init_layer(self.audioset_layer)

    def forward(self, batch, mixup_lambda=None):
        """

        Args:
            batch: Input tensor of shape (batch_size, time).
            mixup_lambda: If not None, apply mixup with given coefficients
                          (default None).

        Returns: clipwise_output, segmentwise_output, framewise_output,
                 embedding
        """

        # Mixup in time domain
        if self.mixup_time and self.training and mixup_lambda is not None:
            batch = mixup(batch, mixup_lambda)

        x = batch[:, None, :]        # (batch_size, 1, time)
        x = self.mel_spectrogram(x)  # (batch_size, 1, n_mels, time)
        x = self.amplitude_to_db(x)  # (batch_size, 1, n_mels, time)

        x = torch.transpose(x, 1, 2)        # (batch_size, n_mels, 1, time)
        x = torch.transpose(x, 2, 3)        # (batch_size, n_mels, time, 1)

        frames_num = x.shape[2]

        x = self.bn0(x)

        x = torch.transpose(x, 1, 3)        # (batch_size, 1, time, n_mels)

        if self.training:
            if self.spec_aug:
                x = self.spec_aug_time(x)
                x = self.spec_aug_freq(x)
            # Mixup on spectrogram
            if self.mixup_freq and mixup_lambda is not None:
                x = mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')

        x = F.dropout(x, p=0.2, training=(self.training and
                                          self.dropout))
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=(self.training and
                                          self.dropout))
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=(self.training and
                                          self.dropout))
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=(self.training and
                                          self.dropout))
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=(self.training and
                                          self.dropout))
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=(self.training and
                                          self.dropout))
        x = torch.mean(x, dim=3)

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        embedding = x
        segmentwise_output = torch.sigmoid(self.audioset_layer(x))
        (clipwise_output, _) = torch.max(segmentwise_output, dim=1)

        # Get framewise output
        framewise_output = _interpolate(segmentwise_output,
                                        self.interpolate_ratio)
        framewise_output = _pad_framewise_output(framewise_output, frames_num)

        return ModelOutput(clipwise_output, segmentwise_output,
                           framewise_output, embedding)


class ResNet22(nn.Module):
    def __init__(self, *, sample_rate, win_length, hop_length, n_mels, f_min,
                 f_max, classes_num, **kwargs):
        """

        Args:
            sample_rate: Sample rate of audio signal.
            win_length: Window size for MelSpectrogram.
            hop_length: Length of hop between STFT windows in MelSpectrogram.
            n_mels: Number of mel filterbanks in MelSpectrogram.
            f_min: Minimum frequency for MelSpectrogram.
            f_max: Maximum frequency for MelSpectrogram.
            classes_num: Amount of classes used for training.
            **kwargs: 'window_fn', 'center', 'pad_mode' for MelSpectrogram
                      (defaults 'hann', 'center', 'reflect'),
                      'top_db' for AmplitudeToDB (default None),
                      'num_features' for BatchNorm2d (default 64),
                      'embedding_size' for the amount of neurons connecting the
                       last two fully connected layers (default 2048)
        """

        super().__init__()
        logging.info("Initializing model ResNet22.")
        logging.info(f"ResNet22 - parameter 'classes_num' is {classes_num}.")
        window_fn = kwargs.get('window_fn', torch.hann_window)
        center = kwargs.get('center', True)
        pad_mode = kwargs.get('pad_mode', 'reflect')
        top_db = kwargs.get('top_db', None)
        num_features = kwargs.get('num_features', 64)
        logging.info(f"ResNet22 - parameter 'num_features is: {num_features}.")
        embedding_size = kwargs.get('embedding_size', 2048)
        logging.info(f"ResNet22 - parameter 'embedding_size is:"
                     f" {embedding_size}.")

        # Spectrogram extractor
        logging.info(f"ResNet22 Spectrogram - parameter 'sample_rate' is: "
                     f"{sample_rate}.")
        logging.info(f"ResNet22 Spectrogram - parameter 'win_length' is: "
                     f"{win_length}.")
        logging.info(f"ResNet22 Spectrogram - parameter 'hop_length' is: "
                     f"{hop_length}.")
        logging.info(f"ResNet22 Spectrogram - parameter 'f_min' is: "
                     f"{f_min}.")
        logging.info(f"ResNet22 Spectrogram - parameter 'f_max' is: "
                     f"{f_max}.")
        logging.info(f"ResNet22 Spectrogram - parameter 'n_mels' is: "
                     f"{n_mels}.")
        logging.info(f"ResNet22 Spectrogram - parameter 'center' is:"
                     f" {center}.")
        logging.info(f"ResNet22 Spectrogram - parameter 'pad_mode' is:"
                     f" {pad_mode}.")
        logging.info(f"ResNet22 Spectrogram - parameter 'top_db' is:"
                     f" {top_db}.")
        self.mel_spectrogram = MelSpectrogram(sample_rate=sample_rate,
                                              n_fft=win_length,
                                              win_length=win_length,
                                              hop_length=hop_length,
                                              f_min=f_min, f_max=f_max,
                                              n_mels=n_mels,
                                              window_fn=window_fn,
                                              power=2, onesided=True,
                                              center=center,
                                              pad_mode=pad_mode)
        self.amplitude_to_db = AmplitudeToDB(stype="power", top_db=top_db)

        # Spec augmenter
        self.spec_aug_time = _SpecAugmentation(mask_param=64,
                                               stripes_num=2,
                                               axis=2)
        self.spec_aug_freq = _SpecAugmentation(mask_param=8,
                                               stripes_num=2,
                                               axis=3)

        self.bn0 = nn.BatchNorm2d(num_features)

        self.conv_block1 = _ConvBlock(in_channels=1, out_channels=64)

        self.resnet = _ResNet(block=_ResnetBasicBlock, layers=[2, 2, 2, 2])

        self.conv_block_after1 = _ConvBlock(in_channels=512, out_channels=2048)

        self.fc1 = nn.Linear(2048, embedding_size, bias=True)
        self.fc_audioset = nn.Linear(embedding_size, classes_num, bias=True)

        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, batch, mixup_lambda=None):
        """

        Args:
            batch: Input tensor of shape (batch_size, time).
            mixup_lambda: If not None, apply mixup with given coefficients
                          (default None).

        Returns: clipwise_output, segmentwise_output, framewise_output,
                 embedding
        """

        x = batch[:, None, :]        # (batch_size, 1, time)
        x = self.mel_spectrogram(x)  # (batch_size, 1, n_mels, time)
        x = self.amplitude_to_db(x)  # (batch_size, 1, n_mels, time)

        x = torch.transpose(x, 1, 2)        # (batch_size, n_mels, 1, time)
        x = torch.transpose(x, 2, 3)        # (batch_size, n_mels, time, 1)
        x = self.bn0(x)

        x = torch.transpose(x, 1, 3)        # (batch_size, 1, time, n_mels)

        if self.training:
            x = self.spec_aug_time(x)
            x = self.spec_aug_freq(x)
            if mixup_lambda is not None:
                x = mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.resnet(x)
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block_after1(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        return ModelOutput(clipwise_output, None, None, embedding)


class ResNet38(nn.Module):
    def __init__(self, *, sample_rate, win_length, hop_length, n_mels, f_min,
                 f_max, classes_num, **kwargs):
        """

        Args:
            sample_rate: Sample rate of audio signal.
            win_length: Window size for MelSpectrogram.
            hop_length: Length of hop between STFT windows in MelSpectrogram.
            n_mels: Number of mel filterbanks in MelSpectrogram.
            f_min: Minimum frequency for MelSpectrogram.
            f_max: Maximum frequency for MelSpectrogram.
            classes_num: Amount of classes used for training.
            **kwargs: 'window_fn', 'center', 'pad_mode' for MelSpectrogram
                      (defaults 'hann', 'center', 'reflect'),
                      'top_db' for AmplitudeToDB (default None),
                      'num_features' for BatchNorm2d (default 64),
                      'embedding_size' for the amount of neurons connecting the
                       last two fully connected layers (default 2048)
        """

        super().__init__()

        logging.info("Initializing model ResNet38.")
        logging.info(f"ResNet38 - parameter 'classes_num' is: {classes_num}.")
        window_fn = kwargs.get('window_fn', torch.hann_window)
        center = kwargs.get('center', True)
        pad_mode = kwargs.get('pad_mode', 'reflect')
        top_db = kwargs.get('top_db', None)
        num_features = kwargs.get('num_features', 64)
        logging.info(f"ResNet38 - parameter 'num_features' is: {num_features}.")
        embedding_size = kwargs.get('embedding_size', 2048)
        logging.info(f"ResNet38 - parameter 'embedding_size' is:"
                     f" {embedding_size}.")

        # Spectrogram extractor
        logging.info(f"ResNet38 Spectrogram - parameter 'sample_rate' is: "
                     f"{sample_rate}.")
        logging.info(f"ResNet38 Spectrogram - parameter 'win_length' is: "
                     f"{win_length}.")
        logging.info(f"ResNet38 Spectrogram - parameter 'hop_length' is: "
                     f"{hop_length}.")
        logging.info(f"ResNet38 Spectrogram - parameter 'f_min' is: "
                     f"{f_min}.")
        logging.info(f"ResNet38 Spectrogram - parameter 'f_max' is: "
                     f"{f_max}.")
        logging.info(f"ResNet38 Spectrogram - parameter 'n_mels' is: "
                     f"{n_mels}.")
        logging.info(f"ResNet38 Spectrogram - parameter 'center' is:"
                     f" {center}.")
        logging.info(f"ResNet38 Spectrogram - parameter 'pad_mode' is:"
                     f" {pad_mode}.")
        logging.info(f"ResNet38 Spectrogram - parameter 'top_db' is:"
                     f" {top_db}.")
        self.mel_spectrogram = MelSpectrogram(sample_rate=sample_rate,
                                              n_fft=win_length,
                                              win_length=win_length,
                                              hop_length=hop_length,
                                              f_min=f_min, f_max=f_max,
                                              n_mels=n_mels,
                                              window_fn=window_fn,
                                              power=2, onesided=True,
                                              center=center,
                                              pad_mode=pad_mode)
        self.amplitude_to_db = AmplitudeToDB(stype="power", top_db=top_db)

        # Spec augmenter
        self.spec_aug_time = _SpecAugmentation(mask_param=64,
                                               stripes_num=2,
                                               axis=2)
        self.spec_aug_freq = _SpecAugmentation(mask_param=8,
                                               stripes_num=2,
                                               axis=3)

        self.bn0 = nn.BatchNorm2d(num_features)

        self.conv_block1 = _ConvBlock(in_channels=1, out_channels=64)

        self.resnet = _ResNet(block=_ResnetBasicBlock, layers=[3, 4, 6, 3])

        self.conv_block_after1 = _ConvBlock(in_channels=512, out_channels=2048)

        self.fc1 = nn.Linear(2048, embedding_size, bias=True)
        self.fc_audioset = nn.Linear(embedding_size, classes_num, bias=True)

        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, batch, mixup_lambda=None):
        """

        Args:
            batch: Input tensor of shape (batch_size, time).
            mixup_lambda: If not None, apply mixup with given coefficients
                          (default None).

        Returns: clipwise_output, segmentwise_output, framewise_output,
                 embedding
        """

        x = batch[:, None, :]        # (batch_size, 1, time)
        x = self.mel_spectrogram(x)  # (batch_size, 1, n_mels, time)
        x = self.amplitude_to_db(x)  # (batch_size, 1, n_mels, time)

        x = torch.transpose(x, 1, 2)        # (batch_size, n_mels, 1, time)
        x = torch.transpose(x, 2, 3)        # (batch_size, n_mels, time, 1)
        x = self.bn0(x)

        x = torch.transpose(x, 1, 3)        # (batch_size, 1, time, n_mels)

        if self.training:
            x = self.spec_aug_time(x)
            x = self.spec_aug_freq(x)
            if mixup_lambda is not None:
                x = mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.resnet(x)
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block_after1(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        return ModelOutput(clipwise_output, None, None, embedding)


class ResNet54(nn.Module):
    def __init__(self, *, sample_rate, win_length, hop_length, n_mels, f_min,
                 f_max, classes_num, **kwargs):
        """

        Args:
            sample_rate: Sample rate of audio signal.
            win_length: Window size for MelSpectrogram.
            hop_length: Length of hop between STFT windows in MelSpectrogram.
            n_mels: Number of mel filterbanks in MelSpectrogram.
            f_min: Minimum frequency for MelSpectrogram.
            f_max: Maximum frequency for MelSpectrogram.
            classes_num: Amount of classes used for training.
            **kwargs: 'window_fn', 'center', 'pad_mode' for MelSpectrogram
                      (defaults 'hann', 'center', 'reflect'),
                      'top_db' for AmplitudeToDB (default None),
                      'num_features' for BatchNorm2d (default 64),
                      'embedding_size' for the amount of neurons connecting the
                       last two fully connected layers (default 2048)
        """

        super().__init__()

        logging.info("Initializing model ResNet54.")
        logging.info(f"ResNet54 - parameter 'classes_num' is: {classes_num}.")
        window_fn = kwargs.get('window_fn', torch.hann_window)
        center = kwargs.get('center', True)
        pad_mode = kwargs.get('pad_mode', 'reflect')
        top_db = kwargs.get('top_db', None)
        num_features = kwargs.get('num_features', 64)
        logging.info(f"ResNet54 - parameter 'num_features' is: {num_features}.")
        embedding_size = kwargs.get('embedding_size', 2048)
        logging.info(f"ResNet54 - parameter 'embedding_size' is:"
                     f" {embedding_size}.")

        # Spectrogram extractor
        logging.info(f"ResNet54 Spectrogram - parameter 'sample_rate' is: "
                     f"{sample_rate}.")
        logging.info(f"ResNet54 Spectrogram - parameter 'win_length' is: "
                     f"{win_length}.")
        logging.info(f"ResNet54 Spectrogram - parameter 'hop_length' is: "
                     f"{hop_length}.")
        logging.info(f"ResNet54 Spectrogram - parameter 'f_min' is: "
                     f"{f_min}.")
        logging.info(f"ResNet54 Spectrogram - parameter 'f_max' is: "
                     f"{f_max}.")
        logging.info(f"ResNet54 Spectrogram - parameter 'n_mels' is: "
                     f"{n_mels}.")
        logging.info(f"ResNet54 Spectrogram - parameter 'center' is:"
                     f" {center}.")
        logging.info(f"ResNet54 Spectrogram - parameter 'pad_mode' is:"
                     f" {pad_mode}.")
        logging.info(f"ResNet54 Spectrogram - parameter 'top_db' is:"
                     f" {top_db}.")
        self.mel_spectrogram = MelSpectrogram(sample_rate=sample_rate,
                                              n_fft=win_length,
                                              win_length=win_length,
                                              hop_length=hop_length,
                                              f_min=f_min, f_max=f_max,
                                              n_mels=n_mels,
                                              window_fn=window_fn,
                                              power=2, onesided=True,
                                              center=center,
                                              pad_mode=pad_mode)
        self.amplitude_to_db = AmplitudeToDB(stype="power", top_db=top_db)

        # Spec augmenter
        self.spec_aug_time = _SpecAugmentation(mask_param=64,
                                               stripes_num=2,
                                               axis=2)
        self.spec_aug_freq = _SpecAugmentation(mask_param=8,
                                               stripes_num=2,
                                               axis=3)

        self.bn0 = nn.BatchNorm2d(num_features)

        self.conv_block1 = _ConvBlock(in_channels=1, out_channels=64)

        self.resnet = _ResNet(block=_ResnetBottleneck, layers=[3, 4, 6, 3])

        self.conv_block_after1 = _ConvBlock(in_channels=2048, out_channels=2048)

        self.fc1 = nn.Linear(2048, embedding_size, bias=True)
        self.fc_audioset = nn.Linear(embedding_size, classes_num, bias=True)

        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, batch, mixup_lambda=None):
        """

        Args:
            batch: Input tensor of shape (batch_size, time).
            mixup_lambda: If not None, apply mixup with given coefficients
                          (default None).

        Returns: clipwise_output, segmentwise_output, framewise_output,
                 embedding
        """

        x = batch[:, None, :]        # (batch_size, 1, time)
        x = self.mel_spectrogram(x)  # (batch_size, 1, n_mels, time)
        x = self.amplitude_to_db(x)  # (batch_size, 1, n_mels, time)

        x = torch.transpose(x, 1, 2)        # (batch_size, n_mels, 1, time)
        x = torch.transpose(x, 2, 3)        # (batch_size, n_mels, time, 1)
        x = self.bn0(x)

        x = torch.transpose(x, 1, 3)        # (batch_size, 1, time, n_mels)

        if self.training:
            x = self.spec_aug_time(x)
            x = self.spec_aug_freq(x)
            if mixup_lambda is not None:
                x = mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.resnet(x)
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block_after1(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        return ModelOutput(clipwise_output, None, None, embedding)


class Res1dNet31(nn.Module):
    def __init__(self, *, classes_num, **kwargs):
        """

        Args:
            classes_num: Amount of classes used for training.
            **kwargs: 'num_features' for BatchNorm2d (default 64),
                      'embedding_size' for the amount of neurons connecting the
                      last two fully connected layers (default 2048)
        """

        super().__init__()

        logging.info("Initializing model Res1dNet31.")
        logging.info(f"Res1dNet31 - parameter 'classes_num' is {classes_num}.")
        num_features = kwargs.get('num_features', 64)
        logging.info(f"Res1dNet31 - parameter 'num_features' is {num_features}.")
        embedding_size = kwargs.get('embedding_size', 2048)
        logging.info(f"Res1dNet31 - parameter 'embedding_size' is"
                     f" {embedding_size}.")

        self.conv0 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=11,
                               stride=5, padding=5, bias=False)
        self.bn0 = nn.BatchNorm1d(num_features)

        self.resnet = _ResNetWav1d(block=_ResnetBasicBlockWav1d,
                                   layers=[2, 2, 2, 2, 2, 2, 2])

        self.fc1 = nn.Linear(2048, embedding_size, bias=True)
        self.fc_audioset = nn.Linear(embedding_size, classes_num, bias=True)

        init_layer(self.conv0)
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, batch, mixup_lambda=None):
        """

        Args:
            batch: Input tensor of shape (batch_size, time).
            mixup_lambda: If not None, apply mixup with given coefficients
                          (default None).

        Returns: clipwise_output, segmentwise_output, framewise_output,
                 embedding
        """

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

        return ModelOutput(clipwise_output, None, None, embedding)


class Res1dNet51(nn.Module):
    def __init__(self, *, classes_num, **kwargs):
        """

        Args:
            classes_num: Amount of classes used for training.
            **kwargs: 'num_features' for BatchNorm2d (default 64),
                      'embedding_size' for the amount of neurons connecting the
                      last two fully connected layers (default 2048)
        """

        super().__init__()

        logging.info("Initializing model Res1dNet51.")
        logging.info(f"Res1dNet51 - parameter 'classes_num' is {classes_num}.")
        num_features = kwargs.get('num_features', 64)
        logging.info(f"Res1dNet51 - parameter 'num_features' is {num_features}.")
        embedding_size = kwargs.get('embedding_size', 2048)
        logging.info(f"Res1dNet51 - parameter 'embedding_size' is"
                     f" {embedding_size}.")

        self.conv0 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=11,
                               stride=5, padding=5, bias=False)
        self.bn0 = nn.BatchNorm1d(num_features)

        self.resnet = _ResNetWav1d(block=_ResnetBasicBlockWav1d,
                                   layers=[2, 3, 4, 6, 4, 3, 2])

        self.fc1 = nn.Linear(2048, embedding_size, bias=True)
        self.fc_audioset = nn.Linear(embedding_size, classes_num, bias=True)

        init_layer(self.conv0)
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, batch, mixup_lambda=None):
        """

        Args:
            batch: Input tensor of shape (batch_size, time).
            mixup_lambda: If not None, apply mixup with given coefficients
                          (default None).

        Returns: clipwise_output, segmentwise_output, framewise_output,
                 embedding
        """

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

        return ModelOutput(clipwise_output, None, None, embedding)


class MobileNetV1(nn.Module):
    def __init__(self, *, sample_rate, win_length, hop_length, n_mels, f_min,
                 f_max, classes_num, **kwargs):
        """

        Args:
            sample_rate: Sample rate of audio signal.
            win_length: Window size for MelSpectrogram.
            hop_length: Length of hop between STFT windows in MelSpectrogram.
            n_mels: Number of mel filterbanks in MelSpectrogram.
            f_min: Minimum frequency for MelSpectrogram.
            f_max: Maximum frequency for MelSpectrogram.
            classes_num: Amount of classes used for training.
            **kwargs: 'window_fn', 'center', 'pad_mode' for MelSpectrogram
                      (defaults 'hann', 'center', 'reflect'),
                      'top_db' for AmplitudeToDB (default None),
                      'num_features' for BatchNorm2d (default 64),
                      'embedding_size' for the amount of neurons connecting the
                       last two fully connected layers (default 1024)
        """

        super().__init__()

        logging.info("Initializing model MobileNetV1.")
        logging.info(f"MobileNetV1 - parameter 'classes_num' is: "
                     f"{classes_num}.")
        window_fn = kwargs.get('window_fn', torch.hann_window)
        center = kwargs.get('center', True)
        pad_mode = kwargs.get('pad_mode', 'reflect')
        top_db = kwargs.get('top_db', None)
        num_features = kwargs.get('num_features', 64)
        logging.info(f"MobileNetV1 - parameter 'num_features' is: "
                     f"{num_features}.")
        embedding_size = kwargs.get('embedding_size', 1024)
        logging.info(f"MobileNetV1 - parameter 'embedding_size' is: "
                     f"{embedding_size}.")

        # Spectrogram extractor
        logging.info(f"MobileNetV1 Spectrogram - parameter 'sample_rate' is: "
                     f"{sample_rate}.")
        logging.info(f"MobileNetV1 Spectrogram - parameter 'win_length' is: "
                     f"{win_length}.")
        logging.info(f"MobileNetV1 Spectrogram - parameter 'hop_length' is: "
                     f"{hop_length}.")
        logging.info(f"MobileNetV1 Spectrogram - parameter 'f_min' is: "
                     f"{f_min}.")
        logging.info(f"MobileNetV1 Spectrogram - parameter 'f_max' is: "
                     f"{f_max}.")
        logging.info(f"MobileNetV1 Spectrogram - parameter 'n_mels' is: "
                     f"{n_mels}.")
        logging.info(f"MobileNetV1 Spectrogram - parameter 'center' is:"
                     f" {center}.")
        logging.info(f"MobileNetV1 Spectrogram - parameter 'pad_mode' is:"
                     f" {pad_mode}.")
        logging.info(f"MobileNetV1 Spectrogram - parameter 'top_db' is:"
                     f" {top_db}.")
        self.mel_spectrogram = MelSpectrogram(sample_rate=sample_rate,
                                              n_fft=win_length,
                                              win_length=win_length,
                                              hop_length=hop_length,
                                              f_min=f_min, f_max=f_max,
                                              n_mels=n_mels,
                                              window_fn=window_fn,
                                              power=2, onesided=True,
                                              center=center,
                                              pad_mode=pad_mode)
        self.amplitude_to_db = AmplitudeToDB(stype="power", top_db=top_db)

        # Spec augmenter
        self.spec_aug_time = _SpecAugmentation(mask_param=64,
                                               stripes_num=2,
                                               axis=2)
        self.spec_aug_freq = _SpecAugmentation(mask_param=8,
                                               stripes_num=2,
                                               axis=3)

        self.bn0 = nn.BatchNorm2d(num_features)

        self.features = nn.Sequential(
                _ConvBnV1(1, 32, 2),
                _ConvDw(32, 64, 1),
                _ConvDw(64, 128, 2),
                _ConvDw(128, 128, 1),
                _ConvDw(128, 256, 2),
                _ConvDw(256, 256, 1),
                _ConvDw(256, 512, 2),
                _ConvDw(512, 512, 1),
                _ConvDw(512, 512, 1),
                _ConvDw(512, 512, 1),
                _ConvDw(512, 512, 1),
                _ConvDw(512, 512, 1),
                _ConvDw(512, 1024, 2),
                _ConvDw(1024, 1024, 1))

        self.fc1 = nn.Linear(1024, embedding_size, bias=True)
        self.fc_audioset = nn.Linear(embedding_size, classes_num, bias=True)

        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, batch, mixup_lambda=None):
        """

        Args:
            batch: Input tensor of shape (batch_size, time).
            mixup_lambda: If not None, apply mixup with given coefficients
                          (default None).

        Returns: clipwise_output, segmentwise_output, framewise_output,
                 embedding
        """

        x = batch[:, None, :]        # (batch_size, 1, time)
        x = self.mel_spectrogram(x)  # (batch_size, 1, n_mels, time)
        x = self.amplitude_to_db(x)  # (batch_size, 1, n_mels, time)

        x = torch.transpose(x, 1, 2)        # (batch_size, n_mels, 1, time)
        x = torch.transpose(x, 2, 3)        # (batch_size, n_mels, time, 1)
        x = self.bn0(x)

        x = torch.transpose(x, 1, 3)        # (batch_size, 1, time, n_mels)

        if self.training:
            x = self.spec_aug_time(x)
            x = self.spec_aug_freq(x)
            if mixup_lambda is not None:
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

        return ModelOutput(clipwise_output, None, None, embedding)


class MobileNetV2(nn.Module):
    def __init__(self, *, sample_rate, win_length, hop_length, n_mels, f_min,
                 f_max, classes_num, **kwargs):
        """

        Args:
            sample_rate: Sample rate of audio signal.
            win_length: Window size for MelSpectrogram.
            hop_length: Length of hop between STFT windows in MelSpectrogram.
            n_mels: Number of mel filterbanks in MelSpectrogram.
            f_min: Minimum frequency for MelSpectrogram.
            f_max: Maximum frequency for MelSpectrogram.
            classes_num: Amount of classes used for training.
            **kwargs: 'window_fn', 'center', 'pad_mode' for MelSpectrogram
                      (defaults 'hann', 'center', 'reflect'),
                      'top_db' for AmplitudeToDB (default None),
                      'num_features' for BatchNorm2d (default 64),
                      'embedding_size' for the amount of neurons connecting the
                       last two fully connected layers (default 1024)
        """

        super().__init__()

        logging.info("Initializing model MobileNetV2.")
        logging.info(f"MobileNetV2 - parameter 'classes_num' is: "
                     f"{classes_num}.")
        window_fn = kwargs.get('window_fn', torch.hann_window)
        center = kwargs.get('center', True)
        pad_mode = kwargs.get('pad_mode', 'reflect')
        top_db = kwargs.get('top_db', None)
        num_features = kwargs.get('num_features', 64)
        logging.info(f"MobileNetV2 - parameter 'num_features' is: "
                     f"{num_features}.")
        embedding_size = kwargs.get('embedding_size', 1024)
        logging.info(f"MobileNetV2 - parameter 'embedding_size' is: "
                     f"{embedding_size}.")

        # Spectrogram extractor
        logging.info(f"MobileNetV2 Spectrogram - parameter 'sample_rate' is: "
                     f"{sample_rate}.")
        logging.info(f"MobileNetV2 Spectrogram - parameter 'win_length' is: "
                     f"{win_length}.")
        logging.info(f"MobileNetV2 Spectrogram - parameter 'hop_length' is: "
                     f"{hop_length}.")
        logging.info(f"MobileNetV2 Spectrogram - parameter 'f_min' is: "
                     f"{f_min}.")
        logging.info(f"MobileNetV2 Spectrogram - parameter 'f_max' is: "
                     f"{f_max}.")
        logging.info(f"MobileNetV2 Spectrogram - parameter 'n_mels' is: "
                     f"{n_mels}.")
        logging.info(f"MobileNetV2 Spectrogram - parameter 'center' is:"
                     f" {center}.")
        logging.info(f"MobileNetV2 Spectrogram - parameter 'pad_mode' is:"
                     f" {pad_mode}.")
        logging.info(f"MobileNetV2 Spectrogram - parameter 'top_db' is:"
                     f" {top_db}.")
        self.mel_spectrogram = MelSpectrogram(sample_rate=sample_rate,
                                              n_fft=win_length,
                                              win_length=win_length,
                                              hop_length=hop_length,
                                              f_min=f_min, f_max=f_max,
                                              n_mels=n_mels,
                                              window_fn=window_fn,
                                              power=2, onesided=True,
                                              center=center,
                                              pad_mode=pad_mode)
        self.amplitude_to_db = AmplitudeToDB(stype="power", top_db=top_db)

        # Spec augmenter
        self.spec_aug_time = _SpecAugmentation(mask_param=64,
                                               stripes_num=2,
                                               axis=2)
        self.spec_aug_freq = _SpecAugmentation(mask_param=8,
                                               stripes_num=2,
                                               axis=3)

        self.bn0 = nn.BatchNorm2d(num_features)

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

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(
            last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [_ConvBnV2(1, input_channel, 2)]
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
        self.features.append(_Conv1x1Bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        self.fc1 = nn.Linear(1280, embedding_size, bias=True)
        self.fc_audioset = nn.Linear(embedding_size, classes_num, bias=True)

        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, batch, mixup_lambda=None):
        """

        Args:
            batch: Input tensor of shape (batch_size, time).
            mixup_lambda: If not None, apply mixup with given coefficients
                          (default None).

        Returns: clipwise_output, segmentwise_output, framewise_output,
                 embedding
        """

        x = batch[:, None, :]        # (batch_size, 1, time)
        x = self.mel_spectrogram(x)  # (batch_size, 1, n_mels, time)
        x = self.amplitude_to_db(x)  # (batch_size, 1, n_mels, time)

        x = torch.transpose(x, 1, 2)        # (batch_size, n_mels, 1, time)
        x = torch.transpose(x, 2, 3)        # (batch_size, n_mels, time, 1)
        x = self.bn0(x)

        x = torch.transpose(x, 1, 3)        # (batch_size, 1, time, n_mels)

        if self.training:
            x = self.spec_aug_time(x)
            x = self.spec_aug_freq(x)
            if mixup_lambda is not None:
                x = mixup(x, mixup_lambda)

        x = self.features(x)

        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        return ModelOutput(clipwise_output, None, None, embedding)


class LeeNet11(nn.Module):
    def __init__(self, *, classes_num, **kwargs):
        """

        Args:
            classes_num: Amount of classes used for training.
            **kwargs: 'embedding_size' for the amount of neurons connecting the
                      last two fully connected layers (default 512)

        """
        super().__init__()

        embedding_size = kwargs.get('embedding_size', 512)

        self.conv_block1 = _LeeNetConvBlock(1, 64, 3, 3)
        self.conv_block2 = _LeeNetConvBlock(64, 64, 3, 1)
        self.conv_block3 = _LeeNetConvBlock(64, 64, 3, 1)
        self.conv_block4 = _LeeNetConvBlock(64, 128, 3, 1)
        self.conv_block5 = _LeeNetConvBlock(128, 128, 3, 1)
        self.conv_block6 = _LeeNetConvBlock(128, 128, 3, 1)
        self.conv_block7 = _LeeNetConvBlock(128, 128, 3, 1)
        self.conv_block8 = _LeeNetConvBlock(128, 128, 3, 1)
        self.conv_block9 = _LeeNetConvBlock(128, 256, 3, 1)

        self.fc1 = nn.Linear(256, embedding_size, bias=True)
        self.fc_audioset = nn.Linear(embedding_size, classes_num, bias=True)

        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, batch, mixup_lambda=None):
        """

        Args:
            batch: Input tensor of shape (batch_size, time).
            mixup_lambda: If not None, apply mixup with given coefficients
                          (default None).

        Returns: clipwise_output, segmentwise_output, framewise_output,
                 embedding
        """

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

        return ModelOutput(clipwise_output, None, None, embedding)


class LeeNet24(nn.Module):
    def __init__(self, *, classes_num, dropout=True, **kwargs):
        """

        Args:
            classes_num: Amount of classes used for training.
            dropout: If True, apply dropout in dropout layers during training.
            **kwargs: 'embedding_size' for the amount of neurons connecting the
                      last two fully connected layers (default 1024)
        """
        super().__init__()

        logging.info("Initializing model LeeNet24.")
        logging.info(f"LeeNet24 - parameter 'classes_num' is {classes_num}.")
        self.dropout = dropout
        logging.info(f"LeeNet24 - parameter 'dropout' is {dropout}.")

        embedding_size = kwargs.get('embedding_size', 1024)
        logging.info(f"LeeNet24 - parameter 'embedding_size' is {embedding_size}.")

        self.conv_block1 = _LeeNetConvBlock2(1, 64, 3, 3)
        self.conv_block2 = _LeeNetConvBlock2(64, 96, 3, 1)
        self.conv_block3 = _LeeNetConvBlock2(96, 128, 3, 1)
        self.conv_block4 = _LeeNetConvBlock2(128, 128, 3, 1)
        self.conv_block5 = _LeeNetConvBlock2(128, 256, 3, 1)
        self.conv_block6 = _LeeNetConvBlock2(256, 256, 3, 1)
        self.conv_block7 = _LeeNetConvBlock2(256, 512, 3, 1)
        self.conv_block8 = _LeeNetConvBlock2(512, 512, 3, 1)
        self.conv_block9 = _LeeNetConvBlock2(512, 1024, 3, 1)

        self.fc1 = nn.Linear(1024, embedding_size, bias=True)
        self.fc_audioset = nn.Linear(embedding_size, classes_num, bias=True)

        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, batch, mixup_lambda=None):
        """

        Args:
            batch: Input tensor of shape (batch_size, time).
            mixup_lambda: If not None, apply mixup with given coefficients
                          (default None).

        Returns: clipwise_output, segmentwise_output, framewise_output,
                 embedding
        """

        x = batch[:, None, :]

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = mixup(x, mixup_lambda)

        x = self.conv_block1(x)
        x = F.dropout(x, p=0.1, training=(self.training and
                                          self.dropout))
        x = self.conv_block2(x, pool_size=3)
        x = F.dropout(x, p=0.1, training=(self.training and
                                          self.dropout))
        x = self.conv_block3(x, pool_size=3)
        x = F.dropout(x, p=0.1, training=(self.training and
                                          self.dropout))
        x = self.conv_block4(x, pool_size=3)
        x = F.dropout(x, p=0.1, training=(self.training and
                                          self.dropout))
        x = self.conv_block5(x, pool_size=3)
        x = F.dropout(x, p=0.1, training=(self.training and
                                          self.dropout))
        x = self.conv_block6(x, pool_size=3)
        x = F.dropout(x, p=0.1, training=(self.training and
                                          self.dropout))
        x = self.conv_block7(x, pool_size=3)
        x = F.dropout(x, p=0.1, training=(self.training and
                                          self.dropout))
        x = self.conv_block8(x, pool_size=3)
        x = F.dropout(x, p=0.1, training=(self.training and
                                          self.dropout))
        x = self.conv_block9(x, pool_size=1)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        return ModelOutput(clipwise_output, None, None, embedding)


class DaiNet19(nn.Module):
    def __init__(self, *, classes_num, **kwargs):
        """

        Args:
            classes_num: Amount of classes used for training.
            **kwargs: 'embedding_size' for the amount of neurons connecting the
                      last two fully connected layers (default 512)
        """

        super().__init__()

        logging.info("Initializing model DaiNet19.")
        logging.info(f"DaiNet19 - parameter 'classes_num' is {classes_num}.")
        num_features = kwargs.get('num_features', 64)
        logging.info(f"DaiNet19 - parameter 'num_features' is"
                     f" {num_features}.")
        embedding_size = kwargs.get('embedding_size', 512)
        logging.info(f"DaiNet19 - parameter 'embedding_size' is"
                     f" {embedding_size}.")

        self.conv0 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=80,
                               stride=4, padding=0, bias=False)
        self.bn0 = nn.BatchNorm1d(num_features)
        self.conv_block1 = _DaiNetResBlock(64, 64, 3)
        self.conv_block2 = _DaiNetResBlock(64, 128, 3)
        self.conv_block3 = _DaiNetResBlock(128, 256, 3)
        self.conv_block4 = _DaiNetResBlock(256, 512, 3)

        self.fc1 = nn.Linear(512, embedding_size, bias=True)
        self.fc_audioset = nn.Linear(embedding_size, classes_num, bias=True)

        init_layer(self.conv0)
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, batch, mixup_lambda=None):
        """

        Args:
            batch: Input tensor of shape (batch_size, time).
            mixup_lambda: If not None, apply mixup with given coefficients
                          (default None).

        Returns: clipwise_output, segmentwise_output, framewise_output,
                 embedding
        """

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

        return ModelOutput(clipwise_output, None, None, embedding)
