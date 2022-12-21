import argparse

import torch

import panns.models.models

__all__ = ['load_model', 'model_parser']

model_parser = argparse.ArgumentParser(add_help=False)
model_parser.add_argument('--model_type', type=str, required=True,
                          help="Name of model to train")
model_parser.add_argument('--classes_num', type=int, default=110,
                          help="Amount of classes used in the dataset ("
                               "default 110)")
model_parser.add_argument('--win_length', type=int, default=1024,
                          help="Window size of filter to be used in training ("
                               "default 1024)")
model_parser.add_argument('--hop_length', type=int, default=320,
                          help="Hop size of filter to be used in training ("
                               "default 320)")
model_parser.add_argument('--sample_rate', type=int, default=32000,
                          help="Sample rate of the used audio clips; supported "
                               "values are 32000, 16000, 8000 (default 32000)")
model_parser.add_argument('--f_min', type=int, default=50,
                          help="Minimum frequency to be used when creating "
                               "Logmel filterbank (default 50)")
model_parser.add_argument('--f_max', type=int, default=14000,
                          help="Maximum frequency to be used when creating "
                               "Logmel filterbank (default 14000)")
model_parser.add_argument('--n_mels', type=int, default=64,
                          help="Amount of mel filters to use in the filterbank "
                               "(default 64)")
model_parser.add_argument('--decision_level', choices=['max', 'att', 'avg'],
                          default=None, help="If given, model will create "
                                             "Strong labels using the given "
                                             "function")
spec_aug = model_parser.add_mutually_exclusive_group()
spec_aug.add_argument('--spec_aug', action='store_true',
                      help="If set, use Spectrogram Augmentation during "
                           "training")
spec_aug.add_argument('--no_spec_aug', action='store_false',
                      help="If set, do not use Spectrogram Augmentation "
                           "during training")
mixup_time = model_parser.add_mutually_exclusive_group()
mixup_time.add_argument('--mixup_time', action='store_true',
                        help='If set, perform mixup in time domain')
mixup_time.add_argument('--no_mixup_time', action='store_false',
                        help='If set, do not perform mixup in time domain')
mixup_freq = model_parser.add_mutually_exclusive_group()
mixup_freq.add_argument('--mixup_freq', action='store_true',
                        help='If set, perform mixup in frequency domain')
mixup_freq.add_argument('--no_mixup_freq', action='store_false',
                        help='If set, do not perform mixup in frequency domain')
dropout = model_parser.add_mutually_exclusive_group()
dropout.add_argument('--dropout', action='store_true',
                     help='If set, perform dropout when training')
dropout.add_argument('--no_dropout', action='store_false',
                     help='If set, do not perform dropout when training')
wavegram = model_parser.add_mutually_exclusive_group()
wavegram.add_argument('--wavegram', action='store_true',
                      help='If set, use wavegram features')
wavegram.add_argument('--no_wavegram', action='store_false',
                      help='If set, use wavegram features')
spectrogram = model_parser.add_mutually_exclusive_group()
spectrogram.add_argument('--spectrogram', action='store_true',
                         help='If set, use spectrogram features')
spectrogram.add_argument('--no_spectrogram', action='store_false',
                         help='If set, do not use spectrogram features')
center = model_parser.add_mutually_exclusive_group()
center.add_argument('--center', action='store_true',
                    help="Set True to center in torchaudio.MelSpectrogram")
center.add_argument('--no_center', action='store_false',
                    help="Set False to center in torchaudio.MelSpectrogram")
model_parser.add_argument('--pad_mode', type=str, required=False,
                          default='reflect', help='Argument for '
                                                  'torchaudio.MelSpectrogram')
model_parser.add_argument('--top_db', type=float, default=None, required=False,
                          help='Argument for torchaudio.AmplitudeToDB')
model_parser.add_argument('--num_features', type=int, default=None,
                          required=False, help='Argument for '
                                               'torch.nn.BatchNorm2d')
model_parser.add_argument('--embedding_size', type=int, default=None,
                          required=False, help='Amount of nodes connecting '
                                               'the last two layers of the '
                                               'model')


def load_model(model, checkpoint=None, **kwargs):
    """
    
    Args:
        model: Name of model class from panns.models.models.
        checkpoint: Path to saved checkpoint (state_dict) of the model
                    to be loaded (default None).
        **kwargs: Other keyword arguments for the model.

    Returns: model
    """

    if model in panns.models.models.__all__:
        model = eval("panns.models.models." + model)
    else:
        raise ValueError(f"'{model}' is not among the defined models.")

    if kwargs['top_db'] is None: del kwargs['top_db']
    if kwargs['num_features'] is None: del kwargs['num_features']
    if kwargs['embedding_size'] is None: del kwargs['embedding_size']

    model = model(**kwargs)

    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint))

    return model
