import torch

import panns.models.models

__all__ = ['load_model']


def load_model(model, sample_rate, win_length, hop_length, n_mels, f_min, f_max,
               classes_num, checkpoint=None):
    """Instantiate a model object and load a checkpoint for it.
    Parameters
    __________
    model : str,
        Name of model class from panns.models.models
    sample_rate : int,
        Sample rate of audios used for training (passed to model constructor)
    win_length : int,
        Length of Hamming window used in the model (passed to model
                                                    constructor)
    hop_length : int,
        Length of hop for the Hamming window used in the model (passed to
                                                            model constructor)
    n_mels : int,
        Number of mel bins used in the model (passed to model constructor)
    f_min : int,
        Minimum frequency for mel spectrogram (passed to model constructor)
    f_max : int,
        Maximum frequency for mel spectrogram (passed to model constructor)
    classes_num : int,
        Number of classes used in the model (passed to model constructor)
    checkpoint : str, optional (default None)
        Path to saved checkpoint (state_dict) of the model to be loaded
    """

    if model in panns.models.models.__all__:
        model = eval("panns.models."+model)
    else:
        raise ValueError(f"'{model}' is not among the defined models.")

    model = model(sample_rate=sample_rate, window_size=win_length,
                  hop_size=hop_length, mel_bins=n_mels, fmin=f_min, fmax=f_max,
                  classes_num=classes_num)

    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint))

    return model
