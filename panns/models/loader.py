import panns.models.models

__all__ = ['load_model']


def load_model(model,
               sample_rate,
               win_length,
               hop_length,
               n_mels,
               f_min,
               f_max,
               classes_num):

    if model in panns.models.__all__:
        model = eval("panns.models."+model)
    else:
        raise ValueError(f"'{model}' is not among the defined models.")

    model = model(sample_rate=sample_rate, window_size=win_length,
                  hop_size=hop_length, mel_bins=n_mels, fmin=f_min, fmax=f_max,
                  classes_num=classes_num)

    return model
