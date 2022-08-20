import numpy as np

__all__ = ['float32_to_int16',
           'int16_to_float32',
           'pad_or_truncate']


def float32_to_int16(x):
    assert np.max(np.abs(x)) <= 1.2
    x = np.clip(x, -1, 1)
    return (x * 32767.).astype(np.int16)

def int16_to_float32(x):
    return (x / 32767.).astype(np.float32)
    

def pad_or_truncate(x, audio_length):
    """Pad or truncate audio to match audio_length."""
    if len(x) <= audio_length:
        return np.concatenate((x, np.zeros(audio_length - len(x))), axis=0)
    else:
        return x[0 : audio_length]
