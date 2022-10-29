from torch.utils.data import Dataset
import numpy as np
import h5py

__all__ = ['AudioSetDataset']


class AudioSetDataset(Dataset):
    """Take meta of the audio clip, return waveform and target vector."""
    def __init__(self, hdf5_path, target_path):
        self._hdf5_data = h5py.File(hdf5_path, 'r')
        self._target = np.load(target_path)

    def __del__(self):
        self._hdf5_data.close()
        del self._target

    def __len__(self):
        return self._target.shape[0]

    def __getitem__(self, i):
        """Load waveform and target of an audio clip.
        """

        waveform = self._hdf5_data['waveform'][i]
        target = self._target[i, :]

        return waveform, target

# TODO - add Balanced Sampler
