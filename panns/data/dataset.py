from torch.utils.data import Dataset
import numpy as np
import h5py

__all__ = ['AudioSetDataset']


class AudioSetDataset(Dataset):
    """Class that returns an AudioSet segment audiofile and target."""
    def __init__(self, hdf5_path, target_path=None):
        """Initialize the dataset loading.

        Args:
            hdf5_path: str, Path to hdf5 object containing a dataset 'waveform'
                of shape (audios_num, clip_length)
            target_path: str, Path to npy array storing the target of each
                audio clip of shape (audios_num, ...)
        """
        self._hdf5_data = h5py.File(hdf5_path, 'r')
        self._target = np.load(target_path) if target_path is not None else None

    def __del__(self):
        self._hdf5_data.close()
        del self._target

    def __len__(self) -> int:
        return self._hdf5_data['waveform'].shape[0]

    def __getitem__(self, i):
        waveform = self._hdf5_data['waveform'][i]
        target = None if self._target is None else self._target[i, :].astype(
                np.float32)

        return waveform, target
