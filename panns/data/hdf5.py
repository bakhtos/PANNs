import numpy as np
import argparse
import csv
import os
import glob
import datetime
import time
import logging
import h5py
import librosa

from panns.utils.metadata_utils import get_labels, get_weak_target
from panns.utils.logging_utils import create_logging 

__all__ = ['wav_to_hdf5']

def wav_to_hdf5(*, audios_dir, hdf5_path,
                   audio_names, logs_dir=None,
                   clip_length=10000, sample_rate=32000,
                   mini_data=0):
    """Pack waveform of several audio clips to a single hdf5 file.

       Parameters
       __________

       audios_dir : str,
            Path to the directory containing files to be packed
       hdf5_path : str,
            Path to save the hdf5-packed file
       audio_names : numpy.ndarray,
            Array of file names in the dataset, in the same order as in target
       logs_dir : str, optional (default None)
            Directory to save logs into, if None creates a directory 'logs' in CWD
       clip_length : int, optional (default 10000)
            Length (in ms) of audio clips used in the dataset
       sample_rate : int, optional (default 32000)
            Sample rate of packed audios
       mini_data : int, optional (default 0)
            If greater than 0, use only this many first files
    """

    clip_samples = sample_rate*clip_length//1000

    os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)

    if logs_dir is None:
        logs_dir = os.path.join(os.getcwd(), 'logs')
    create_logging(logs_dir, filemode='w')
    logging.info('Write logs to {}'.format(logs_dir))
    

    if mini_data > 0:
        audio_names = audio_names[0:mini_data]

    audios_num = len(audio_names)

    # Pack waveform to hdf5
    start_time = time.time()

    with h5py.File(hdf5_path, 'w') as hf:
        hf.create_dataset('waveform', shape=(audios_num, clip_samples), dtype=np.float32)

        # Pack waveform & target of several audio clips to a single hdf5 file
        for n in range(audios_num):
            audio_path = os.path.join(audios_dir, audio_names[n])

            if os.path.isfile(audio_path):
                logging.info(f'{n} - {audio_path}')
                (audio, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True, dtype=np.float32)
                if len(audio) >= clip_samples: audio2 = audio[:clip_samples]
                if len(audio) < clip_samples:
                    audio2 = np.zeros(clip_samples)
                    audio2[:len(audio)] = audio

                hf['waveform'][n] = audio2
            else:
                logging.info(f'{n} - File does not exist: {audio_path}')

    fin_time = time.time()
    logging.info(f'Written to {hdf5_path}')
    logging.info(f'Pack time: {fin_time-start_time:.3f}')
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--audios_dir', type=str, required=True,
                                help='Directory with the  downloaded audio.')
    parser.add_argument('--audio_names_path', type=str, required=True,
                        help='Path to .npy file to load audio filenames to be packed in this order')
    parser.add_argument('--hdf5_path', type=str, required=True,
                                            help='Path to save packed hdf5.')
    parser.add_argument('--sample_rate', type=int, default=44100,
                                         help='Sample rate of the used audios.')
    parser.add_argument('--mini_data', type=int, default=0,
                            help='If specified, use only this many audios.')
    parser.add_argument('--clip_length', type=int, default=10000,
                help='Length (in ms) of audio clips used in the dataset (default 10000)')
    parser.add_argument('--logs_dir', type=str, default=None,
                            help='Directory to save logs into.')
    args = parser.parse_args()
    
    audio_names = np.load(args.audio_names_path)

    wav_to_hdf5(audios_dir=args.audios_dir,
                      hdf5_path=args.hdf5_path,
                      audio_names=audio_names,
                      clip_length=args.clip_length,
                      sample_rate=args.sample_rate,
                      mini_data=args.mini_data,
                      logs_dir=args.logs_dir)