import os
import time
import logging
import argparse
import pickle

import numpy as np
import h5py
import librosa

from panns.logging import create_logging

__all__ = ['wav_to_hdf5']


def wav_to_hdf5(*, audios_dir, hdf5_path,
                audio_names, logs_dir=None,
                clip_length=10000, sample_rate=32000,
                mini_data=0):
    """Pack waveform of several audio clips to a single hdf5 file.

        Args:
            audios_dir: str,
                Path to the directory containing files to be packed
            hdf5_path: str,
                Path to save the hdf5-packed file
            audio_names: numpy.ndarray,
                Array of file names in the dataset, in the same order as in target
            logs_dir: str, Directory to save logs into,
                if None creates a directory 'logs' in CWD (default None)
            clip_length: int, Length (in ms) of audio clips used in the dataset
                (default 10000)
            sample_rate: int, Sample rate of packed audios (default 32000)
            mini_data: int, If greater than 0, use only this many first files
                (default 0)
    """

    clip_samples = sample_rate * clip_length // 1000

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
        hf.create_dataset('waveform', shape=(audios_num, clip_samples),
                          dtype=np.float32)

        # Pack waveform & target of several audio clips to a single hdf5 file
        for n in range(audios_num):
            start_time_file = time.time()
            audio_path = os.path.join(audios_dir, "Y"+audio_names[n]+".wav")

            if os.path.isfile(audio_path):
                (audio, _) = librosa.core.load(audio_path, sr=sample_rate,
                                               mono=True, dtype=np.float32)
                if len(audio) >= clip_samples: audio2 = audio[:clip_samples]
                if len(audio) < clip_samples:
                    audio2 = np.zeros(clip_samples)
                    audio2[:len(audio)] = audio

                hf['waveform'][n] = audio2
                fin_time_file = time.time()
                logging.info(f'{n} - {audio_path} packed in '
                             f'{fin_time_file-start_time_file:.3f} s')
            else:
                logging.info(f'{n} - File does not exist: {audio_path}')

    fin_time = time.time()
    logging.info(f'Written to {hdf5_path}')
    logging.info(f'Pack time: {fin_time - start_time:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--audios_dir', type=str, required=True,
                        help="Directory with the downloaded audio")
    parser.add_argument('--audio_names_path', type=str, required=True,
                        help="Path to the pickle file to load audio filenames "
                             "to be packed in this order")
    parser.add_argument('--hdf5_path', type=str, required=True,
                        help="Path to save packed hdf5")
    parser.add_argument('--sample_rate', type=int, default=32000,
                        help="Sample rate of the used audios (default 32000)")
    parser.add_argument('--mini_data', type=int, default=0,
                        help="If specified, use only this many audios")
    parser.add_argument('--clip_length', type=int, default=10000,
                        help="Length (in ms) of audio clips used in the "
                             "dataset (default 10000)")
    parser.add_argument('--logs_dir', type=str, default=None,
                        help="Directory to save logs into "
                             "(defaults to 'logs' in CWD)")
    args = parser.parse_args()

    with open(args.audio_names_path, 'rb') as f:
        audio_names = pickle.load(f)

    wav_to_hdf5(audios_dir=args.audios_dir,
                hdf5_path=args.hdf5_path,
                audio_names=audio_names,
                clip_length=args.clip_length,
                sample_rate=args.sample_rate,
                mini_data=args.mini_data,
                logs_dir=args.logs_dir)
