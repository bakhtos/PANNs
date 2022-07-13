import numpy as np
import argparse
import os
import glob
import datetime
import time
import logging
import h5py
import librosa

from metadata_utils import get_labels_metadata, read_metadata
from file_utils import create_folder, get_filename
from logger import create_logging 
from array_utils import float32_to_int16, pad_or_truncate


def waveforms_to_hdf5(audios_dir, csv_path, waveforms_hdf5_path, mini_data=False,
                           sample_rate, classes_num):
    """.. py:function:: waveforms_to_hdf5(audios_dir, csv_path, waveforms_hdf5_path, sample_rate, classes_num [, mini_data=False])
    
        Pack waveform and target of several audio clips to a single hdf5 file. 

        :param str audios_dir: Path to the directory containing files to be packed
        :param str csv_path: Path to the csv (tsv) file of  weak class labels for the audios
        :param str waveforms_hdf5_path: Path to save the hdf5-packed file
        :param boolean mini_data: If True, use only 10 files (for debugging, default False)
        :param int sample_rate: Sample rate of packed audios
        :param int classes_num: Amount of classes used in the dataset
        :return: None
        :rtype: None
    """

    clip_samples = sample_rate*10

    _,_,_,_,id_to_ix,_ = get_labels_metadata()

    # Paths
    if mini_data:
        prefix = 'mini_'
        waveforms_hdf5_path += '.mini'
    else:
        prefix = ''

    create_folder(os.path.dirname(waveforms_hdf5_path))

    logs_dir = '_logs/pack_waveforms_to_hdf5/{}{}'.format(prefix, get_filename(csv_path))
    create_folder(logs_dir)
    create_logging(logs_dir, filemode='w')
    logging.info('Write logs to {}'.format(logs_dir))
    
    # Read csv file
    audio_names, targets = read_metadata(csv_path, classes_num, id_to_ix)

    if mini_data:
        mini_num = 10
        audio_names = audio_names[0:10]
        tagrets = targets[0:10]

    audios_num = len(audio_names)

    # Pack waveform to hdf5
    total_time = time.time()

    with h5py.File(waveforms_hdf5_path, 'w') as hf:
        hf.create_dataset('audio_name', shape=((audios_num,)), dtype='S20')
        hf.create_dataset('waveform', shape=((audios_num, clip_samples)), dtype=np.int16)
        hf.create_dataset('target', shape=((audios_num, classes_num)), dtype=np.bool)
        hf.attrs.create('sample_rate', data=sample_rate, dtype=np.int32)

        # Pack waveform & target of several audio clips to a single hdf5 file
        for n in range(audios_num):
            audio_path = os.path.join(audios_dir, audio_names[n])

            if os.path.isfile(audio_path):
                logging.info('{} {}'.format(n, audio_path))
                (audio, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
                audio = pad_or_truncate(audio, clip_samples)

                hf['audio_name'][n] = audio_names[n].encode()
                hf['waveform'][n] = float32_to_int16(audio)
                hf['target'][n] = targets[n]
            else:
                logging.info('{} File does not exist! {}'.format(n, audio_path))

    logging.info('Write to {}'.format(waveforms_hdf5_path))
    logging.info('Pack hdf5 time: {:.3f}'.format(time.time() - total_time))
          

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--audios_dir', type=str, required=True, help='Directory with the  downloaded audio.')
    parser.add_argument('--csv_path', type=str, required=True, help='Path of csv file containing audio info.')
    parser.add_argument('--waveforms_hdf5_path', type=str, required=True, help='Path to save packed hdf5.')
    parser.add_argument('--mini_data', action='store_true', default=False, help='Set true to only use 10 audios (for debugging).')
    parser.add_argument('--sample_rate', type=int, default=44100, help='Sample rate of the used audios.')
    parser.add_argument('--classes_num', type=int, default=110, help='The amount of classes used in the dataset.')

    args = parser.parse_args()
    waveforms_to_hdf5(audios_dir=args.audios_dir, csv_path=args.csv_path,
                           waveforms_hdf5_path=args.waveforms_hdf5_path,
                           mini_data=args.mini_data, sample_rate=args.sample_rate,
                           classes_num=args.classes_num)
