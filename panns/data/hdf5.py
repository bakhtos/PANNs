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

from panns.utils.file_utils import create_folder, get_sub_filepaths, get_filename
from panns.utils.metadata_utils import get_labels_metadata, read_metadata
from panns.utils.logging_utils import create_logging 
from panns.utils.array_utils import float32_to_int16, pad_or_truncate

def waveforms_to_hdf5(audios_dir, csv_path, waveforms_hdf5_path,
                           sample_rate, classes_num, mini_data=0):
    """.. py:function:: waveforms_to_hdf5(audios_dir, csv_path, waveforms_hdf5_path, sample_rate, classes_num, [mini_data=0])
    
        Pack waveform and target of several audio clips to a single hdf5 file. 

        :param str audios_dir: Path to the directory containing files to be packed
        :param str csv_path: Path to the csv (tsv) file of  weak class labels for the audios
        :param str waveforms_hdf5_path: Path to save the hdf5-packed file
        :param int sample_rate: Sample rate of packed audios
        :param int classes_num: Amount of classes used in the dataset
        :param int mini_data: If greater than 0, use only this many first files (for debugging, default 0)
        :return: None
        :rtype: None
    """

    clip_samples = sample_rate*10

    _,_,_,_,id_to_ix,_ = get_labels_metadata()


    create_folder(os.path.dirname(waveforms_hdf5_path))

    logs_dir = os.path.join('_logs', 'pack_waveforms_to_hdf5',get_filename(csv_path))
    create_folder(logs_dir)
    create_logging(logs_dir, filemode='w')
    logging.info('Write logs to {}'.format(logs_dir))
    
    # Read csv file
    audio_names, targets = read_metadata(csv_path, classes_num, id_to_ix)

    if mini_data > 0:
        audio_names = audio_names[0:mini_data]
        tagrets = targets[0:mini_data]

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
 

def create_indexes(waveforms_hdf5_path, indexes_hdf5_path):
    """.. py:function:: create_indexes(waveforms_hdf5_path, indexes_hdf5_path)

        Create indexes of hdf5-packed files for dataloader to read when training.

        :param str waveforms_hdf5_path: Path of the hdf5-packed audios
        :param str indexes_hdf5_path: Path to save indexes of the packed audios
        :return: None
        :rtype: None
    """

    # Paths
    create_folder(os.path.dirname(indexes_hdf5_path))

    with h5py.File(waveforms_hdf5_path, 'r') as hr:
        with h5py.File(indexes_hdf5_path, 'w') as hw:
            audios_num = len(hr['audio_name'])
            hw.create_dataset('audio_name', data=hr['audio_name'][:], dtype='S20')
            hw.create_dataset('target', data=hr['target'][:], dtype=np.bool)
            hw.create_dataset('hdf5_path', data=[waveforms_hdf5_path.encode()] * audios_num, dtype='S200')
            hw.create_dataset('index_in_hdf5', data=np.arange(audios_num), dtype=np.int32)

    print('Write to {}'.format(indexes_hdf5_path))
          

def combine_indexes(indexes_hdf5s_dir, full_indexes_hdf5_path, classes_num):
    """.. py:function:: combine_indexes(indexes_hdf5s_dir, full_indexes_hdf5_path, classes_num)

        Combine several hdf5 indexes to a single hdf5.

        :param str indexes_hdf5_dir: Path to a directory with all indexes to be combined
        :param str full_indexes_hdf5_path: Path to save the combined index
        :param int classes_num: Amount of classes used in the dataset
        :return: None
        :rtype: None
    """

    # Paths
    paths = get_sub_filepaths(indexes_hdf5s_dir)

    print('Total {} hdf5 to combine.'.format(len(paths)))

    with h5py.File(full_indexes_hdf5_path, 'w') as full_hf:
        full_hf.create_dataset(
            name='audio_name', 
            shape=(0,), 
            maxshape=(None,), 
            dtype='S20')
        
        full_hf.create_dataset(
            name='target', 
            shape=(0, classes_num), 
            maxshape=(None, classes_num), 
            dtype=np.bool)

        full_hf.create_dataset(
            name='hdf5_path', 
            shape=(0,), 
            maxshape=(None,), 
            dtype='S200')

        full_hf.create_dataset(
            name='index_in_hdf5', 
            shape=(0,), 
            maxshape=(None,), 
            dtype=np.int32)

        for path in paths:
            with h5py.File(path, 'r') as part_hf:
                print(path)
                n = len(full_hf['audio_name'][:])
                new_n = n + len(part_hf['audio_name'][:])

                full_hf['audio_name'].resize((new_n,))
                full_hf['audio_name'][n : new_n] = part_hf['audio_name'][:]

                full_hf['target'].resize((new_n, classes_num))
                full_hf['target'][n : new_n] = part_hf['target'][:]

                full_hf['hdf5_path'].resize((new_n,))
                full_hf['hdf5_path'][n : new_n] = part_hf['hdf5_path'][:]

                full_hf['index_in_hdf5'].resize((new_n,))
                full_hf['index_in_hdf5'][n : new_n] = part_hf['index_in_hdf5'][:]
                
    print('Write combined full hdf5 to {}'.format(full_indexes_hdf5_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_create_indexes = subparsers.add_parser('create_indexes')
    parser_create_indexes.add_argument('--waveforms_hdf5_path', type=str, required=True, help='Path of packed waveforms hdf5.')
    parser_create_indexes.add_argument('--indexes_hdf5_path', type=str, required=True, help='Path to write out indexes hdf5.')

    parser_combine_full_indexes = subparsers.add_parser('combine_indexes')
    parser_combine_full_indexes.add_argument('--indexes_hdf5s_dir', type=str, required=True, help='Directory containing indexes hdf5s to be combined.')
    parser_combine_full_indexes.add_argument('--full_indexes_hdf5_path', type=str, required=True, help='Path to write out full indexes hdf5 file.')
    parser_combine_full_indexes.add_argument('--classes_num', type=int, default=110, help='The amount of classes used in the dataset.')

    parser_waveforms_to_hdf5 = subparsers.add_parser('waveforms_to_hdf5')
    parser_waveforms_to_hdf5.add_argument('--audios_dir', type=str, required=True, help='Directory with the  downloaded audio.')
    parser_waveforms_to_hdf5.add_argument('--csv_path', type=str, required=True, help='Path of csv file containing audio info.')
    parser_waveforms_to_hdf5.add_argument('--waveforms_hdf5_path', type=str, required=True, help='Path to save packed hdf5.')
    parser_waveforms_to_hdf5.add_argument('--sample_rate', type=int, default=44100, help='Sample rate of the used audios.')
    parser_waveforms_to_hdf5.add_argument('--classes_num', type=int, default=110, help='The amount of classes used in the dataset.')
    parser_waveforms_to_hdf5.add_argument('--mini_data', type=int, default=0, help='If specified, use only this many audios.')

    args = parser.parse_args()
    
    if args.mode == 'create_indexes':
        create_indexes(waveforms_hdf5_path=args.waveforms_hdf5_path,
                       indexes_hdf5_path=args.indexes_hdf5_path)

    elif args.mode == 'combine_indexes':
        combine_indexes(indexes_hdf5s_dir=args.indexes_hdf5s_dir,
                        full_indexes_hdf5_path=args.full_indexes_hdf5_path,
                        classes_num=args.classes_num)

    elif args.mode == 'waveforms_to_hdf5':
        waveforms_to_hdf5(audios_dir=args.audios_dir, csv_path=args.csv_path,
                          waveforms_hdf5_path=args.waveforms_hdf5_path,
                          sample_rate=args.sample_rate,
                          classes_num=args.classes_num,
                          mini_data=args.mini_data)
    else:
        raise Exception('Incorrect arguments!')
