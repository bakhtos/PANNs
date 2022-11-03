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

__all__ = ['wav_to_hdf5', 'create_indexes', 'combine_indexes']

def wav_to_hdf5(*, audios_dir, hdf5_path,
                   audio_names, target, logs_dir=None,
                   clip_length=10000, sample_rate=32000, classes_num=110,
                   mini_data=0):
    """Pack waveform and target of several audio clips to a single hdf5 file. 
       Parameters
       __________

       audios_dir : str,
            Path to the directory containing files to be packed
       hdf5_path : str,
            Path to save the hdf5-packed file
       audio_names : numpy.ndarray,
            Array of file names in the dataset, in the same order as in target
       target : numpy.ndarray,
            The target (weak) label array to pack
       logs_dir : str, optional (default None)
            Directory to save logs into, if None creates a directory 'logs' in CWD
       clip_length : int, optional (default 10000)
            Length (in ms) of audio clips used in the dataset
       sample_rate : int, optional (default 32000)
            Sample rate of packed audios
       classes_num : int, optional (default 110),
            Amount of classes used in the dataset
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
        target = target[0:mini_data]

    audios_num = len(audio_names)

    # Pack waveform to hdf5
    start_time = time.time()

    with h5py.File(hdf5_path, 'w') as hf:
        hf.create_dataset('audio_name', shape=(audios_num,), dtype='S20')
        hf.create_dataset('waveform', shape=(audios_num, clip_samples), dtype=np.float32)
        hf.create_dataset('target', shape=(audios_num, classes_num), dtype=np.bool)
        hf.attrs.create('sample_rate', data=sample_rate, dtype=np.int32)

        # Pack waveform & target of several audio clips to a single hdf5 file
        for n in range(audios_num):
            audio_path = os.path.join(audios_dir, audio_names[n])

            if os.path.isfile(audio_path):
                logging.info(f'{n} - {audio_path}')
                (audio, _) = librosa.core.load(audio_path, sr=sample_rate,
                                               mono=True, dtype=np.float32)
                if len(audio) >= clip_samples: audio2 = audio[:clip_samples]
                if len(audio) < clip_samples:
                    audio2 = np.zeros(clip_samples)
                    audio2[:len(audio)] = audio

                hf['audio_name'][n] = audio_names[n].encode()
                hf['waveform'][n] = audio2
                hf['target'][n] = target[n]
            else:
                logging.info(f'{n} - File does not exist: {audio_path}')

    fin_time = time.time()
    logging.info(f'Written to {hdf5_path}')
    logging.info(f'Pack time: {fin_time-start_time:.3f}')
 

def create_indexes(*, hdf5_path, hdf5_index_path, logs_dir=None):
    """Create indexes of hdf5-packed files for dataloader to read when training.

       hdf5_path : str,
            Path of the hdf5-packed audios
       hdf5_index_path : str,
            Path to save index of the packed audios
       logs_dir : str, optional (default None)
            Directory to save logs into, if None creates a directory 'logs' in CWD
    """

    # Paths
    os.makedirs(os.path.dirname(hdf5_index_path), exist_ok=True)
    if logs_dir is None:
        logs_dir = os.path.join(os.getcwd(), 'logs')
    create_logging(logs_dir)

    logging.info(f'Creating indexes for {hdf5_path}')
    start_time = time.time()

    with h5py.File(hdf5_path, 'r') as hr:
        with h5py.File(hdf5_index_path, 'w') as hw:
            audios_num = len(hr['audio_name'])
            hw.create_dataset('audio_name', data=hr['audio_name'][:], dtype='S20')
            hw.create_dataset('target', data=hr['target'][:], dtype=np.bool)
            hw.create_dataset('hdf5_path', data=[hdf5_path.encode()] * audios_num, dtype='S200')
            hw.create_dataset('index_in_hdf5', data=np.arange(audios_num), dtype=np.int32)

    fin_time = time.time()
    logging.info(f'Written to {hdf5_index_path}')
    logging.info(f'Pack time: {fin_time-start_time:.3f}')
          

def combine_indexes(*, hdf5_indexes_dir, hdf5_full_index_path, logs_dir=None,
                       classes_num=110):
    """Combine several hdf5 indexes to a single hdf5.

       hdf5_indexes_dir : str,
            Path to a directory with all indexes to be combined
       hdf5_full_index_path : str,
            Path to save the combined index
       classes_num : int, optional (default 100)
            Amount of classes used in the dataset
       logs_dir : str, optional (default None)
            Directory to save logs into, if None creates a directory 'logs' in CWD
    """

    # Paths
    paths = [path.path for path in os.scandir(hdf5_indexes_dir) if path.is_file()]

    if logs_dir is None:
        logs_dir = os.path.join(os.getcwd(), 'logs')
    create_logging(logs_dir)

    logging.info(f'Combining indexes from {hdf5_indexes_dir}')
    logging.info(f'Total {len(paths)} hdf5 to combine.')

    start_time_tot = time.time()
    with h5py.File(hdf5_full_index_path, 'w') as full_hf:
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
                logging.info(f'Proicessing file {path}')
                start_time = time.time()
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
                fin_time = time.time()
                logging.info(f"Pack time {fin_time-start_time}")
    fin_time_tot = time.time() 
    logging.info(f'Written combined full hdf5 to {hdf5_full_index_path}')
    logging.info(f'Total pack time: {fin_time_tot-start_time_tot:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_create_indexes = subparsers.add_parser('create_indexes')
    parser_create_indexes.add_argument('--hdf5_path', type=str, required=True,
                                       help='Path of packed waveforms hdf5.')
    parser_create_indexes.add_argument('--hdf5_index_path', type=str, required=True,
                                       help='Path to write out indexes hdf5.')
    parser_create_indexes.add_argument('--logs_dir', type=str, default=None,
                            help='Directory to save logs into.')

    parser_combine_full_indexes = subparsers.add_parser('combine_indexes')
    parser_combine_full_indexes.add_argument('--hdf5_indexes_dir', type=str,
      required=True, help='Directory containing indexes hdf5s to be combined.')
    parser_combine_full_indexes.add_argument('--hdf5_full_index_path', type=str,
            required=True, help='Path to write out full indexes hdf5 file.')
    parser_combine_full_indexes.add_argument('--classes_num', type=int, default=110,
                            help='The amount of classes used in the dataset.')
    parser_combine_full_indexes.add_argument('--logs_dir', type=str, default=None,
                            help='Directory to save logs into.')

    parser_waveforms_to_hdf5 = subparsers.add_parser('wav_to_hdf5')
    parser_waveforms_to_hdf5.add_argument('--audios_dir', type=str, required=True,
                                help='Directory with the  downloaded audio.')
    parser_waveforms_to_hdf5.add_argument('--csv_path', type=str, required=True,
                                help='Path of csv file containing audio info.')
    parser_waveforms_to_hdf5.add_argument('--class_labels_path', type=str, required=True,
                help="Dataset labels in tsv format (as in 'Reformatted' dataset)")
    parser_waveforms_to_hdf5.add_argument('--selected_classes_path', type=str, required=True,
                help="List of all selected classes, one per line")
    parser_waveforms_to_hdf5.add_argument('--hdf5_path', type=str, required=True,
                                            help='Path to save packed hdf5.')
    parser_waveforms_to_hdf5.add_argument('--sample_rate', type=int, default=44100,
                                         help='Sample rate of the used audios.')
    parser_waveforms_to_hdf5.add_argument('--classes_num', type=int, default=110,
                            help='The amount of classes used in the dataset.')
    parser_waveforms_to_hdf5.add_argument('--mini_data', type=int, default=0,
                            help='If specified, use only this many audios.')
    parser_waveforms_to_hdf5.add_argument('--clip_length', type=int, default=10000,
                help='Length (in ms) of audio clips used in the dataset (default 10000)')
    parser_waveforms_to_hdf5.add_argument('--logs_dir', type=str, default=None,
                            help='Directory to save logs into.')
    args = parser.parse_args()
    
    if args.mode == 'create_indexes':
        create_indexes(hdf5_path=args.hdf5_path,
                       hdf5_index_path=args.hdf5_index_path,
                       logs_dir=args.logs_dir)

    elif args.mode == 'combine_indexes':
        combine_indexes(hdf5_indexes_dir=args.hdf5_indexes_dir,
                        hdf5_full_index_path=args.hdf5_full_index_path,
                        classes_num=args.classes_num, logs_dir=args.logs_dir)

    elif args.mode == 'wav_to_hdf5':
        class_ids,_,_,_ = get_labels(args.classes_labels_path, args.selected_classes_path)
        audio_names, target = get_weak_target(args.csv_path, args.class_ids)

        wav_to_hdf5(audios_dir=args.audios_dir,
                          hdf5_path=args.hdf5_path,
                          audio_names=audio_names, target=target,
                          clip_length=args.clip_length,
                          sample_rate=args.sample_rate,
                          classes_num=args.classes_num,
                          mini_data=args.mini_data,
                          logs_dir=args.logs_dir)
    else:
        raise Exception('Incorrect arguments!')
