import argparse
import math
import os
import time
import logging

import numpy as np
import pandas as pd

import panns.base_logging
TARGET_LOGGER = logging.getLogger('panns')

__all__ = ['get_target']


def get_target(data, target_type, *, sample_rate=None, hop_length=None,
               clip_length=None):
    """

    Args:
        data: pandas.DataFrame,
            Dataframe representing the loaded dataset (loaded from
            'Reformatted'-like tsv file)
        target_type: str,
            Whether to create a 'weak' or 'strong' target array.
        sample_rate: int,
            Sample rate of the used audios (for strong target, default None)
        hop_length: int,
            Hop length of the window used during Spectrogram extraction (for
            strong target, default None)
        clip_length: int,
            Length of used audios in ms (for strong target, default None)

    Returns:
        target : Target array of either weak labels with shape (files, classes)
                 or strong labels with shape (files, frames, classes)
    """

    assert target_type in ['strong', 'weak']

    TARGET_LOGGER.info("Constructing target tensor")
    start_time = time.time()
    file_ids = data['filename'].unique()
    class_ids = data['event_label'].unique()

    if target_type == 'strong':
        hop_length_seconds = hop_length/sample_rate
        frames_num = int((clip_length/1000)/hop_length_seconds)
        target = np.zeros((file_ids.size, class_ids.size, frames_num), dtype=bool)
    else:
        target = np.zeros((file_ids.size, class_ids.size), dtype=bool)

    file_id_to_ix = {}
    for i in range(file_ids.size):
        file_id_to_ix[file_ids[i]] = i
    class_id_to_ix = {}
    for i in range(class_ids.size):
        class_id_to_ix[class_ids[i]] = i

    for line in data.itertuples(index=False):
        file_id = line.filename
        class_id = line.event_label
        file_ix = file_id_to_ix[file_id]
        class_ix = class_id_to_ix[class_id]

        if target_type == 'strong':
            onset = float(line.onset)
            offset = float(line.offset)
            TARGET_LOGGER.info(f"{file_id} - {class_id} - {onset}:{offset}")
            onset = math.floor(onset/hop_length_seconds)
            offset = math.ceil(offset/hop_length_seconds)
            target[file_ix][class_ix][onset:offset] = True
        else:
            TARGET_LOGGER.info(f"{file_id} - {class_id}")
            target[file_ix][class_ix] = True

    if target_type == 'strong':
        target = np.transpose(target, (0, 2, 1))

    fin_time = time.time()
    TARGET_LOGGER.info(f"Target tensor construction finished; time: {fin_time-start_time:.3f} s")

    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, required=True,
                        help="Path to the dataset tsv file")
    parser.add_argument('--target_type', type=str, required=False, choices=[
                        'weak', 'strong'], default='weak',
                        help='Whether to create a weak or '
                        'strong label tensor (strong also requires '
                        'sample_rate, hop_length and clip-length parameters)')
    parser.add_argument('--target_path', type=str, default='target.npy',
                        help="Path to save the target numpy array"
                             " (defaults to 'target.npy' in CWD)")
    parser.add_argument('--hop_length', type=int, default=None, required=False,
                        help="Hop size of filter to be used in training")
    parser.add_argument('--sample_rate', type=int, default=None, required=False,
                        help="Sample rate of the used audio clips")
    parser.add_argument('--clip_length', type=int, default=None, required=False,
                        help="Length (in ms) of audio clips used in the "
                             "dataset")
    args = parser.parse_args()

    if args.target_type == 'strong':
        if args.sample_rate is None:
            raise AttributeError("Strong label target was requested, but no"
                                 "sample_rate given")
        if args.hop_length is None:
            raise AttributeError("Strong label target was requested, but no"
                                 "hop_length given")
        if args.clip_length is None:
            raise AttributeError("Strong label target was requested, but no"
                                 "clip_length given")
    data = pd.read_csv(args.dataset_path, delimiter='\t')
    target = get_target(data, target_type=args.target_type,
                        sample_rate=args.sample_rate,
                        hop_length=args.hop_length,
                        clip_length=args.clip_length)

    dir_name = os.path.dirname(args.target_path)
    if dir_name != '':
        os.makedirs(dir_name, exist_ok=True)

    np.save(args.target_path, target)
