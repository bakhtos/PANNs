import argparse
import math

import numpy as np
import pandas as pd

__all__ = ['get_weak_target',
           'get_strong_target']


def get_weak_target(data_path):
    """ Create weak labels target numpy array.

    Args:
        data_path : str,
            Dataset file to create weak target from (in 'Reformatted' format).

    Returns:
        target : Target array of weak labels with shape (files, classes).
    """

    target = np.zeros((0, 0), dtype=bool)
    file_id_to_ix = dict()
    class_id_to_ix = dict()
    data = pd.read_csv(data_path, delimiter='\t')
    for line in data.itertuples(index=False):
        file_id = line.filename
        class_id = line.event_label

        if file_id not in file_id_to_ix:
            file_id_to_ix[file_id] = len(file_id_to_ix)
            target = np.concatenate((target, np.zeros((1, target.shape[1]),
                                                      dtype=bool)), axis=0)

        if class_id not in class_id_to_ix:
            class_id_to_ix[class_id] = len(class_id_to_ix)
            target = np.concatenate((target, np.zeros((target.shape[0], 1),
                                                      dtype=bool)), axis=1)

        file_ix = file_id_to_ix[file_id]
        class_ix = class_id_to_ix[class_id]

        target[file_ix][class_ix] = True

    return target


def get_strong_target(data_path, *, sample_rate, hop_length, clip_length):

    hop_length_seconds = hop_length/sample_rate
    frames_num = int((clip_length/1000)/hop_length_seconds)

    target = np.zeros((0, 0, frames_num), dtype=bool)
    file_id_to_ix = {}
    class_id_to_ix = {}
    data = pd.read_csv(data_path, delimiter='\t')
    for line in data.itertuples(index=False):
        file_id = line.filename
        class_id = line.event_label
        onset = line.onset
        offset = line.offset

        onset = float(onset)
        onset = math.floor(onset/hop_length_seconds)
        offset = float(offset)
        offset = math.ceil(offset/hop_length_seconds)

        if file_id not in file_id_to_ix:
            file_id_to_ix[file_id] = len(file_id_to_ix)
            target = np.concatenate((target, np.zeros((1, target.shape[1],
                                                       frames_num),
                                                      dtype=bool)), axis=0)
        if class_id not in class_id_to_ix:
            class_id_to_ix[class_id] = len(class_id_to_ix)
            target = np.concatenate((target, np.zeros((target.shape[0],
                                                       1, frames_num),
                                                      dtype=bool)), axis=1)

        file_ix = file_id_to_ix[file_id]
        class_ix = class_id_to_ix[class_id]

        target[file_ix][class_ix][onset:offset] = True

    target = np.transpose(target, (0, 2, 1))

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
        target = get_strong_target(args.data_path,
                                   sample_rate=args.sample_rate,
                                   hop_length=args.hop_length,
                                   clip_length=args.clip_length)
    else:
        target = get_weak_target(args.data_path)

    np.save(args.target_path, target)
