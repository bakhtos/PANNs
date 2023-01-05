import argparse
import math

import numpy as np
import pandas as pd

__all__ = ['get_weak_target',
           'get_strong_target']


def get_weak_target(data):
    """ Create weak labels target numpy array.

    Args:
        data: str,
            Dataframe representing the loaded dataset (loaded from
            'Reformatted'-like tsv file)

    Returns:
        target : Target array of weak labels with shape (files, classes).
    """

    file_ids = data['filename'].unique()
    class_ids = data['event_label'].unique()

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

        target[file_ix][class_ix] = True

    return target


def get_strong_target(data, *, sample_rate, hop_length, clip_length):
    """

    Args:
        data: pandas.DataFrame,
            Dataframe representing the loaded dataset (loaded from
            'Reformatted'-like tsv file)
        sample_rate: int,
            Sample rate of the used audios.
        hop_length: int,
            Hop length of the window used during Spectrogram extraction.
        clip_length: int,
            Length of used audios (in ms).

    Returns:
        target : Target array of weak labels with shape (files, frames, classes).
    """

    hop_length_seconds = hop_length/sample_rate
    frames_num = int((clip_length/1000)/hop_length_seconds)
    file_ids = data['filename'].unique()
    class_ids = data['event_label'].unique()

    target = np.zeros((file_ids.size, class_ids.size, frames_num), dtype=bool)

    file_id_to_ix = {}
    for i in range(file_ids.size):
        file_id_to_ix[file_ids[i]] = i
    class_id_to_ix = {}
    for i in range(class_ids.size):
        class_id_to_ix[class_ids[i]] = i

    for line in data.itertuples(index=False):
        file_id = line.filename
        class_id = line.event_label
        onset = line.onset
        offset = line.offset

        onset = float(onset)
        onset = math.floor(onset/hop_length_seconds)
        offset = float(offset)
        offset = math.ceil(offset/hop_length_seconds)

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
        data = pd.read_csv(args.dataset_path, delimiter='\t')
        target = get_strong_target(data,
                                   sample_rate=args.sample_rate,
                                   hop_length=args.hop_length,
                                   clip_length=args.clip_length)
    else:
        data = pd.read_csv(args.dataset_path, delimiter='\t')
        target = get_weak_target(data)

    np.save(args.target_path, target)
