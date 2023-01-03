import argparse
import copy
import math
import pickle

import numpy as np

__all__ = ['get_class_labels',
           'get_weak_target',
           'get_strong_target']


def get_class_labels(class_labels_path, selected_classes_path):

    """ Map selected labels from label to id and index and vice versa.

    Args:
        class_labels_path : str,
            Dataset labels in tsv format (in 'Reformatted' format).
        selected_classes_path : str,
            List of class ids selected for training, one per line.

    Returns:
        ids, labels
            -List of all selected classes' ids,
            in the order they were given in 'selected_classes_path'.
            -List of all selected classes' labels,
            in the order they were given in 'selected_classes_path'.
    """

    selected_classes_file = open(selected_classes_path, 'r')
    selected_classes = set()
    for line in selected_classes_file:
        if line.endswith('\n'):
            line = line[:-1]
        selected_classes.add(line)  # TODO - change to .removesuffix() when Python 3.9 is supported
    selected_classes_file.close()

    class_labels_file = open(class_labels_path, 'r')
    labels = []
    ids = []  # Each label has a unique id such as "/m/068hy"
    for line in class_labels_file:
        id_, label = line.split('\t')
        if id_ in selected_classes:
            if label.endswith('\n'):
                label = label[:-1]  # TODO - change to .removesuffix() when Python 3.9 is supported
            ids.append(id_)
            labels.append(label)
    class_labels_file.close()

    return ids, labels


def get_weak_target(data_path, class_ids):
    """ Create weak labels target numpy array.

    Args:
        data_path : str,
            Dataset file to create weak target from (in 'Reformatted' format).
        class_ids : list[str],
            List of class ids, index in the list will correspond to the index
            in the target array.

    Returns:
        audio_names, target
            -List of all files from data_path, index in this list corresponds
            to the index in the target array.
            -Target array of weak labels with shape (videos, classes).
    """

    class_id_to_ix = {id_: ix for ix, id_ in enumerate(class_ids)}
    zero_vector = [0.0] * len(class_ids)
    target = []
    audio_names = []
    count = 0
    video_id_to_ix = dict()
    file = open(data_path, 'r')
    for line in file:
        parts = line.split('\t')
        video_id = parts[0]
        if video_id == 'filename': continue
        label = parts[1]
        if label.endswith('\n'):
            label = label[:-1]  # TODO - change to .removesuffix() when Python 3.9 is supported

        if video_id not in video_id_to_ix:
            video_id_to_ix[video_id] = count
            count += 1
            target.append(copy.deepcopy(zero_vector))
            audio_names.append(video_id)

        video_ix = video_id_to_ix[video_id]
        class_ix = class_id_to_ix[label]

        target[video_ix][class_ix] = 1.0
    file.close()

    target = np.array(target, dtype=np.bool)

    return audio_names, target


def get_strong_target(data_path, class_ids, sample_rate, hop_length,
                      clip_length):

    hop_length_seconds = hop_length/sample_rate
    frames_num = int((clip_length/1000)/hop_length_seconds)+1

    class_id_to_ix = {id_: ix for ix, id_ in enumerate(class_ids)}
    zero_vector = [[0.0] * frames_num] * len(class_ids)
    target = []
    audio_names = []
    count = 0
    video_id_to_ix = dict()
    file = open(data_path, 'r')
    for line in file:
        parts = line.split('\t')
        video_id = parts[0]
        if video_id == 'filename': continue
        label = parts[1]
        onset = parts[2]
        offset = parts[3]
        if label.endswith('\n'):
            label = label[:-1]  # TODO - change to .removesuffix() when Python 3.9 is supported
        if onset.endswith('\n'):
            onset = onset[:-1]  # TODO - change to .removesuffix() when Python 3.9 is supported
        if offset.endswith('\n'):
            offset = offset[:-1]  # TODO - change to .removesuffix() when Python 3.9 is supported
        onset = float(onset)
        onset = math.floor(onset/hop_length_seconds)
        offset = float(offset)
        offset = math.ceil(offset/hop_length_seconds)

        if video_id not in video_id_to_ix:
            video_id_to_ix[video_id] = count
            count += 1
            target.append(copy.deepcopy(zero_vector))
            audio_names.append(video_id)

        video_ix = video_id_to_ix[video_id]
        class_ix = class_id_to_ix[label]

        target[video_ix][class_ix][onset:offset] = [1.0]*(offset-onset)
    file.close()

    target = np.array(target, dtype=np.bool)
    target = np.transpose(target, (0, 2, 1))

    return audio_names, target


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--class_labels_path', type=str, required=True,
                        help="Dataset labels in tsv format (in 'Reformatted' "
                             "format)")
    parser.add_argument('--selected_classes_path', type=str, required=True,
                        help="List of class ids selected for training, "
                             "one per line")
    parser.add_argument('--data_path', type=str, required=True,
                        help="Dataset file to create weak target from (in "
                             "'Reformatted' format)")
    parser.add_argument('--target_type', type=str, required=False, choices=[
                        'weak', 'strong'], default='weak',
                        help='Whether to create a weak or '
                        'strong label tensor (strong also requires '
                        'sample_rate, hop_length and clip-length parameters)')
    parser.add_argument('--hop_length', type=int, default=None, required=False,
                        help="Hop size of filter to be used in training")
    parser.add_argument('--sample_rate', type=int, default=None, required=False,
                        help="Sample rate of the used audio clips")
    parser.add_argument('--clip_length', type=int, default=None, required=False,
                        help="Length (in ms) of audio clips used in the "
                             "dataset")
    parser.add_argument('--audio_names_path', type=str, default='audio_names.npy',
                        help="Path to save the audio_names numpy array"
                             " (defaults to 'audio_names.pickle' in CWD)")
    parser.add_argument('--target_path', type=str, default='target.npy',
                        help="Path to save the target numpy array"
                             " (defaults to 'target.npy' in CWD)")
    args = parser.parse_args()

    ids, _ = get_class_labels(args.class_labels_path, args.selected_classes_path)

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
        audio_names, target = get_strong_target(args.data_path, ids,
                                                args.sample_rate,
                                                args.hop_length,
                                                args.clip_length)
    else:
        audio_names, target = get_weak_target(args.data_path, ids)
    with open(args.audio_names_path, 'wb') as f:
        pickle.dump(audio_names, f)

    np.save(args.target_path, target)
