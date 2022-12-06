import argparse
import copy
import math

import numpy as np

__all__ = ['get_labels',
           'get_weak_target',
           'get_strong_target']


def get_labels(class_labels_path, selected_classes_path):

    """ Map selected labels from label to id and index and vice versa.

    Args:
        class_labels_path : str,
            Dataset labels in tsv format (in 'Reformatted' format).
        selected_classes_path : str,
            List of class ids selected for training, one per line.

    Returns:
        ids, labels, lb_to_id, id_to_lb
            -List of all selected classes' ids,
            in the order they were given in 'selected_classes_path'.
            -List of all selected classes' labels,
            in the order they were given in 'selected_classes_path'.
            -Map from selected classes' labels to their ids.
            -Map from selected classes' ids to their labels.
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

    lb_to_id = dict()
    id_to_lb = dict()

    for label, id_ in zip(labels, ids):
        lb_to_id[label] = id_
        id_to_lb[id_] = label

    return ids, labels, lb_to_id, id_to_lb


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

    return np.array(audio_names), np.array(target)


def get_strong_target(data_path, class_ids, sample_rate, hop_length,
                      clip_length):

    hop_length_seconds = hop_length/sample_rate
    frames_num = int((clip_length/1000)/hop_length_seconds)+1

    class_id_to_ix = {id_: ix for ix, id_ in enumerate(class_ids)}
    zero_vector = [[0.0] * len(class_ids)]*frames_num
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

        target[video_ix][onset:offset][class_ix] = 1.0
    file.close()

    return np.array(audio_names), np.array(target)


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
    parser.add_argument('--audio_names_path', type=str, default='audio_names.npy',
                        help="Path to save the audio_names numpy array"
                             " (defaults to 'audio_names.npy' in CWD)")
    parser.add_argument('--target_weak_path', type=str, default='target_weak.npy',
                        help="Path to dave the weak target numpy array"
                             " (defaults to 'target_weak.npy' in CWD)")
    args = parser.parse_args()

    ids, _, _, _ = get_labels(args.class_labels_path, args.selected_classes_path)
    audio_names, target_weak = get_weak_target(args.data_path, ids)
    np.save(args.audio_names_path, audio_names)
    np.save(args.target_weak_path, target_weak)
