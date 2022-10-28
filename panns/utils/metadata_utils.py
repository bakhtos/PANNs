import copy

import numpy as np

__all__ = ['get_labels',
           'get_weak_target']


def get_labels(class_labels_path, selected_classes_path):
    """ Map selected labels from label to id and index and vice versa.

    Parameters
    __________

    class_labels_path : str,
        Dataset labels in tsv format (as in 'Reformatted' dataset).
    selected_classes_path : str,
        List of class ids selected for training, one per line.

    Returns
    _______

    ids : list,
        List of all selected classes' ids,
        in the order they were given in 'selected_classes_path'.
    labels : list,
        List of all selected classes' labels,
        in the order they were given in 'selected_classes_path'.
    lb_to_id : dict[str] -> int,
        Map from selected classes' labels to their ids.
    id_to_lb : dict[str] -> int,
        Map from selected classes' ids to their labels.
    """

    selected_classes_file = open(selected_classes_path, 'r')
    selected_classes = set()
    for line in selected_classes_file:
        selected_classes.add(line.removesuffix('\n'))
    selected_classes_file.close()

    class_labels_file = open(class_labels_path, 'r')
    labels = []
    ids = []  # Each label has a unique id such as "/m/068hy"
    for line in class_labels_file:
        id_, label = line.split('\t')
        if id_ in selected_classes:
            label = label.removesuffix('\n')
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

    Parameters
    __________

    data_path : str,
        Dataset file to create target from (in 'Reformatted' format).
    class_ids : list[str],
        List of class ids, index in the list will correspond to the index
        in the target array.

    Returns
    _______

    audio_names : numpy.array[str],
        List of all files from data_path, index in this list corresponds
        to the index in the target array.
    target : numpy.array[int],
        Target array of weak labels with shape (videos, classes).
    """

    id_to_ix = {id_: ix for ix, id_ in enumerate(class_ids)}
    zero_vector = [0] * len(class_ids)
    target = []
    audio_names = []
    count = 0
    video_id_to_ix = dict()
    file = open(data_path, 'r')
    for line in file:
        parts = line.split('\t')
        video_id = parts[0]
        if video_id == 'filename': continue
        label = parts[1].removesuffix('\n')

        if video_id not in video_id_to_ix:
            video_id_to_ix[video_id] = count
            count += 1
            target.append(copy.deepcopy(zero_vector))
            audio_names.append(video_id)

        video_ix = video_id_to_ix[video_id]
        class_ix = id_to_ix[label]

        target[video_ix][class_ix] = 1
    file.close()

    return np.array(audio_names), np.array(target)
