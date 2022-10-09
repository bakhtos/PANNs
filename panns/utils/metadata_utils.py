import os
import numpy as np
import copy

__all__ = ['get_labels',
           'read_metadata']

def get_labels(class_labels_path, selected_classes_path):
    ''' Map selected labels from label to id and index and vice versa.

    Parameters
    __________

    class_labels_path : str,
        Dataset labels in tsv format (as in 'Reformatted' dataset').
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
    lb_to_ix : dict[str] -> int,
        Map from selected classes' labels to their index in 'labels'.
    ix_to_lb : dict[int] -> str,
        Map from selected classes' index in 'labels' to the class label.
    id_to_ix : dict[str] -> int,
        Map from selected classes' ids to their index in 'ids'.
    ix_to_id : dict[int] -> str,
        Map from selected classes' index in 'ids' to the class id.
    '''

    selected_classes_file = open(selected_classes, 'r')
    selected_classes = set()
    for line in selected_classes_file:
        selected_classes.add(line.removesuffix('\n'))
    selected_classes_file.close()
    
    class_labels_file = open(class_labels_path,  'r')
    labels = []
    ids = []    # Each label has a unique id such as "/m/068hy"
    for line in class_labels_file:
        id_, label = line.split('\t')
        if id_ in selected_classes:
            label = label.removesuffix('\n')
            ids.append(id_)
            labels.append(label)
    class_labels_file.close()

    lb_to_ix = dict()
    ix_to_lb = dict()
    id_to_ix = dict()
    ix_to_id = dict()
    
    for i, (label, id_) in enumerate(zip(labels,ids)):
        lb_to_ix[label] = i
        ix_to_lb[i] = label
        id_to_ix[id_] = i
        ix_to_id[i] = id_ 

    return ids, labels, lb_to_ix, ix_to_lb, id_to_ix, ix_to_id


def read_metadata(csv_path, classes_num, id_to_ix):
    """Read metadata of AudioSet from a csv file.  """

    zero_vector = [0]*classes_num
    targets = []
    audio_names = []
    count = 0  
    video_id_to_ix = dict()
    file = open(csv_path, 'r')
    for line in file:
        video_id, label = line.split('\t')
        label=label[:-1]

        if video_id not in video_id_to_ix:
            video_id_to_ix[video_id] = count
            count +=1
            targets.append(copy.deepcopy(zero_vector))
            audio_name = 'Y'+video_id+'.wav'
            audio_names.append(audio_name)

        video_ix = video_id_to_ix[video_id]
        class_ix = id_to_ix[label]
        
        targets[video_ix][class_ix] = 1
    file.close()

    return np.array(audio_names), np.array(targets)
