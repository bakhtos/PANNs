import os
import numpy as np
import copy

__all__ = ['get_labels_metadata',
           'read_metadata']

def get_labels_metadata(class_list_path, class_codes_path):
    # Load label
    class_list_file = open(class_list_path, 'r')
    selected_classes = set()
    for line in class_list_file:
        selected_classes.add(line[:-1])
    class_list_file.close()
    
    class_codes_file = open(class_codes_path,  'r')
    labels = []
    ids = []    # Each label has a unique id such as "/m/068hy"
    for line in class_codes_file:
        code, label = line.split('\t')
        if code in selected_classes:
            ids.append(code)
            labels.append(label)
    class_codes_file.close()

    lb_to_ix = dict()
    ix_to_lb = dict()
    id_to_ix = dict()
    ix_to_id = dict()
    
    for i, (label, code) in enumerate(zip(labels,ids)):
        lb_to_ix[label] = i
        ix_to_lb[i] = label
        id_to_ix[code] = i
        ix_to_id[i] = code 

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
