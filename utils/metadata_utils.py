import os
import numpy as np
import copy


def get_labels_metadata():
    # Load label
    selected_classes_path = os.path.join('metadata', 'selected_classes.txt')
    file_classes = open(selected_classes_path, 'r')
    selected_classes = set()
    for line in file_classes:
        selected_classes.add(line[:-1])
    file_classes.close()
    
    mid_to_disp = os.path.join('metadata', 'mid_to_display_name.tsv')
    file_mid = open(mid_to_disp,  'r')
    labels = []
    ids = []    # Each label has a unique id such as "/m/068hy"
    for line in file_mid:
        code, label = line.split('\t')
        if code in selected_classes:
            ids.append(code)
            labels.append(label)
    file_mid.close()

    lb_to_ix = dict()
    ix_to_lb = dict()
    
    for i, label in enumerate(labels):
        lb_to_ix[label] = i
        ix_to_lb[i] = label

    id_to_ix = dict()
    ix_to_id = dict()
    for i, code in enumerate(ids):
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
