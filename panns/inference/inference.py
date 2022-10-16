import argparse

import torch
import numpy as np
import pandas as pd
from dcase_util.containers import metadata

from panns.data.loaders import AudioSetDataset, EvaluateSampler, collate_fn
from panns.forward import forward
from panns.utils.file_utils import create_folder
from panns.utils.metadata_utils import get_labels_metadata
import panns.models


def inference(*, eval_indexes_hdf5_path,                                                    
                 checkpoint_path,                                               
                 model_type,                                                    
                 window_size=1024,                                              
                 hop_size=320,                                                  
                 sample_rate=32000,                                             
                 mel_bins=64,                                                   
                 fmin=50, fmax=14000,                                           
                 cuda=False,                                                    
                 classes_num=110,                                               
                 sed=False,
                 num_workers=8, batch_size=32):
    '''Obtain audio tagging or sound event detection results from a model.

    Return either a clipwise_output or framewise_output of a model after
    going through the entire provided dataset. If SED was requested for a model
    that cannot provide framewise_output, automatically switches to AT.

    :param str eval_indexes_hdf5_path: Path to hdf5 index of the evaluation set
    :param str checkpoint_path: Path to the saved checkpoint of the model
                                (as created by panns.train)
    :param str model_type: Name of the model saved in checkpoint
                           (must be one of classes defined in panns.models.models.py)
    :param int window_size: Window size of filter used in training (default 1024)
    :param int hop_size: Hop size of filter used in training (default 320)
    :param int sample_rate: Sample rate of the used audio clips; supported values
                            are 32000, 16000, 8000 (default 32000)
    :param int mel_bins: Amount of mel filters used in the model
    :param int fmin: Minimum frequency used in Logmel filterbank of the model
    :param int fmax: Maximum frequency used in Logmel filterbank of the model
    :param bool cuda: If True, try to use GPU for inference (default False)
    :param int classes_num: Amount of classes used in the dataset (default 110)
    :param bool sed: If True, perform Sound Event Detection, otherwise Audio Tagging
                     (default False)
    :param int num_workers: Amount of workers to pass to torch.utils.data.DataLoader()
                            (default 8)
    :param int batch_size: Batch size to use for evaluation (default 32)
    :return: result - Array of either clipwise or framewise output
    :rtype: numpy.ndarray
    :return: audio_names - Names of audios used in the provided eval dataset
    :rtype: numpy.ndarray
    :raises ValueError: if model_type not found in panns.models.models.py
    '''

                                                                             
    # Model                                                                     
    if model_type in panns.models.__all__:                                            
        Model = eval("panns.models."+model_type)                                                
    else:                                                                       
        raise ValueError(f"'{model_type}' is not among the defined models.")    
                                                                                
    model = Model(sample_rate=sample_rate, window_size=window_size,             
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax,             
        classes_num=classes_num)                                                
                                                                                
    if sed and not model.sed_model:
        print(f"Warning! Asked to perform SED but {model_type} is not a SED model."
              "Performing Audio Tagging instead.")
        sed = False
                                                                                
    device = torch.device('cuda') if (cuda and torch.cuda.is_available()) else torch.device('cpu')
                                                                                
    checkpoint = torch.load(checkpoint_path, map_location=device)               
    model.load_state_dict(checkpoint['model'])
    # Parallel                                                                  
    if device.type == 'cuda':                                                   
        model.to(device)                                                        
        print(f'Using GPU. GPU number: {torch.cuda.device_count()}')            
        sed_model = model.sed_model
        model = torch.nn.DataParallel(model)                                    
        model.sed_model = sed_model
    else:                                                                       
        print('Using CPU.')

    dataset = AudioSetDataset(sample_rate=sample_rate)
    # Evaluate sampler                                                          
    eval_sampler = EvaluateSampler(                                             
        hdf5_index_path=eval_indexes_hdf5_path, batch_size=batch_size)        
    eval_loader = torch.utils.data.DataLoader(dataset=dataset,                  
        batch_sampler=eval_sampler, collate_fn=collate_fn,                      
        num_workers=num_workers, pin_memory=True)

    output_dict = forward(model, eval_loader)

    audio_names = output_dict['audio_name']
    
    if sed:
        result = output_dict['framewise_output']
    else:
        result = output_dict['clipwise_output']

    return result, audio_names


def find_contiguous_regions(activity_array):
    '''Detect blocks of consecutive 1/True values in an activity array (vector).

    :param numpy.ndarray activity_array: An activity array (vector) of 1s and 0s
            (True/False values)
    :return: change_indices - Array of two columns, indicating intervals of
            blocks of consecutive True values
    :rtype: numpy.ndarray
    '''

    # Find the changes in the activity_array
    change_indices = np.logical_xor(activity_array[1:], activity_array[:-1]).nonzero()[0]

    # Shift change_index with one, focus on frame after the change.
    change_indices += 1

    if activity_array[0]:
        # If the first element of activity_array is True add 0 at the beginning
        change_indices = np.r_[0, change_indices]

    if activity_array[-1]:
        # If the last element of activity_array is True, add the length of the array
        change_indices = np.r_[change_indices, len(activity_array)]

    # Reshape the result into two columns
    return change_indices.reshape((-1, 2))


def detect_events(*, frame_probabilities,
                  ix_to_id,
                  filenames,
                  threshold=0.5,
                  minimum_event_length=0.1,
                  minimum_event_gap=0.1,
                  sample_rate=32000,
                  hop_size=320):
    '''Detect Sound Events using a given framewise probability array.

    :param numpy.ndarray frame_probabilities: A two-dimensional array of framewise
        probablities of classes. First dimension corresponds to the classes,
        second to the frames of the audio clip
    :param dict ix_to_id: Dictionary mapping event indexes (from 0 to classes_num-1) used
        to access the event in the frame matrix to their ids (codes)
    :param str filenames: Name of the audio clip to which the frame_probabilities correspond.
    :param float threshold: Threshold used to binarize the frame_probabilites.
        Values higher than the threshold are considered as 'event detected' (default 0.5)
    :param int minimum_event_length: Minimum length (in seconds) of detetected event
        to be considered really present
    :param int minimum_event_gap: Minimum length (in seconds) of a gap between
        two events to distinguish them, if the gap is smaller events are merged
    :param int sample_rate: Sample rate of audio clips used in the dataset (default 32000)
    :param int hop_size: Hop length which was used to obtain the frame_probabilities (default 320)
    '''

    hop_length_seconds = hop_size/sample_rate
    results = []
    activity_array = frame_probabilities > threshold
    change_indices = np.logical_xor(activity_array[:,1:,:], activity_array[:,:-1,:])
    n_files = frame_probabilities.shape[0]
    for f in range(n_files):
        filename = filenames[f]
        for event_ix, event_id in ix_to_id.items():
            event_activity = change_indices[f,:, event_ix].nonzero()[0] + 1

            if activity_array[f,0,event_ix]:
                # If the first element of activity_array is True add 0 at the beginning
                event_activity = np.r_[0, event_activity]

            if activity_array[f,-1,event_ix]:
                # If the last element of activity_array is True, add the length of the array
                event_activity = np.r_[event_activity, activity_array.shape[1]]

            event_activity = event_activity.reshape((-1, 2)) * hop_length_seconds

            # Store events
            if event_activity.size !=0:
                current_onset = event_activity[0][0]
                current_offset = event_activity[0][1]
            for event in event_activity:
                if (minimum_event_length is not None and
                    event[1]-event[0] < minimum_event_length): continue
                if minimum_event_gap is not None:
                    if (event[0] - current_offset >= minimum_event_gap):
                        results.append(metadata.MetaDataItem({'onset': current_onset,
                                                  'offset': current_offset,
                                                  'filename': filename,
                                                  'event_label': event_id}))
                        current_onset = event[0]
                    current_offset = event[1]

                else:
                    results.append(metadata.MetaDataItem({'onset': event[0],
                                                  'offset': event[1],
                                                  'filename': filename,
                                                  'event_label': event_id}))
            if minimum_event_gap is not None and event_activity.size != 0:
                results.append(metadata.MetaDataItem({'onset': current_onset,
                                                  'offset': current_offset,
                                                  'filename': filename,
                                                  'event_label': event_id}))

    results = metadata.MetaDataContainer(results)

    return results


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_indexes_hdf5_path', type=str, required=True,
                        help="Path to hdf5 index of the evaluation set")
    parser.add_argument('--model_type', type=str, required=True,
                        help="Name of model to train")
    parser.add_argument('--checkpoint_path', type=str,
                        help="File to load the NN checkpoint from")
    parser.add_argument('--window_size', type=int, default=1024,
                        help="Window size of filter to be used in training (default 1024)")
    parser.add_argument('--hop_size', type=int, default=320,
                        help="Hop size of filter to be used in traning (default 320)")
    parser.add_argument('--sample_rate', type=int, default=32000,
                        help="Sample rate of the used audio clips; supported values are 32000, 16000, 8000 (default 32000)")
    parser.add_argument('--fmin', type=int, default=50,
                        help="Minimum frequency to be used when creating Logmel filterbank (default 50)")
    parser.add_argument('--fmax', type=int, default=14000,
                        help="Maximum frequency to be used when creating Logmel filterbank (default 14000)")
    parser.add_argument('--mel_bins', type=int, default=64,
                        help="Amount of mel filters to use in the filterbank (default 64)")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size to use for training/evaluation (default 32)")
    parser.add_argument('--cuda', action='store_true', default=False,
                        help="If set, try to use GPU for traning")
    parser.add_argument('--sed', action='store_true', default=False,
                        help='If set, perform Sound Event Detection, otherwise Audio Tagging')
    parser.add_argument('--classes_num', type=int, default=110,
                        help="Amount of classes used in the dataset (default 110)")
    parser.add_argument('--num_workers', type=int, default=8,
                        help="Amount of workers to pass to torch.utils.data.DataLoader (default 8)")
    parser.add_argument('--class_list_path', type=str, required=True)
    parser.add_argument('--class_codes_path', type=str, required=True)

    args = parser.parse_args()

    _,_,_,_,_,ix_to_id = get_labels_metadata(args.class_list_path, args.class_codes_path)
    results, audio_names = inference(eval_indexes_hdf5_path=args.eval_indexes_hdf5_path,
                                     model_type=args.model_type,
                                     checkpoint_path=args.checkpoint_path,
                                     window_size=args.window_size,
                                     hop_size=args.hop_size,
                                     sample_rate=args.sample_rate,
                                     fmin=args.fmin, fmax=args.fmax,
                                     mel_bins=args.mel_bins,
                                     batch_size=args.batch_size,
                                     cuda=args.cuda, sed=args.sed,
                                     classes_num=args.classes_num,
                                     num_workers=args.num_workers)

    events = detect_events(frame_probabilities=results,
                  ix_to_id=ix_to_id,
                  filenames=audio_names,
                  threshold=0.5,
                  minimum_event_length=0.1,
                  minimum_event_gap=0.1,
                  sample_rate=args.sample_rate,
                  hop_size=args.hop_size)

    events.save('events.txt', fields=['filename', 'event_label', 'onset', 'offset'],
                header=True, delimiter='\t')
