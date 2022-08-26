import torch
import numpy as np
import pandas as pd
from dcase_util.containers import metadata

from panns.data.loaders import AudioSetDataset, EvaluationSampler, collate_fn
from panns.forward import forward
from panns.utils.file_utils import create_folder
from panns.models import *


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
    :raises ValueError: if model_type not found in panns.models.models.py
    '''

                                                                             
    # Model                                                                     
    if model_type in models.__all__:                                            
        Model = eval(model_type)                                                
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
        model = torch.nn.DataParallel(model)                                    
    else:                                                                       
        print('Using CPU.')

    dataset = AudioSetDataset(sample_rate=sample_rate)
    # Evaluate sampler                                                          
    eval_sampler = EvaluateSampler(                                             
        indexes_hdf5_path=eval_indexes_hdf5_path, batch_size=batch_size)        
    eval_loader = torch.utils.data.DataLoader(dataset=dataset,                  
        batch_sampler=eval_sampler, collate_fn=collate_fn,                      
        num_workers=num_workers, pin_memory=True)

    output_dict = forward(model, eval_loader)
    
    if sed:
        result = output_dict['framewise_output']
    else:
        result = output_dict['clipwise_output']


def find_contiguous_regions(activity_array):

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
                  class_labels,
                  filename,
                  threshold=0.5,
                  minimum_event_length=0.1,
                  minimum_event_gap=0.1,
                  sample_rate=32000,
                  hop_length=320):

    hop_length_seconds = hop_length/sample_rate
    results = []
    for event_ix, event_label in ix_to_lb.items():
        # Binarization
        event_activity = frame_probabilities[event_ix, :] > threshold

        # Convert active frames into segments and translate frame indices into time stamps
        event_segments = find_contiguous_regions(event_activity) * hop_length_seconds

        # Store events
        for event in event_segments:
            results.append(metadata.MetaDataItem({'onset': event[0],
                                                  'offset': event[1],
                                                  'filename': filename,
                                                  'event_label': event_label}))

    results = metadata.MetaDataContainer(results)

    # Event list post-processing
    results = results.process_events(minimum_event_length=minimum_event_length,
                                     minimum_event_gap=minimum_event_gap)
    return results


if __name__ == '__main__':

    _,labels,_,_,_,_ = get_labels_metadata()
