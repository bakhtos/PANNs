import torch
import numpy as np

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
                 sed=False, verbose=False,
                 num_workers=8, batch_size=32):


    _,labels,_,_,_,_ = get_labels_metadata()                                    
                                                                             
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
