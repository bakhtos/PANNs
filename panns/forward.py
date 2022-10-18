import torch
import numpy as np

from panns.utils.pytorch_utils import move_data_to_device, append_to_dict

__all__ = ['forward']

def forward(model, generator, return_input=False, 
    return_target=False):
    """Forward data to a model.
    
    Args: 
      model: object
      generator: object
      return_input: bool
      return_target: bool

    Returns:
      audio_name: (audios_num,)
      clipwise_output: (audios_num, classes_num)
      (ifexist) segmentwise_output: (audios_num, segments_num, classes_num)
      (ifexist) framewise_output: (audios_num, frames_num, classes_num)
      (optional) return_input: (audios_num, segment_samples)
      (optional) return_target: (audios_num, classes_num)
    """
    output_dict = {}
    device = next(model.parameters()).device

    # Forward data to a model in mini-batches
    for n, batch_data_dict in enumerate(generator):
        batch_waveform = move_data_to_device(batch_data_dict['waveform'], device)
        
        with torch.no_grad():
            model.eval()
            #  Second_output is either 'embedding' (non-sed model) or 'framewise_output' (sed model)
            clipwise_output, second_output = model(batch_waveform)

        append_to_dict(output_dict, 'audio_name', batch_data_dict['audio_name'])

        append_to_dict(output_dict, 'clipwise_output', 
            clipwise_output.data.cpu().numpy())

        if model.sed_model:
            append_to_dict(output_dict, 'framewise_output',
                           second_output.data.cpu().numpy())
        else:
            append_to_dict(output_dict, 'embedding',
                           second_output.data.cpu().numpy())
            
        if return_input:
            append_to_dict(output_dict, 'waveform', batch_data_dict['waveform'])
            
        if return_target:
            if 'target' in batch_data_dict.keys():
                append_to_dict(output_dict, 'target', batch_data_dict['target'])

    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)

    return output_dict
