import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import argparse

import numpy as np
import librosa.load, librosa.stft
import matplotlib.pyplot as plt
import torch

from file_utils import create_folder, get_filename
from models import *
from pytorch_utils import move_data_to_device


def inference(*, audio_path,
                 checkpoint_path,
                 model_type,
                 window_size=1024,
                 hop_size=320,
                 sample_rate=32000,
                 mel_bins=64,
                 fmin=50, fmax=14000,
                 cuda=False,
                 classes_num=110,
                 sed=False, verbose=False):
    """Perform Audio Tagging or Sound Event Detection of an audio clip with a pre-trained model.

    If used with verbose=True, will also print the top-10 classes and save a figure
    of SED results in './figures/AUDIO_NAME.png', otherwise simply returns
    the model output and list of class labels.

    :param str audio_path: WAV audiofile to be used for inference
    :param str checkpoint_path: Read a checkpoint to be used for inference from this path
    :param str model_type: Name of model to train (one of the model classes defined in models.py)
    :param int window_size: Window size of filter to be used in training (default 1024)
    :param int hop_size: Hop size of filter to be used in traning (default 320)
    :param int sample_rate: Sample rate of the used audio clips (default 32000)
    :param int mel_bins: Amount of mel filters to use in the filterbank (default 64)
    :param int fmin: Minimum frequency to be used when creating Logmel filterbank (default 50)
    :param int fmax: Maximum frequency to be used when creating Logmel filterbank (default 14000)
    :param bool cuda: If True, try to use GPU for traning (default False)
    :param int classes_num: Amount of classes used in the dataset (default 110)
    :param bool sed: If True, perform SED, otherwise Audio Tagging (default False)
    :param bool verbose: If True, print top 10 classes and save a SED figure (default False)
    :return result: Either clipwise or framewise output of the model (for AT/SED respectively)
    :rtype numpy.ndarray:
    :return labels: List of labels in order used in the resulting matrix
    :rtype list[str]:
    :raises ValueError: If model_type not found in defined models
    """
    
    _,labels,_,_,_,_ = get_labels_metadata()

    # Model
    if model_type in models.__all__:
        Model = eval(model_type)
    else:
        raise ValueError(f"'{model_type}' is not among the defined models.")

    model = Model(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num)
    
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
    
    # Load audio
    waveform, _ = librosa.load(audio_path, sr=sample_rate, mono=True)

    waveform = waveform[None, :]    # (1, audio_length)
    waveform = move_data_to_device(waveform, device)

    # Forward
    with torch.no_grad():
        model.eval()
        # framewise_output can be 'embedding' if non-sed model is used
        clipwise_output, framewise_output = model(waveform, None)

    if sed:
        result = framewise_output.data.cpu().numpy()[0] # shape = (time_steps, classes_num)
        if verbose:
            sorted_indexes = np.argsort(np.max(result, axis=0))[::-1]
            top_result_mat = result[:, sorted_indexes[0:10]]
            stft = librosa.stft(y=waveform[0].data.cpu().numpy(), n_fft=window_size,
                hop_length=hop_size, window='hann', center=True)
            frames_num = stft.shape[-1]
            frames_per_second = sample_rate // hop_size


            fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 4))
            axs[0].matshow(np.log(np.abs(stft)), origin='lower', aspect='auto', cmap='jet')
            axs[0].set_ylabel('Frequency bins')
            axs[0].set_title('Log spectrogram')
            axs[1].matshow(top_result_mat.T, origin='upper', aspect='auto', cmap='jet', vmin=0, vmax=1)
            axs[1].xaxis.set_ticks(np.arange(0, frames_num, frames_per_second))
            axs[1].xaxis.set_ticklabels(np.arange(0, frames_num / frames_per_second))
            axs[1].yaxis.set_ticks(np.arange(0, top_k))
            axs[1].yaxis.set_ticklabels(np.array(labels)[sorted_indexes[0 : top_k]])
            axs[1].yaxis.grid(color='k', linestyle='solid', linewidth=0.3, alpha=0.3)
            axs[1].set_xlabel('Seconds')
            axs[1].xaxis.set_ticks_position('bottom')
            plt.tight_layout()
            create_folder('figures')

            fig_path = os.path.join('figures', get_filename(audio_path)+'.png')
            print(f'Save sound event detection visualization of {audio_path} to {fig_path}.')
            plt.savefig(fig_path)

    else:
        result = clipwise_output.data.cpu().numpy()[0] # shape = (classes_num,)
        if verbose:
            sorted_indexes = np.argsort(result)[::-1]
            # Print audio tagging top probabilities
            for k in range(10):
                print(labels[sorted_indexes[k]], f"{result[sorted_indexes[k]]}:.3f")

    return result, labels


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', type=str, required=True,
                        help='WAV audiofile to be analyzed.')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Model checkpoint to use for inference.')
    parser.add_argument('--model_type', type=str, required=True,
                        help='Name of class which model is used.')
    parser.add_argument('--window_size', type=int, default=1024,
                        help='Window size of filter to be used in training (default 1024)')
    parser.add_argument('--hop_size', type=int, default=320,
                        help='Hop size of filter to be used in training (default 320)')
    parser.add_argument('--sample_rate', type=int, default=32000,
                        help='Sample rate of used audio clip (default 32000)')
    parser.add_argument('--mel_bins', type=int, default=64,
                        help='Amount of mel filters to use in the filterbank (default 64)')
    parser.add_argument('--fmin', type=int, default=50,
                        help='Minimum frequency to be used when creating Logmel filterbank (default 50)')
    parser.add_argument('--fmax', type=int, default=14000,
                        help='Maximum frequency to be used when creating Logmel filterbank (default 14000)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='If set, try to use GPU for inference')
    parser.add_argument('--classes_num', type=int, default=110,
                        help='Amount of classes used in the dataset (default 110)')
    parser.add_argument('--sed', action='store_true', default=False,
                        help='If set, perform Sound Event Detection, otherwise perform Audio Tagging')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='If set, print top-10 results and make a figure for SED')

    args = parser.parse_args()

    inference(audio_path=args.audio_path,
              checkpoint_path=args.checkpoint_path,
              model_type=args.model_type,
              window_size=args.window_size,
              hop_size=args.hop_size,
              sample_rate=args.sample_rate,
              mel_bins=args.mel_bins,
              fmin=args.fmin, fmax=args.fmax,
              cuda=args.cuda,
              classes_num=args.classes_num,
              sed=args.sed, verbose=args.verbose)
