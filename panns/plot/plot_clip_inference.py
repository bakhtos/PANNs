import os
import argparse

import numpy as np
import librosa
import matplotlib.pyplot as plt
import torch

import panns.models
from panns.data.metadata import get_class_labels


@torch.no_grad()
def plot_clip_inference(*, model,
                        audio_path,
                        labels,
                        win_length=1024,
                        hop_length=320,
                        sample_rate=32000,
                        top_k=10,
                        cuda=False,
                        sed=False):
    """Perform Audio Tagging or Sound Event Detection of an audio clip with a pre-trained model.

    The figure of top classes' detections drawn on top of spectrogram is
    saved next to the audio.

    Parameters
    __________
    model : torch.nn.Module subclass,
        Model object to use
    audio_path : str,
        WAV audiofile to be used for inference
    labels: numpy.ndarray,
        Array of class labels in the same order they are store in model
    win_length : int,
        Length of the Hanning window used for spectrogram extraction
    hop_length : int,
        Hop length of the Hanning window used for spectrogram extraction
    sample_rate : int,
        Sample rate of the used audio clips (default 32000)
    top_k : int,
        Use that much top-rated classes in outputs
    cuda : bool,
        If True, try to use GPU for training (default False)
    sed : bool,
        If True, perform SED, otherwise Audio Tagging (default False)
    """

    device = torch.device('cuda') if (
                cuda and torch.cuda.is_available()) else torch.device('cpu')

    # Parallel
    if device.type == 'cuda':
        model.to(device)
        print(f'Using GPU. GPU number: {torch.cuda.device_count()}')
        model = torch.nn.DataParallel(model)
    else:
        print('Using CPU.')

    # Load audio
    waveform, _ = librosa.load(audio_path, sr=sample_rate, mono=True)

    model_input = torch.Tensor(waveform[None, :], device=device)

    # Forward
    model.eval()
    # framewise_output can be 'embedding' if non-sed model is used
    clipwise_output, framewise_output = model(model_input, None)

    if sed:
        result = framewise_output.data.cpu().numpy()[
            0]  # shape = (time_steps, classes_num)
        sorted_indexes = np.argsort(np.max(result, axis=0))[::-1]
        top_result_mat = result[:, sorted_indexes[0:top_k]]
        stft = librosa.stft(y=waveform, n_fft=win_length,
                            hop_length=hop_length, window='hann', center=True)
        frames_num = stft.shape[-1]
        frames_per_second = sample_rate // hop_length

        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 4))
        axs[0].matshow(np.log(np.abs(stft)), origin='lower', aspect='auto',
                       cmap='jet')
        axs[0].set_ylabel('Frequency bins')
        axs[0].set_title('Log spectrogram')
        axs[1].matshow(top_result_mat.T, origin='upper', aspect='auto',
                       cmap='jet', vmin=0, vmax=1)
        axs[1].xaxis.set_ticks(np.arange(0, frames_num, frames_per_second))
        axs[1].xaxis.set_ticklabels(
            np.arange(0, frames_num / frames_per_second))
        axs[1].yaxis.set_ticks(np.arange(0, top_k))
        axs[1].yaxis.set_ticklabels(np.array(labels)[sorted_indexes[0: top_k]])
        axs[1].yaxis.grid(color='k', linestyle='solid', linewidth=0.3,
                          alpha=0.3)
        axs[1].set_xlabel('Seconds')
        axs[1].xaxis.set_ticks_position('bottom')
        plt.tight_layout()

        fig_path = os.path.splitext(audio_path)[1] + '.png'
        print('Save sound event detection visualization' 
              f'of {audio_path} to {fig_path}.')
        plt.savefig(fig_path)

    else:
        result = clipwise_output.data.cpu().numpy()[0]  # shape = (classes_num,)
        sorted_indexes = np.argsort(result)[::-1]
        # Print audio tagging top probabilities
        for k in range(top_k):
            print(labels[sorted_indexes[k]], f"{result[sorted_indexes[k]]}:.3f")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', type=str, required=True,
                        help='WAV audiofile to be analyzed.')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Model checkpoint to use for inference.')
    parser.add_argument('--class_labels_path', type=str, required=True,
                        help="Dataset labels in tsv format (in 'Reformatted' format)")
    parser.add_argument('--selected_classes_path', type=str, required=True,
                        help="List of class ids selected for training, one per line")
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

    _, labels = get_class_labels(args.class_labels_path, args.selected_classes_path)

    model = panns.models.load_model(model=args.model_type,
                                    win_length=args.window_size,
                                    hop_length=args.hop_size,
                                    sample_rate=args.sample_rate,
                                    n_mels=args.mel_bins,
                                    f_min=args.fmin, f_max=args.fmax,
                                    classes_num=args.classes_num,
                                    checkpoint=args.checkpoint_path)
    plot_clip_inference(model=model,
                        audio_path=args.audio_path,
                        label=labels,
                        win_length=args.window_size,
                        hop_length=args.hop_size,
                        sample_rate=args.sample_rate,
                        cuda=args.cuda,
                        sed=args.sed, verbose=args.verbose)
