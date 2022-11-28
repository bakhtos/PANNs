import argparse

import torch
import torch.utils.data
import numpy as np

from panns.data.dataset import AudioSetDataset
from panns.forward import forward
from panns.utils.metadata_utils import get_labels
import panns.models

__all__ = ['inference', 'detect_events']


def inference(*, hdf5_files_path_eval,
              target_weak_path_eval,
              checkpoint_path,
              model,
              cuda=False,
              sed=False,
              num_workers=8, batch_size=32):
    """Obtain audio tagging or sound event detection results from a model.

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
    :raises ValueError: if model_type not found in panns.models.models.py
    """

    if sed and not model.sed_model:
        print(f"Warning! Asked to perform SED but given model is not a SED "
              f"model."
              "Performing Audio Tagging instead.")
        sed = False

    device = torch.device('cuda') if (cuda and torch.cuda.is_available()) \
        else torch.device('cpu')

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    # Parallel
    model.to(device)
    model = torch.nn.DataParallel(model)
    if device.type == 'cuda':
        print(f'Using GPU. GPU number: {torch.cuda.device_count()}')
    else:
        print('Using CPU.')

    dataset = AudioSetDataset(hdf5_files_path_eval, target_weak_path_eval)
    eval_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              persistent_workers=True,
                                              pin_memory=True)

    clipwise_output, second_output, _, _ = forward(model, eval_loader)

    if sed:
        result = second_output
    else:
        result = clipwise_output

    return result


def detect_events(*, frame_probabilities,
                  label_id_list,
                  filenames,
                  output='events.txt',
                  threshold=0.5,
                  minimum_event_length=0.1,
                  minimum_event_gap=0.1,
                  sample_rate=32000,
                  hop_size=320):
    """Detect Sound Events using a given framewise probability array.

    :param numpy.ndarray frame_probabilities: A two-dimensional array of framewise probablities of classes. First dimension corresponds to the classes,
        second to the frames of the audio clip
    :param dict label_id_list:
        List of class ids used to index the frame_probabilities tensor
    :param numpy.ndarray filenames: Name of the audio clip to which the
    frame_probabilities correspond.
    :param str output: Filename to write detected events into (default 'events.txt')
    :param float threshold: Threshold used to binarize the frame_probabilites.
        Values higher than the threshold are considered as 'event detected' (default 0.5)
    :param int minimum_event_length: Minimum length (in seconds) of detetected event
        to be considered really present
    :param int minimum_event_gap: Minimum length (in seconds) of a gap between
        two events to distinguish them, if the gap is smaller events are merged
    :param int sample_rate: Sample rate of audio clips used in the dataset (default 32000)
    :param int hop_size: Hop length which was used to obtain the frame_probabilities (default 320)
    """

    event_file = open(output, 'w')
    event_file.write('filename\tevent_label\tonset\toffset\n')

    hop_length_seconds = hop_size / sample_rate
    activity_array = frame_probabilities >= threshold
    change_indices = np.logical_xor(activity_array[:, 1:, :],
                                    activity_array[:, :-1, :])

    for file_ix in range(frame_probabilities.shape[0]):
        filename = filenames[file_ix]
        for event_ix, event_id in enumerate(label_id_list):
            event_activity = change_indices[file_ix, :, event_ix].nonzero()[
                                 0] + 1

            if activity_array[file_ix, 0, event_ix]:
                # If the first element of activity_array is True add 0 at the beginning
                event_activity = np.r_[0, event_activity]

            if activity_array[file_ix, -1, event_ix]:
                # If the last element of activity_array is True, add the length of the array
                event_activity = np.r_[event_activity, activity_array.shape[1]]

            event_activity = event_activity.reshape(
                    (-1, 2)) * hop_length_seconds

            # Store events
            if event_activity.size != 0:
                current_onset = event_activity[0][0]
                current_offset = event_activity[0][1]
            for event in event_activity:
                need_write = False
                if minimum_event_gap is not None:
                    if event[0] - current_offset >= minimum_event_gap:
                        need_write = True
                        onset_write = current_onset
                        offset_write = current_offset
                        current_onset = event[0]
                    current_offset = event[1]
                else:
                    need_write = True
                    onset_write = event[0]
                    offset_write = event[1]
                if need_write and (minimum_event_length is None or
                                   offset_write - onset_write > minimum_event_length):
                    event_file.write(
                            f'{filename}\t{event_id}\t{onset_write}\t{offset_write}\n')

            if (minimum_event_gap is not None and event_activity.size != 0
                    and (
                            minimum_event_length is None or current_offset - current_onset >
                            minimum_event_length)):
                event_file.write(f'{filename}\t{event_id}\t{current_onset}'
                                 f'\t{current_offset}\n')

    event_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5_files_path_eval', type=str, required=True,
                        help="Path to hdf5 file of the eval split")
    parser.add_argument('--target_weak_path_eval', type=str, required=True,
                        help="Path to the weak target array of the eval split")
    parser.add_argument('--audio_names_path', type=str, required=True,
                        help='Path to .npy file to load audio filenames to be packed in this order')
    parser.add_argument('--model_type', type=str, required=True,
                        help="Name of model to train")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help="File to load the NN checkpoint from")
    parser.add_argument('--selected_classes_path', type=str, required=True,
                        help="Dataset class labels in tsv format (as in "
                             "'Reformatted' dataset)")
    parser.add_argument('--class_labels_path', type=str, required=True,
                        help='List of selected classes that were used in training'
                             '/are used in the model, one per line')
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
                        help="If set, try to use GPU for training")
    parser.add_argument('--sed', action='store_true', default=False,
                        help='If set, perform Sound Event Detection, otherwise Audio Tagging')
    parser.add_argument('--classes_num', type=int, default=110,
                        help="Amount of classes used in the dataset (default 110)")
    parser.add_argument('--num_workers', type=int, default=8,
                        help="Amount of workers to pass to torch.utils.data.DataLoader (default 8)")
    parser.add_argument('--threshold', type=float, default=0.5,
                        help="Threshold for frame activity tensor, values above the threshold"
                             " are interpreted as 'event present' (default 0.5)")
    parser.add_argument('--minimum_event_length', type=float, default=0.1,
                        help="Events shorter than this ae filtered out (default 0.1)")
    parser.add_argument('--minimum_event_gap', type=float, default=0.1,
                        help="Two consecutive events with gap between them less"
                             " than this are joined together (default 0.1)")

    args = parser.parse_args()
    model = panns.models.load_model(args.model_type, args.sample_rate,
                                    args.window_size, args.hop_size,
                                    args.mel_bins, args.fmin, args.fmax,
                                    args.classes_num)
    results = inference(
            hdf5_files_path_eval=args.hdf5_files_path_eval,
            target_weak_path_eval=args.target_weak_path_eval,
            model=model,
            checkpoint_path=args.checkpoint_path,
            batch_size=args.batch_size,
            cuda=args.cuda, sed=args.sed,
            num_workers=args.num_workers)

    audio_names = np.load(args.audio_names_path)

    ids, _, _, _ = get_labels(args.class_labels_path,
                              args.selected_classes_path)
    detect_events(frame_probabilities=results,
                  label_id_list=ids,
                  filenames=audio_names,
                  threshold=args.threshold,
                  minimum_event_length=args.minimum_event_length,
                  minimum_event_gap=args.minimum_event_gap,
                  sample_rate=args.sample_rate,
                  hop_size=args.hop_size)
