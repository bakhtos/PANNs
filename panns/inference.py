import argparse

import torch
import torch.utils.data
import numpy as np

from panns.data.dataset import AudioSetDataset
from panns.forward import forward
from panns.utils.metadata_utils import get_labels
from panns.models.loader import load_model

__all__ = ['detect_events']


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

            event_activity = event_activity.reshape((-1, 2)) * hop_length_seconds

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
    parser.add_argument('--hdf5_files_path', type=str, required=True,
                        help="Path to hdf5 file of the eval split")
    parser.add_argument('--target_weak_path', type=str, required=True,
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

    model = load_model(args.model_type, args.sample_rate,
                       args.window_size, args.hop_size,
                       args.mel_bins, args.fmin, args.fmax,
                       args.classes_num,
                       checkpoint=args.checkpoint_path)

    device = torch.device('cuda') if (args.cuda and torch.cuda.is_available()) \
        else torch.device('cpu')

    # Parallel
    if device.type == 'cuda':
        print(f'Using GPU. GPU number: {torch.cuda.device_count()}')
        model.to(device)
        model = torch.nn.DataParallel(model)
    else:
        print('Using CPU.')

    dataset = AudioSetDataset(args.hdf5_files_path, args.target_weak_path)
    eval_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.num_workers,
                                              persistent_workers=True,
                                              pin_memory=True)

    clipwise_output, second_output, _, _ = forward(model, eval_loader)

    audio_names = np.load(args.audio_names_path)

    ids, _, _, _ = get_labels(args.class_labels_path,
                              args.selected_classes_path)

    detect_events(frame_probabilities=second_output,
                  label_id_list=ids,
                  filenames=audio_names,
                  threshold=args.threshold,
                  minimum_event_length=args.minimum_event_length,
                  minimum_event_gap=args.minimum_event_gap,
                  sample_rate=args.sample_rate,
                  hop_size=args.hop_size)
