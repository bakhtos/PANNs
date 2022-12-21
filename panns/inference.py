import argparse
import os
import logging

import torch.utils.data
import numpy as np

from panns.data.dataset import AudioSetDataset
from panns.forward import forward
from panns.utils.metadata_utils import get_labels
from panns.utils.logging_utils import create_logging
from panns.models.loader import load_model, model_parser

__all__ = ['detect_events']


def detect_events(*, frame_probabilities,
                  label_id_list,
                  filenames,
                  output='events.txt',
                  threshold=0.5,
                  minimum_event_length=0.1,
                  minimum_event_gap=0.1,
                  sample_rate=32000,
                  hop_length=320):
    """Detect Sound Events using a given framewise probability array.

    Args:
        frame_probabilities: numpy.ndarray, A two-dimensional array of
            framewise probabilities of classes. First dimension corresponds
            to the classes, second to the frames of the audio clip
        label_id_list: list, List of class ids used to index the
            frame_probabilities tensor
        filenames: numpy.ndarray, Name of the audio clip to which the
            frame_probabilities correspond.
        output: str, Filename to write detected events into (default 'events.txt')
        threshold: float, Threshold used to binarize the frame_probabilities.
            Values higher than the threshold are considered as 'event detected'
            (default 0.5)
        minimum_event_length: float, Minimum length (in seconds) of detected event
            to be considered really present (default 0.1)
        minimum_event_gap: float, Minimum length (in seconds) of a gap between
            two events to distinguish them, if the gap is smaller events are merged
            (default 0.1)
        sample_rate: int, Sample rate of audio clips used in the dataset
            (default 32000)
        hop_length: int, Hop length which was used to obtain the frame_probabilities
            (default 320)

    Returns:
        Writes detected events for each file into output file.
    """

    logging.info(f"Detecting events with parameters threshold={threshold}, "
                 f"minimum_event_length={minimum_event_length}, "
                 f"minimum_event_gap={minimum_event_gap}i, hop_length="
                 f"{hop_length}, sample_rate={sample_rate}.")

    event_file = open(output, 'w')
    event_file.write('filename\tevent_label\tonset\toffset\n')
    logging.info(f"Created file {output}.")

    hop_length_seconds = hop_length / sample_rate
    activity_array = frame_probabilities >= threshold
    change_indices = np.logical_xor(activity_array[:, 1:, :],
                                    activity_array[:, :-1, :])

    for file_ix in range(frame_probabilities.shape[0]):
        filename = filenames[file_ix]
        logging.info(f"Processing file {filename}...")
        for event_ix, event_id in enumerate(label_id_list):
            logging.info(f"Processing event {event_id}...")
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
                    and (minimum_event_length is None or current_offset - current_onset >
                         minimum_event_length)):
                event_file.write(f'{filename}\t{event_id}\t{current_onset}'
                                 f'\t{current_offset}\n')

    logging.info("Detection finished.")
    event_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[model_parser])
    files = parser.add_argument_group('Files', 'Arguments to specify paths '
                                               'to necessary files')
    files.add_argument('--hdf5_files_path', type=str, required=True,
                       help="Path to hdf5 file of the eval split")
    files.add_argument('--target_weak_path', type=str, required=True,
                       help="Path to the weak target array of the eval split")
    files.add_argument('--audio_names_path', type=str, required=True,
                       help='Path to .npy file to load audio filenames '
                            'to be packed in this order')
    files.add_argument('--output_path', type=str, default='events.txt',
                       help="File to save detected events")
    files.add_argument('--checkpoint_path', type=str, required=True,
                       help="File to load the NN checkpoint from")
    files.add_argument('--logs_dir', type=str, help="Directory to save the "
                                                    "logs into")
    files.add_argument('--selected_classes_path', type=str, required=True,
                       help="Dataset class labels in tsv format (as in "
                            "'Reformatted' dataset)")
    files.add_argument('--class_labels_path', type=str, required=True,
                       help="List of selected classes that were used in "
                            "training are used in the model, one per line")
    training = parser.add_argument_group("Training", "Parameters to customize "
                                                     "training")
    training.add_argument('--batch_size', type=int, default=32,
                          help="Batch size to use for training/evaluation ("
                             "default 32)")
    training.add_argument('--cuda', action='store_true', default=False,
                          help="If set, try to use GPU for training")
    training.add_argument('--num_workers', type=int, default=8,
                          help="Amount of workers to pass to "
                               "torch.utils.data.DataLoader (default 8)")
    infer = parser.add_argument_group('Inference', 'Parameters to control '
                                                   'inference')
    infer.add_argument('--threshold', type=float, default=0.5,
                       help="Threshold for frame activity tensor, values "
                            "above the threshold are interpreted as'event "
                            "present' (default 0.5)")
    infer.add_argument('--minimum_event_length', type=float, default=0.1,
                       help="Events shorter than this ae filtered out (default 0.1)")
    infer.add_argument('--minimum_event_gap', type=float, default=0.1,
                       help="Two consecutive events with gap between them less"
                            " than this are joined together (default 0.1)")

    args = parser.parse_args()

    logs_dir = args.logs_dir
    if logs_dir is None:
        logs_dir = os.path.join(os.getcwd(), 'logs')
    create_logging(logs_dir, filemode='w')

    spec_aug = args.spec_aug or args.no_spec_aug
    mixup_time = args.mixup_time or args.no_mixup_time
    mixup_freq = args.mixup_freq or args.no_mixup_freq
    dropout = args.dropout or args.no_dropout
    wavegram = args.wavegram or args.no_wavegram
    spectrogram = args.spectrogram or args.no_spectrogram
    center = args.center or args.no_center

    model = load_model(model=args.model_type,
                       checkpoint=args.resume_checkpoint_path,
                       spec_aug=spec_aug, mixup_time=mixup_time,
                       mixup_freq=mixup_freq, dropout=dropout,
                       wavegram=wavegram, spectrogram=spectrogram,
                       decision_level=args.decision_level, center=center,
                       win_length=args.win_length, hop_length=args.hop_length,
                       n_mels=args.n_mels, f_min=args.f_min, f_max=args.f_max,
                       pad_mode=args.pad_mode, top_db=args.top_db,
                       num_features=args.num_features,
                       embedding_size=args.embedding_size,
                       classes_num=args.classes_num)

    device = torch.device('cuda') if (args.cuda and torch.cuda.is_available()) \
        else torch.device('cpu')

    # Parallel
    if device.type == 'cuda':
        logging.info(f'Using GPU. GPU number: {torch.cuda.device_count()}')
        model.to(device)
        model = torch.nn.DataParallel(model)
    else:
        logging.info('Using CPU.')

    dataset = AudioSetDataset(args.hdf5_files_path, args.target_weak_path)
    eval_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.num_workers,
                                              persistent_workers=True,
                                              pin_memory=True)

    clipwise_output, framewise_output, _, _ = forward(model, eval_loader)

    framewise_output = framewise_output.cpu().numpy()
    audio_names = np.load(args.audio_names_path)

    ids, _, _, _ = get_labels(args.class_labels_path,
                              args.selected_classes_path)

    detect_events(frame_probabilities=framewise_output,
                  label_id_list=ids,
                  filenames=audio_names,
                  threshold=args.threshold,
                  output=args.output_path,
                  minimum_event_length=args.minimum_event_length,
                  minimum_event_gap=args.minimum_event_gap,
                  sample_rate=args.sample_rate,
                  hop_length=args.hop_length)
