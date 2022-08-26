import numpy as np
import pandas as pd
from sklearn import metrics

from pytorch_utils import forward
from utilities import get_filename
import config

from dcase_util.containers import metadata
import sed_eval
from psds_eval import PSDSEval

from IPython import embed


segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(
    event_label_list=config.labels,
    time_resolution=1.0
)
# Data necessary for using psds metric
metafile = {'filename': '1', 'duration': [180]}
metadatatext = pd.DataFrame(metafile)


def get_psds_format(reference):
    data = {'onset':[], 'offset':[], 'event_label':[]}
    for res in reference:
        data['onset'].append(res['onset'])
        data['offset'].append(res['offset'])
        data['event_label'].append(res['event_label'])

    annotations = pd.DataFrame.from_dict(data)
    annotations.insert(loc=0, column='filename', value=['1'] * annotations.shape[0])
    return annotations


def find_contiguous_regions(activity_array):
    # Find the changes in the activity_array
    change_indices = np.logical_xor(activity_array[1:], activity_array[:-1]).nonzero()[0]
    #change_indices = (activity_array[1:]^activity_array[:-1]).nonzero(as_tuple=True)[0]
    #change_indices = torch.nonzero(np.logical_xor(activity_array[1:], activity_array[:-1])).size(0)
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


def process_event(class_labels, frame_probabilities, threshold, hop_length_seconds):
    results = []
    for event_id, event_label in enumerate(class_labels):
        # Binarization
        #event_id = 4
        event_activity = frame_probabilities[event_id, :] > threshold

        # Convert active frames into segments and translate frame indices into time stamps
        event_segments = find_contiguous_regions(event_activity) * hop_length_seconds

        # Store events
        for event in event_segments:
            results.append(
                metadata.MetaDataItem(
                    {
                        'event_onset': event[0],
                        'event_offset': event[1],
                        'event_label': event_label
                    }
                )
            )

    results = metadata.MetaDataContainer(results)

    # Event list post-processing
    results = results.process_events(minimum_event_length=None, minimum_event_gap=0.1)
    results = results.process_events(minimum_event_length=0.1, minimum_event_gap=None)
    return results


def calculate_accuracy(y_true, y_score):
    N = y_true.shape[0]
    accuracy = np.sum(np.argmax(y_true, axis=-1) == np.argmax(y_score, axis=-1)) / N
    return accuracy


def segment_based(Y_val, pred):
    for f in range(pred.shape[0]):
        reference = process_event(config.labels, Y_val[f, :, :].T, 0, config.hop_size/config.sample_rate)
        results = process_event(config.labels, pred[f, :, :].T, config.posterior_thresh, config.hop_size/config.sample_rate)
        segment_based_metrics.evaluate(
            reference_event_list=reference,
            estimated_event_list=results
        )

    return segment_based_metrics


def sed_average_precision(strong_target, framewise_output, average):
    """Calculate framewise SED mAP.
    Args:
      strong_target: (N, frames_num, classes_num)
      framewise_output: (N, frames_num, classes_num)
      average: None | 'macro' | 'micro'
    """

    assert strong_target.shape == framewise_output.shape
    (N, time_steps, classes_num) = strong_target.shape

    average_precision = metrics.average_precision_score(
        strong_target.reshape((N * time_steps, classes_num)),
        framewise_output.reshape((N * time_steps, classes_num)),
        average=average)

    return average_precision

#CLASS USED FOR VALIDATE
class Validator(object):
    def __init__(self, model):
        self.model = model

    def validate(self, data_loader):

        # Forward
        output_dict = forward(
            model=self.model,
            generator=data_loader,
            return_target=True)
        # Frame wise statistics

        framewise_output = output_dict['framewise_output']    # (audios_num, classes_num)
        target = output_dict['GT']    # (audios_num, classes_num)

        # Add code for evaluating segment based Error rate and other statistics

        #cm = metrics.confusion_matrix(np.argmax(target, axis=-1), np.argmax(clipwise_output, axis=-1), labels=None)
        #accuracy = calculate_accuracy(target, clipwise_output)

        segment_based_metrics = segment_based(target, framewise_output)

        output = segment_based_metrics.result_report_class_wise()
        print(output)

        overall_segment_based_metrics_ER = segment_based_metrics.overall_error_rate()
        overall_segment_based_metrics_f1 = segment_based_metrics.overall_f_measure()

        f1_overall_1sec_list = overall_segment_based_metrics_f1['f_measure']
        er_overall_1sec_list = overall_segment_based_metrics_ER['error_rate']

        segment_based_metrics.reset()

        framewise_ap = sed_average_precision(target, framewise_output, average='macro')

        statistics = {'sed_metrics': framewise_ap,
                      'ER_1sec': er_overall_1sec_list,
                      'F1_1sec': f1_overall_1sec_list }

        return statistics

# CLASS USED FOR TESTING
class Evaluator(object):
    def __init__(self, model):
        self.model = model

    def evaluate(self, data_loader):
    #def testing(self, data_loader):
        # Forward
        output_dict = forward(
            model=self.model,
            generator=data_loader,
            return_target=True)
        # Frame wise statistics
        embed()
        framewise_output = output_dict['framewise_output']    # (audios_num, classes_num)
        target = output_dict['GT']    # (audios_num, classes_num)

        # Add code for testing using psds

        Y_gt = target.reshape(-1, 6)
        reference_gt = process_event(config.labels, Y_gt.T, 0, config.hop_size/config.sample_rate)
        groundtruth = get_psds_format(reference_gt)

        psds_eval2 = PSDSEval(dtc_threshold=0.1, gtc_threshold=0.1, cttc_threshold=0.3,
                              ground_truth=groundtruth, metadata=metadatatext)
        psds_eval1 = PSDSEval(dtc_threshold=0.7, gtc_threshold=0.7, cttc_threshold=0.3,
                              ground_truth=groundtruth, metadata=metadatatext)

        # Calculate the predictions on test data, in order to calculate ER and F scores
        Y_pred = framewise_output.reshape(-1, 6)
        predicted = process_event(config.labels, Y_pred.T, config.posterior_thresh, config.hop_size/config.sample_rate)
        y_pred = get_psds_format(predicted)

        # Calculate PSDS
        macro_f_ev1, class_fev1 = psds_eval1.compute_macro_f_score(y_pred)
        macro_f_ev2, class_f_ev2 = psds_eval2.compute_macro_f_score(y_pred)

        segment_based_metrics = segment_based(target, framewise_output)

        output = segment_based_metrics.result_report_class_wise()
        print(output)

        overall_segment_based_metrics_ER = segment_based_metrics.overall_error_rate()
        overall_segment_based_metrics_f1 = segment_based_metrics.overall_f_measure()

        f1_overall_1sec_list = overall_segment_based_metrics_f1['f_measure']
        er_overall_1sec_list = overall_segment_based_metrics_ER['error_rate']

        segment_based_metrics.reset()


        statistics = {'ER_1sec': er_overall_1sec_list,
                      'F1_1sec': f1_overall_1sec_list,
                      'macro_f_ev1': macro_f_ev1,
                      'macro_f_ev2': macro_f_ev2
                      }

        return statistics
