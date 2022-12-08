# PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition (forked by ML TUNI)

This repository is a fork of https://github.com/qiuqiangkong/audioset_tagging_cnn 
(which is the code release for paper **PANNs: Large-Scale Pretrained Audio Neural Networks
for Audio Patterns Recognition** [1])

## Dataset and metadata

We do not provide here the scripts to download the dataset and assume that the
files are already prepared by the user.

### Storing the dataset and metadata

The codes here assume that all files are already the audio segments which are to be studied
(and not full recordings from which segments are to be extracted), that the
segments are 10 seconds long (or less, although codes should support any length of audio common to all used clips)
and are named according to the pattern `Y*.wav`, 
i.e. begin with 'Y', followed by arbitrary symbols (video's YouTubeID) and having the extension '.wav.'
Data should have a sampling rate of either 32000Hz(preferred), 16000Hz or 8000Hz
(although again, any sample rate common to all files could be possible).

The files should be stored in two separate directories, `train/` and `eval/`, for the
training and evaluation splits, respectively.

Metadata files storing class labels should be provided as tab-separated files, 
as in [Google's AudioSet: Reformatted](https://github.com/bakhtos/GoogleAudioSetReformatted) dataset,
as well as a file (`class_labels.tsv`) listing all class labels and their ids.

These metadata files should only mention audio files that were actually downloaded,
and only the classes that were selected for the model.

Also, a file `selected_classes.txt` listing all classes selected for training line by line should be provided.

***NOTE:*** `filename` fields should not contain the prefix "Y" or the extension ".wav", these
are added by the scripts.

You can check the [dataset](dataset) folder to verify the format of all files.

Create the following environment variables to keep track of all the files and parameters:

```shell
AUDIOS_DIR_TRAIN # Location of the training split of files
AUDIOS_DIR_EVAL # Location of the evaluation split of files
LABELS_TSV_PATH_TRAIN # Path to the tsv file containing strong labels for the train split
LABELS_TSV_PATH_EVAL # Path to the tsv file containing strong labels for the eval split
LOGS_DIR # Location to store logs
SAMPLE_RATE # Sample rate of the audio files
CLIP_LENGTH # Length (in ms) of audio clips used in the dataset
CLASSES_NUM # Amount of classes used in the dataset
CLASS_LABELS_PATH # Path to the tsv file mapping class ids to labels
SELECTED_CLASSES_PATH # Path to the txt file containing list of selected class ids, one on each line
```

These variables will be used later in all shown commands.

### Create audio names list and weak target arrays

Use the `panns.utils.metadata_utils` module from the command line to save the lists of files
as well as the weak target arrays for train and eval splits.
Audio names array will have shape `(audios_num, )`, weak target array - `(audios_num, classes_num)`
(index of an audio name in the audio names array is the same as in target array).

Use the following environment variables:

```shell
AUDIO_NAMES_PATH_TRAIN # Path to save the train audio names array
AUDIO_NAMES_PATH_EVAL # Path to save the eval audio names array
TARGET_WEAK_PATH_TRAIN # Path to save the weak target array for train files
TARGET_WEAK_PATH_EVAL # Path to save the weak target array for eval files
```

And then call the module as follows:

```shell
# Train split
python -m panns.utils.metadata_utils --class_labels_path=$CLASS_LABELS_PATH\
                                     --selected_classes_path=$SELECTED_CLASSES_PATH\
                                     --data_path=$LABELS_TSV_PATH_TRAIN\
                                     --audio_names_path=$AUDIO_NAMES_PATH_TRAIN\
                                     --target_weak_path=$TARGET_WEAK_PATH_TRAIN\
# Eval split
python -m panns.utils.metadata_utils --class_labels_path=$CLASS_LABELS_PATH,
                                     --selected_classes_path=$SELECTED_CLASSES_PATH,
                                     --data_path=$LABELS_TSV_PATH_EVAL,
                                     --audio_names_path=$AUDIO_NAMES_PATH_EVAL,
                                     --target_weak_path=$TARGET_WEAK_PATH_EVAL
```

### Pack waveforms into hdf5 files

For training and evaluation, the actual audio files need to be packed into a [hdf5](https://hdfgroup.org/) object
using the `panns.data.hdf5` module, which relies on the [h5py](https://www.h5py.org/) package.

The audio arrays will be made to match the length `CLIP_LENGTH*SAMPLE_RATE/1000` either by truncating or zero-padding.

Create the following environment variables:

```shell
HDF5_FILES_PATH_TRAIN # Location for the hdf5 compression of train split
HDF5_FILES_PATH_EVAL # Location for the hdf5 compression of eval split
```

And make the calls:

```shell
# Train split
python -m panns.data.hdf5  --audios_dir=$AUDIOS_DIR_TRAIN\
                           --audio_names_path=$AUDIO_NAMES_PATH_TRAIN\
                           --hdf5_path=$HDF5_FILES_PATH_TRAIN\
                           --sample_rate=$SAMPLE_RATE\
                           --clip_length=$CLIP_LENGTH\
                           --logs_dir=$LOGS_DIR
                           
# Eval split
python -m panns.data.hdf5  --audios_dir=$AUDIOS_DIR_EVAL\
                           --audio_names_path=$AUDIO_NAMES_PATH_EVAL\
                           --hdf5_path=$HDF5_FILES_PATH_EVAL\
                           --sample_rate=$SAMPLE_RATE\
                           --clip_length=$CLIP_LENGTH\
                           --logs_dir=$LOGS_DIR

```

***NOTE*** Optionally `--mini_data` parameter can be specified, which only packs the given amount of files.

## Training
The Neural Networks are defined in [panns/models/models.py](panns/models/models.py),
some auxiliary classes are defined in [panns/models/blocks.py]
(panns/models/blocks.py).
Training is performed using [panns/train.py](panns/train.py). 
Training is controlled by the following parameters:
- Window size (```win_length```): size of the sliding window used in LogMel spectrum extraction
- Hop size (```hop_length```): hop size of the window used in LogMel spectrum extraction
- Fmin (```fmin```) and Fmax (```fmax```): minimum and maximum frequencies used in the Mel filterbank
- Mel bins (```mel_bins```): amount of mel filters to be used in the filterbank
- Augmentation (mixup): training can be set up to use Mixup with a given alpha parameter,
  which needs to be provided (```mixup_alpha```), if not provided than Mixup
  not used
- Batch size (```batch_size```): amount of files used in one training loop
- Maximum iteration (```iter_max```): amount of iterations performed (an 
  'iteration' is processing
        of one batch, we do not use epochs in this pipeline)
- Learning rate (```learning_rate```): learning rate parameter for the optimizer
- Number of workers (```num_workers```) to use if training on GPU

Training requires the user to specify which model to use (```model_type```, must 
match one of the classes defined in [panns/models/models.py](panns/models/models.py)).

Also, directories for storing checkpoints, evaluation statistics and logs
can be provided (defined above in environment variables); by default
respectively named folders are created in the current working directory.

In addition, it is possible to load an existing (previously trained) checkpoint
and continue training from that interation, when also ```resume_checkpoint_path```
need to be given.

It is also recommended to create environment variables for the parameters.

Example of initiating training:
```shell
python -m panns.train --hdf5_files_path_train=$HDF5_FILES_PATH_TRAIN\
                      --hdf5_files_path_eval=$HDF5_FILES_PATH_EVAL\
                      --target_weak_path_train=$TARGET_WEAK_PATH_TRAIN\
                      --target_weak_path_eval=$TARGET_WEAK_PATH_EVAL\
                      --model_type='Cnn14_DecisionLevelMax'\
                      --logs_dir=$LOGS_DIR\
                      --checkpoints_dir=$CHECKPOINTS_DIR\
                      --statistics_dir=$STATISTICS_DIR\
                      --win_length=1024\
                      --hop_length=320\
                      --sample_rate=$SAMPLE_RATE\
                      --clip_length=10000\
                      --fmin=50\
                      --fmax=14000\
                      --mel_bins=64\
                      --batch_size=32\
                      --learning_rate=1e-3\
                      --iter_max=600000\
                      --classes_num=$CLASSES_NUM\
                      --num_workers=8
                      --cuda
```
## Inference

It is possible to produce a file with events for a given dataset inferred 
from the trained model in the same format as files in [dataset](dataset)
using [panns.inference](panns/inference.py).
For that a checkpoint of the trained model is needed
as well as an `hdf5` compression of the evaluation set.

The script accepts following parameters:

- `hdf5_files_path`: location of the `hdf5` compression of the dataset
- `target_weak_path`: location of the weak target numpy array for the dataset
- `audio_names_path`: location of the audio_names numpy array (generated 
  with [panns.utils.metadata_utils](panns/utils/metadata_utils.py))
- `output_path`: filename to save the detected events
- `checkpoint_path`: location of the checkpoint of the model to use
- `logs_dir`: directory to write logs into (optional)
- `selected_classes_path`: same as defined for SELECTED_CLASSES_PATH 
  variable above
- `class_labels_path`: same as defined for CLASS_LABELS_PATH variable above
- `threshold`: Output of the model is thresholded by this value, only values 
  greater are considered as 'event detected'
- `minimum_event_gap`: In seconds, minimum gap between two consecutive 
  events so that they are considered separate events; events closer than 
  this are merged together by filling the small gap
- `minimum_event_length`: In seconds, events shorter than this are ignored 
  (first gaps are closed, than short events removed)
- `batch_size`, `cuda`, `num_workers`: Control passing data to the model 
  similarly to training phase
- `model_type`, `classes_num`, `sample_rate`, `mel_bins`, `fmin`, `fmax`, 
  `win_length`, `hop_length`: Parameters for the model

Example of inference:
```shell
python -m panns.inference --hdf5_files_path=$HDF5_FILES_PATH_EVAL\
                          --target_weak_path=$TARGET_WEAK_PATH_EVAL\
                          --audio_names_path=$AUDIO_NAMES_PATH_EVAL\
                          --checkpoint_path=\  #Path to checkpoint in $CHECKPOINTS_DIR
                          --selected_classes_path=$SELECTED_CLASSES_PATH\
                          --class_labels_path=$CLASS_LABELS_PATH\
                          --threshold=0.5\
                          --minimum_event_gap=0.1\
                          --minimum_event_length=0.1\
                          --model_type='Cnn14_DecisionLevelMax'\
                          --sample_rate=32000\
                          --win_length=1024\
                          --hop_length=320\
                          --mel_bins=64\
                          --fmin=50\
                          --fmax=14000\
                          --batch_size=32\
                          --classes_num=110\
                          --num_workers=8
```

## Cite
[1] Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D. Plumbley. "Panns: Large-scale pretrained audio neural networks for audio pattern recognition." IEEE/ACM Transactions on Audio, Speech, and Language Processing 28 (2020): 2880-2894.

## Reference
[2] Gemmeke, J.F., Ellis, D.P., Freedman, D., Jansen, A., Lawrence, W., Moore, R.C., Plakal, M. and Ritter, M., 2017, March. Audio set: An ontology and human-labeled dataset for audio events. In IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 776-780, 2017

[3] Hershey, S., Chaudhuri, S., Ellis, D.P., Gemmeke, J.F., Jansen, A., Moore, R.C., Plakal, M., Platt, D., Saurous, R.A., Seybold, B. and Slaney, M., 2017, March. CNN architectures for large-scale audio classification. In 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 131-135, 2017
