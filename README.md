# PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition (forked by ML TUNI)

This repository is a fork of https://github.com/qiuqiangkong/audioset_tagging_cnn 
(which is the code release for paper **PANNs: Large-Scale Pretrained Audio Neural Networks
for Audio Patterns Recognition** [1])
redesigned to study the difference between using Weak and Strong labels when
training ANNs for Sound Event Detection (SED):
- We only use those architectures that are able to provide Strong labels
for audio classes (i.e. able to perform Sound Event Detection (SED), as opposed to 
Weak labels for Audio Tagging)
- Preprocessing pipeline is slightly reworked, relying on clear separation of training
and evaluation sets of files.
- Codebase of utility functions is refactored to be easier to navigate
- Many programming style errors are corrected


## Train PANNs from scratch
Users can train PANNs from scratch as follows.

### Downloading and storing the dataset

We do not provide here the scripts to download the dataset and assume that the
files are already prepared by the user.
(The original repository contains some scripts to download the Google's Audioset
with Weak Labels dataset)

The codes here assume that all files are the audio segments which are to be studied
(and not full recordings from which segments are to be extracted), that the
segments are 10 seconds long (or less) and are named according to the pattern `Y*.wav`, 
i.e. begin with 'Y', followed by arbitrary symbols and having the extension '.wav.'
Data should have a sampling rate of either 32000Hz(preferred), 16000Hz or 8000Hz.

The files should be stored in two separate directories, **train** and **eval**, for the
training and evaluation splits, respectively.

Metadata files storing class labels should be provided as tab-separated files, 
storing the informationg in following formats:
- For weak labels: `FILE_ID\tCLASS_ID`, one class per line
- For strong labels: `FILE_ID\tSTART_TIME\tEND_TIME\tCLASS_ID`, one class per line

***NOTE:*** FILE_ID should not contain the prefix "Y" or the extension ".wav", these
are added by the scripts.

Also the following files should be created:
- `train_list.txt`: list of all training files, one on each line (obtained with e.g. 
`ls train > train_list.txt`)
- `eval_list.txt`: list of all evaluation files, one on each line (obtained with e.g. 
`ls eval > eval_list.txt`)
- `selected_classes.txt`: list of all used class labels/ids, one on each line

### Setting up the environment
The codebase is developed with Python 3.7. Install requirements as follows:
```shell
python -m pip install -r requirements.txt
```

#### Environment variables
To ensure consistent usage of parameters and avoiding mistakes when calling several
scripts used in the pipeline, it is recommended to set up certain environment variables
and use them as parameters for all called scripts:

```shell
TRAIN_AUDIOS_DIR # Location of the training split of files
EVAL_AUDIOS_DIR # Location of the evaluation split of files
TRAIN_WEAK_CSV_PATH # Path to the csv (tsv) file containing weak labels for the train split
EVAL_WEAK_CSV_PATH # Path to the csv (tsv) file containing weak labels for the eval split
TRAIN_STRONG_CSV_PATH # Path to the csv (tsv) file containing strong labels for the train split
EVAL_STRONG_CSV_PATH # Path to the csv (tsv) file containing strong labels for the eval split
TRAIN_HDF5_FILES_PATH # Location for the hdf5 compression of train split
EVAL_HDF5_FILES_PATH # Location for the hdf5 compression of eval split
TRAIN_HDF5_INDEX_PATH # Location for the hdf5 index of the train hdf5 compression
EVAL_HDF5_INDEX_PATH # Location for the hdf5 index of the eval hdf5 compression
LOGS_DIR # Location to store logs
STATISTICS_DIR # Location to store eval statistics
CHECKPOINTS_DIR # Location to store NN checkpoints
SAMPLE_RATE # Sample rate of the audio files
CLIP_LENGTH # Length (in ms) of audio clips used in the dataset
CLASSES_NUM # Amount of classes used in the dataset
CLASS_LIST_PATH # Path to the txt file containing list of selected class ids, one on each line
CLASS_CODES_PATH # Path to the csv file mapping class ids to labels
```

These variables will be used later in all shown commands.

### Pack waveforms into hdf5 files

The [panns/data/hdf5.py](panns/data/hdf5.py) file can be called
on the command line with option 'wav_to_hdf5' to create the hdf5 compression of the wav files as follows:

```shell
# Train split
python -m panns.data.hdf5 wav_to_hdf5 --audios_dir=$TRAIN_AUDIOS_DIR\
                            --csv_path=$TRAIN_WEAK_CSV_PATH\
                            --class_list_path=$CLASS_LIST_PATH\
                            --class_codes_path=$CLASS_CODES_PATH\
                            --hdf5_path=$TRAIN_HDF5_FILES_PATH\
                            --clip_length=$CLIP_LENGTH\
                            --sample_rate=$SAMPLE_RATE\
                            --classes_num=$CLASSES_NUM\
                            --logs_dir=$LOGS_DIR

# Eval split
python -m panns.data.hdf5 wav_to_hdf5 --audios_dir=$EVAL_AUDIOS_DIR\
                            --csv_path=$EVAL_WEAK_CSV_PATH\
                            --class_list_path=$CLASS_LIST_PATH\
                            --class_codes_path=$CLASS_CODES_PATH\
                            --hdf5_path=$TRAIN_HDF5_FILES_PATH\
                            --clip_length=$CLIP_LENGTH\
                            --sample_rate=$SAMPLE_RATE\
                            --classes_num=$CLASSES_NUM\
                            --logs_dir=$LOGS_DIR
```

### Create hdf5 indexes

The [panns/data/hdf5.py](panns/data/hdf5.py) file can be called on
the command line with the 'create_indexes' option to create the hdf5 indexes
necessary to access the data in the hdf5 containers with the following arguments:

```shell
# Train split
python -m panns.data.hdf5 create_indexes\
                            --hdf5_path=$TRAIN_HDF5_FILES_PATH\
                            --hdf5_index_path=$TRAIN_HDF5_INDEX_PATH\
                            --logs_dir=$LOGS_DIR
# Eval split
python -m panns.data.hdf5 create_indexes\
                            --hdf5_path=$EVAL_HDF5_FILES_PATH\
                            --hdf5_index_path=$EVAL_HDF5_INDEX_PATH\
                            --logs_dir=$LOGS_DIR
```


### Training
The Neural Networks are defined in [panns/models/models.py](panns/models/models.py),
some auxilary classes are defined in [panns/models/blocks.py](panns/models/block.py).
Training is performed using [panns/train.py](panns/train.py). 
Training is controlled by the following parameters:
- Window size (```window_size```): size of the sliding window used in LogMel spectrum extraction
- Hop size (```hop_size```): hop size of the window used in LogMel spectrum extraction
- Fmin (```fmin```) and Fmax (```fmax```): minimum and maximum frequencies used in the Mel filterbank
- Mel bins (```mel_bins```): amount of mel filters to be used in the filterbank
- Augmentation (mixup): training can be set up to use Mixup (flag ```augmentation```)
    with a given alpha parameter, which also needs to be provided (```mixup_alpha```)
- Batch size (```batch_size```): amount of files used in one training loop
- Maximum iteration (```max_iter```): amount of iterations performed (an 'iteration' is processing
        of one batch, we do not use epochs in this pipeline)
- Learning rate (```learning_rate```): learning rate parameter for the optimizer

Training requires the user to specify which model to use (```model_type```, must 
match one of the classes defined in [panns/models/models.py](panns/models/models.py))
and which data sampler to use for training (```sampler```, must match one of the 
classes defined in [panns/data/loaders.py](panns/data/loaders.py)

Also, directories for storing checkpoints, evaluation statistics and logs
can be provided (defined above in environment variables); by default
respectively named folders are created in the current working directory.

In addition it is possible to load an existing (previously trained) checkpoint
and continue training from a given iteration, when also ```resume_iteration```
and ```resume_checkpoint_path``` need to be given.

Example of initiating training:
```shell
python -m panns.train --train_indexes_hdf5_path=$TRAIN_INDEXES_HDF5_PATH\
                      --eval_indexes_hdf5_path=$EVAL_INDEXES_HDF5_PATH\
                      --model_type='DecisionLevelMax'\
                      --logs_dir=$LOGS_DIR\
                      --checkpoints_dir=$CHECKPOINTS_DIR\
                      --statistics_dir=$STATISTICS_DIR\
                      --window_size=1024\
                      --hop_size=320\
                      --sample_rate=$SAMPLE_RATE\
                      --clip_length=10000\
                      --fmin=50\
                      --fmax=14000\
                      --mel_bins=64\
                      --sampler='TrainSampler'\
                      --batch_size=32\
                      --learning_rate=1e-3\
                      --early_stop=6000000\
                      --classes_num=$CLASSES_NUM\
                      --cuda
```


## Cite
[1] Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D. Plumbley. "Panns: Large-scale pretrained audio neural networks for audio pattern recognition." IEEE/ACM Transactions on Audio, Speech, and Language Processing 28 (2020): 2880-2894.

## Reference
[2] Gemmeke, J.F., Ellis, D.P., Freedman, D., Jansen, A., Lawrence, W., Moore, R.C., Plakal, M. and Ritter, M., 2017, March. Audio set: An ontology and human-labeled dataset for audio events. In IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 776-780, 2017

[3] Hershey, S., Chaudhuri, S., Ellis, D.P., Gemmeke, J.F., Jansen, A., Moore, R.C., Plakal, M., Platt, D., Saurous, R.A., Seybold, B. and Slaney, M., 2017, March. CNN architectures for large-scale audio classification. In 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 131-135, 2017
