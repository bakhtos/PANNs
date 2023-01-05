# PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition (forked by ML TUNI)

This repository is a fork of https://github.com/qiuqiangkong/audioset_tagging_cnn 
(which is the code release for paper **PANNs: Large-Scale Pretrained Audio Neural Networks
for Audio Patterns Recognition** [1])

## Dataset and metadata

We do not provide here the scripts to download the dataset and assume that the
files are already prepared by the user.

### Preparing the dataset and metadata

The codes here assume that all files are already the audio segments which are to be studied
and not full recordings from which segments are to be extracted, that the
segments are 10 seconds long (although codes should support any length of audio common to all used clips)
and are named according to the pattern `Y*.wav`, 
i.e. begin with 'Y', followed by arbitrary symbols (video's YouTubeID) and having the extension '.wav'.
Data should have a sampling rate of 32000Hz,
although again, any sample rate common to all files is supported.

The files should be stored in two separate directories, `train/` and `eval/`, for the
training and evaluation splits, respectively.

Metadata files storing class labels for audios should be provided as 
tab-separated files, 
as in [Google's AudioSet: Reformatted](https://github.com/bakhtos/GoogleAudioSetReformatted) dataset.

***NOTE:*** `filename` fields should not contain the prefix "Y" or the extension ".wav", these
are added by the scripts.

These metadata files should only mention audio files that were actually downloaded,
and only the classes that were selected for the model.

You can check the [dataset](dataset) folder to verify the format of all files.

Create the following environment variables to keep track of all files and 
locations:

```shell
AUDIOS_DIR_TRAIN # Location of the training split of files
AUDIOS_DIR_EVAL # Location of the evaluation split of files
DATASET_PATH_TRAIN # Path to the tsv file containing strong labels for the train split
DATASET_PATH_EVAL # Path to the tsv file containing strong labels for the eval split
LOGS_DIR # Location to store logs
```

### Create target arrays

Use the `panns.data.target` module from the command line to save the 
target array for train and eval splits.
Weak target array will have the shape `(files, classes)`,
strong target array - `(files, frames, classes)`.
Weak target can be computed once for a dataset, Strong target depends on 
sample rate, hop length (in samples) and clip length (in ms).

Use the following environment variables:

```shell
TARGET_WEAK_PATH_TRAIN # Path to save the weak target array for train files
TARGET_WEAK_PATH_EVAL # Path to save the weak target array for eval files
TARGET_STRONG_PATH_TRAIN # Path to save the strong target array for train files
TARGET_STRONG_PATH_EVAL # Path to save the strong target array for eval files
```

And then call the module as follows:

```shell
# Weak target
# Train split
python -m panns.data.target --dataset_path=$DATASET_PATH_TRAIN\
                            --target_type=weak\
                            --target_path=$TARGET_WEAK_PATH_TRAIN
# Eval split
python -m panns.data.target --dataset_path=$DATASET_PATH_EVAL\
                            --target_type=weak\
                            --target_path=$TARGET_WEAK_PATH_EVAL
                              
# Strong target
# Train split
python -m panns.data.target --dataset_path=$DATASET_PATH_TRAIN\
                            --target_type=strong\
                            --target_path=$TARGET_STRONG_PATH_TRAIN\
                            --sample_rate=32000\
                            --hop_length=320\
                            --clip_length=10000
# Eval split
python -m panns.data.target --dataset_path=$DATASET_PATH_EVAL\
                            --target_type=strong\
                            --target_path=$TARGET_STRONG_PATH_EVAL\
                            --sample_rate=32000\
                            --hop_length=320\
                            --clip_length=10000
```

### Pack waveforms into hdf5 files

For training and evaluation, the actual audio files need to be packed into a [hdf5](https://hdfgroup.org/) object
using the `panns.data.hdf5` module, which relies on the [h5py](https://www.h5py.org/) package.

The audio arrays will be made to match the length `clip length*sample rate/1000` either by truncating or zero-padding.

Create the following environment variables:

```shell
HDF5_FILES_PATH_TRAIN # Location for the hdf5 compression of train split
HDF5_FILES_PATH_EVAL # Location for the hdf5 compression of eval split
```

And make the calls:

```shell
# Train split
python -m panns.data.hdf5  --audios_dir=$AUDIOS_DIR_TRAIN\
                           --dataset_path=$DATASET_PATH_TRAIN\
                           --hdf5_path=$HDF5_FILES_PATH_TRAIN\
                           --logs_dir=$LOGS_DIR
                           --sample_rate=32000\
                           --clip_length=10000
                           
# Eval split
python -m panns.data.hdf5  --audios_dir=$AUDIOS_DIR_EVAL\
                           --dataset_path=$DATASET_PATH_EVAL\
                           --hdf5_path=$HDF5_FILES_PATH_EVAL\
                           --logs_dir=$LOGS_DIR
                           --sample_rate=32000\
                           --clip_length=10000
```

***NOTE*** Optionally `--mini_data` parameter can be specified, which only packs
the given amount of files.

## Models
The models are defined in [panns/models/models.py](panns/models/models.py),
some auxiliary classes are defined in [panns/models/blocks.py](panns/models/blocks.py).

Models have been significantly reworked compared to [the original implementation](http://github.com/qiuqiangkong/audioset_tagging_cnn).

In particular, custom-written [torchlibrosa](https://github.com/qiuqiangkong/torchlibrosa)
has been replaced with native
[torchaudio](https://pytorch.org/audio/stable/index.html). This applies to 
Spectrogram extraction for models that require it as well as Spectrogram 
Augmentation.

Furthermore, many versions of the CNN14 model in the original implementation 
differed only by a handful of parameters that were hardcoded. They are now 
refactored into the main CNN14 model with the possibility to customize these 
parameters to resemble the original models. CNN6 and CNN10 also had these features 
inserted into them.

In general, these parameters are used to customize the models (some models 
only support some parameters, check [the source](panns/models/models.py)):

- `classes_num`: Amount of classes used
- `wavegram`: Whether to use the Wavegram features (see [1])
- `spectrogram`: Whether to use the log-mel-Spectrogram features
  - `sample_rate`: Sample rate of the original audio
  - `win_length`: Window length to use for MelSpectrogram extraction
  - `hop_length`: Hop length for the window of MelSpectrogram extraction 
  - `n_mels`: Amount of mel filterbanks to use for MelSpectrogram
  - `f_min`: Minimum frequency
  - `f_max`: Maximum frequency
  - `spec_aug`: Whether to use SpectrogramAugmentation during training
- `mixup_time`: Whether to perform mixup in time-domain (before feature 
  extraction)
- `mixup_freq`: Whether to perform mixup in frequency domain (after feature 
  extraction)
- `dropout`: Whether to perform dropout during training
- `decision_level`: Whether to output strong labels (`framewise_output`) and 
  which function to use to generate them
- Additional:
  - `window_fn`, `center`, `pad_mode`: Passed to [MelSpectrogram](https://pytorch.org/audio/stable/generated/torchaudio.transforms.MelSpectrogram.html#torchaudio.transforms.MelSpectrogram)
  - `top_db`: Passed to [AmplitudeToDB](https://pytorch.org/audio/stable/generated/torchaudio.transforms.AmplitudeToDB.html#torchaudio.transforms.AmplitudeToDB)
  - `num_features`: Passed to [BatchNorm2D](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html?highlight=batchnorm2d) (must be correct with respect to the input)
  - `embedding_size`: Amount of nodes connecting the last two layers of the 
    model

Below is the 'conversion table' between models in the original and current
implementations (note that in all cases parameters for Spectrogram are renamed):

| Original                   | Current                                                                                                                                                                                                            |
|----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Cnn6`                     | `Cnn6(spec_aug=True, mixup_time=False, mixup_freq=True, dropout=True, wavegram=False, spectrogram=True, decision_level=None)`                                                                                      |
| `Cnn10`                    | `Cnn10(spec_aug=True, mixup_time=False, mixup_freq=True, dropout=True, wavegram=False, spectrogram=True, decision_level=None)`                                                                                     |
| `Cnn14`                    | `Cnn14(spec_aug=True, mixup_time=False, mixup_freq=True, dropout=True, wavegram=False, spectrogram=True, decision_level=None)`                                                                                     |
| `Cnn14_8k`                 | `Cnn14(spec_aug=True, mixup_time=False, mixup_freq=True, dropout=True, wavegram=False, spectrogram=True, decision_level=None, sample_rate=8000, win_length=256, hop_length=80, n_mels=64, f_mix=50, f_max=4000)`   |
| `Cnn14_16k`                | `Cnn14(spec_aug=True, mixup_time=False, mixup_freq=True, dropout=True, wavegram=False, spectrogram=True, decision_level=None, sample_rate=16000, win_length=512, hop_length=160, n_mels=64, f_mix=50, f_max=8000)` |
| `Cnn14_no_specaug`         | `Cnn14(spec_aug=False, mixup_time=False, mixup_freq=True, dropout=True, wavegram=False, spectrogram=True, decision_level=None)`                                                                                    |
| `Cnn14_no_dropout`         | `Cnn14(spec_aug=True, mixup_time=False, mixup_freq=True, dropout=False, wavegram=False, spectrogram=True, decision_level=None)`                                                                                    |
| `Cnn14_mixup_time_domain`  | `Cnn14(spec_aug=True, mixup_time=True, mixup_freq=False, dropout=True, wavegram=False, spectrogram=True, decision_level=None)`                                                                                     |
| `Cnn14_emb32`              | `Cnn14(spec_aug=True, mixup_time=False, mixup_freq=True, dropout=True, wavegram=False, spectrogram=True, decision_level=None, embedding_size=32)`                                                                  |
| `Cnn14_emb128`             | `Cnn14(spec_aug=True, mixup_time=False, mixup_freq=True, dropout=True, wavegram=False, spectrogram=True, decision_level=None, embedding_size=128)`                                                                 |
| `Cnn14_emb512`             | `Cnn14(spec_aug=True, mixup_time=False, mixup_freq=True, dropout=True, wavegram=False, spectrogram=True, decision_level=None, embedding_size=512)`                                                                 |
| `Cnn14_mel32`              | `Cnn14(spec_aug=True, mixup_time=False, mixup_freq=True, dropout=True, wavegram=False, spectrogram=True, decision_level=None, num_features=32)`                                                                    |
| `Cnn14_mel128`             | `Cnn14(spec_aug=True, mixup_time=False, mixup_freq=True, dropout=True, wavegram=False, spectrogram=True, decision_level=None, num_features=128)`                                                                   |
| `Cnn14_DecisionLevelMax`   | `Cnn14(spec_aug=True, mixup_time=False, mixup_freq=True, dropout=True, wavegram=False, spectrogram=True, decision_level='max')`                                                                                    |
| `Cnn14_DecisionLevelAvg`   | `Cnn14(spec_aug=True, mixup_time=False, mixup_freq=True, dropout=True, wavegram=False, spectrogram=True, decision_level='avg')`                                                                                    |
| `Cnn14_DecisionLevelAtt`   | `Cnn14(spec_aug=True, mixup_time=False, mixup_freq=True, dropout=True, wavegram=False, spectrogram=True, decision_level='att')`                                                                                    |
| `Wavegram_Cnn14`           | `Cnn14(spec_aug=True, mixup_time=False, mixup_freq=True, dropout=True, wavegram=True, spectrogram=False, decision_level=None)`                                                                                     |
| `Wavegram_Logmel_Cnn14`    | `Cnn14(spec_aug=True, mixup_time=False, mixup_freq=True, dropout=True, wavegram=True, spectrogram=True, decision_level=None)`                                                                                      |
| `Wavegram_Logmel128_Cnn14` | Not implemented                                                                                                                                                                                                    |
| `ResNet22`                 | `ResNet22` (same)                                                                                                                                                                                                  |
| `ResNet38`                 | `ResNet38` (same)                                                                                                                                                                                                  |
| `ResNet54`                 | `ResNet54` (same)                                                                                                                                                                                                  |
| `Res1dNet31`               | `Res1dNet31(classes_num)` (other parameters are not used)                                                                                                                                                          |
| `Res1dNet51`               | `Res1dNet51(classes_num)` (other parameters are not used)                                                                                                                                                          |
| `MobileNetV1`              | `MobileNetV1` (same)                                                                                                                                                                                               |
| `MobileNetV2`              | `MobileNetV2` (same)                                                                                                                                                                                               |
| `LeeNet11`                 | `LeeNet11(classes_num)` (other parameters are not used)                                                                                                                                                            |
| `LeeNet24`                 | `LeeNet24(classes_num, dropout=True)` (other parameters are not used, dropout can be set to `False`)                                                                                                               |
| `DaiNet19`                 | `DaiNet19(classes_num)` (other parameters are not used)                                                                                                                                                            |


## Training

Training is performed using [panns/train.py](panns/train.py). 
Training is controlled by the following parameters:
- `model_type`: One of the classes in [panns/models/models.py](panns/models/models.py) (the model used)
- Parameters for the model (see [Models](#Models))
- Batch size (```batch_size```): amount of files used in one training loop
- Maximum iteration (```iter_max```): amount of iterations performed (an 
  'iteration' is processing
        of one batch, we do not use epochs in this pipeline)
- Learning rate (```learning_rate```): learning rate parameter for the optimizer
- Number of workers (```num_workers```) to use if training on GPU

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
                      --f_min=50\
                      --f_max=14000\
                      --n_mels=64\
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
  with [panns.data.target](panns/data/target.py))
- `output_path`: filename to save the detected events
- `checkpoint_path`: location of the checkpoint of the model to use
- `logs_dir`: directory to write logs into (optional)
- `selected_classes_path`: same as defined for SELECTED_CLASSES_PATH 
  variable above
- `class_labels_path`: same as defined for CLASS_LABELS_PATH variable above
- `threshold`: This threshold is applied to the output of the model only values 
  greater are considered as 'event detected'
- `minimum_event_gap`: In seconds, minimum gap between two consecutive 
  events so that they are considered separate events; events closer than 
  this are merged together by filling the small gap
- `minimum_event_length`: In seconds, events shorter than this are ignored 
  (first gaps are closed, than short events removed)
- `batch_size`, `cuda`, `num_workers`: Control passing data to the model 
  similarly to training phase
- `model_type`, `classes_num`, `sample_rate`, `n_mels`, `f_min`, `f_max`, 
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
                          --n_mels=64\
                          --f_min=50\
                          --f_max=14000\
                          --batch_size=32\
                          --classes_num=110\
                          --num_workers=8
```

## Cite
[1] Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D. Plumbley. "Panns: Large-scale pretrained audio neural networks for audio pattern recognition." IEEE/ACM Transactions on Audio, Speech, and Language Processing 28 (2020): 2880-2894.

## Reference
[2] Gemmeke, J.F., Ellis, D.P., Freedman, D., Jansen, A., Lawrence, W., Moore, R.C., Plakal, M. and Ritter, M., 2017, March. Audio set: An ontology and human-labeled dataset for audio events. In IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 776-780, 2017

[3] Hershey, S., Chaudhuri, S., Ellis, D.P., Gemmeke, J.F., Jansen, A., Moore, R.C., Plakal, M., Platt, D., Saurous, R.A., Seybold, B. and Slaney, M., 2017, March. CNN architectures for large-scale audio classification. In 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 131-135, 2017
