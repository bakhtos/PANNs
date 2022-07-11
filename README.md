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
(and not full recordings from which segments are to be extracted) and
 are named according to the pattern `Y*.wav`, 
i.e. begin with 'Y', followed by arbitrary symbols and having the extension '.wav.'

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
SAMPLE_RATE # Sample rate of the audio files
CLASSES_NUM # Amount of classes used in the dataset
```

These variables will be used later in all shown commands.

### Pack waveforms into hdf5 files

The [utils/waveforms_to_hdf5.py](utils/waveforms_to_hdf5.py) file can be called
on the command line to create the hdf5 compression of the wav files as follows:

```shell
# Train split
python utils/waveforms_to_hdf5.py --audios_dir=$TRAIN_AUDIOS_DIR\
                            --csv_path=$TRAIN_WEAK_CSV_PATH\
                            --waveforms_hdf5_path=$TRAIN_HDF5_FILES_PATH\
                            --sample_rate=$SAMPLE_RATE\
                            --classes_num=$CLASSES_NUM

# Eval split
python utils/waveforms_to_hdf5.py --audios_dir=$EVAL_AUDIOS_DIR\
                            --csv_path=$EVAL_WEAK_CSV_PATH\
                            --waveforms_hdf5_path=$EVAL_HDF5_FILES_PATH\
                            --sample_rate=$SAMPLE_RATE\
                            --classes_num=$CLASSES_NUM
```

## Create hdf5 indexes

The [utils/create_indexes.py](utils/create_indexes.py) file can be called on
the command line to create the hdf5 indexes necessary to access the data in the 
hdf5 containers with the following arguments:

```shell
# Train split
python utils/create_indexes.py create_indexes\
                            --waveforms_hdf5_path=$TRAIN_HDF5_FILES_PATH\
                            --indexes_hdf5_path=$TRAIN_HDF5_INDEX_PATH
# Eval split
python utils/create_indexes.py create_indexes\
                            --waveforms_hdf5_path=$EVAL_HDF5_FILES_PATH\
                            --indexes_hdf5_path=$EVAL_HDF5_INDEX_PATH
```


## 4. Train
The [scripts/4_train.sh](scripts/4_train.sh) script contains training, saving checkpoints, and evaluation.

```
WORKSPACE="your_workspace"
CUDA_VISIBLE_DEVICES=0 python3 pytorch/main.py train \
  --workspace=$WORKSPACE \
  --data_type='full_train' \
  --window_size=1024 \
  --hop_size=320 \
  --mel_bins=64 \
  --fmin=50 \
  --fmax=14000 \
  --model_type='Cnn14' \
  --loss_type='clip_bce' \
  --balanced='balanced' \
  --augmentation='mixup' \
  --batch_size=32 \
  --learning_rate=1e-3 \
  --resume_iteration=0 \
  --early_stop=1000000 \
  --cuda
```

## Results
The CNN models are trained on a single card Tesla-V100-PCIE-32GB. (The training also works on a GPU card with 12 GB). The training takes around 3 - 7 days. 

```
Validate bal mAP: 0.005
Validate test mAP: 0.005
    Dump statistics to /workspaces/pub_audioset_tagging_cnn_transfer/statistics/main/sample_rate=32000,window_size=1024,hop_size=320,mel_bins=64,fmin=50,fmax=14000/data_type=full_train/Cnn13/loss_type=clip_bce/balanced=balanced/augmentation=mixup/batch_size=32/statistics.pkl
    Dump statistics to /workspaces/pub_audioset_tagging_cnn_transfer/statistics/main/sample_rate=32000,window_size=1024,hop_size=320,mel_bins=64,fmin=50,fmax=14000/data_type=full_train/Cnn13/loss_type=clip_bce/balanced=balanced/augmentation=mixup/batch_size=32/statistics_2019-09-21_04-05-05.pickle
iteration: 0, train time: 8.261 s, validate time: 219.705 s
------------------------------------
...
------------------------------------
Validate bal mAP: 0.637
Validate test mAP: 0.431
    Dump statistics to /workspaces/pub_audioset_tagging_cnn_transfer/statistics/main/sample_rate=32000,window_size=1024,hop_size=320,mel_bins=64,fmin=50,fmax=14000/data_type=full_train/Cnn13/loss_type=clip_bce/balanced=balanced/augmentation=mixup/batch_size=32/statistics.pkl
    Dump statistics to /workspaces/pub_audioset_tagging_cnn_transfer/statistics/main/sample_rate=32000,window_size=1024,hop_size=320,mel_bins=64,fmin=50,fmax=14000/data_type=full_train/Cnn13/loss_type=clip_bce/balanced=balanced/augmentation=mixup/batch_size=32/statistics_2019-09-21_04-05-05.pickle
iteration: 600000, train time: 3253.091 s, validate time: 1110.805 s
------------------------------------
Model saved to /workspaces/pub_audioset_tagging_cnn_transfer/checkpoints/main/sample_rate=32000,window_size=1024,hop_size=320,mel_bins=64,fmin=50,fmax=14000/data_type=full_train/Cnn13/loss_type=clip_bce/balanced=balanced/augmentation=mixup/batch_size=32/600000_iterations.pth
...
```

An **mean average precision (mAP)** of **0.431** is obtained. The training curve looks like:

<img src="resources/six_figures.png">

Results of PANNs on AudioSet tagging. Dash and solid lines are training mAP and evaluation mAP, respectively. The six plots show the results with different: (a) architectures; (b) data balancing and data augmentation; (c) embedding size; (d) amount of training data; (e) sampling rate; (f) number of mel bins.

## Performance of differernt systems

<img src="resources/mAP_table.png" width=400>

Top rows show the previously proposed methods using embedding features provided by Google. Previous best system achieved an mAP of 0.369 using large feature-attention neural networks. We propose to train neural networks directly from audio recordings. Our CNN14 achieves an mAP of 0.431, and Wavegram-Logmel-CNN achieves an mAP of 0.439.

## Plot figures of [1]
To reproduce all figures of [1], just do:
```
wget -O paper_statistics.zip https://zenodo.org/record/3987831/files/paper_statistics.zip?download=1
unzip paper_statistics.zip
python3 utils/plot_for_paper.py plot_classwise_iteration_map
python3 utils/plot_for_paper.py plot_six_figures
python3 utils/plot_for_paper.py plot_complexity_map
python3 utils/plot_for_paper.py plot_long_fig
```

## Fine-tune on new tasks
After downloading the pretrained models. Build fine-tuned systems for new tasks is simple!

```
MODEL_TYPE="Transfer_Cnn14"
CHECKPOINT_PATH="Cnn14_mAP=0.431.pth"
CUDA_VISIBLE_DEVICES=0 python3 pytorch/finetune_template.py train \
    --sample_rate=32000 \
    --window_size=1024 \
    --hop_size=320 \
    --mel_bins=64 \
    --fmin=50 \
    --fmax=14000 \
    --model_type=$MODEL_TYPE \
    --pretrained_checkpoint_path=$CHECKPOINT_PATH \
    --cuda
```

Here is an example of fine-tuning PANNs to GTZAN music classification: https://github.com/qiuqiangkong/panns_transfer_to_gtzan

## Demos
We apply the audio tagging system to build a sound event detection (SED) system. The SED prediction is obtained by applying the audio tagging system on consecutive 2-second segments. The video of demo can be viewed at: <br>
https://www.youtube.com/watch?v=7TEtDMzdLeY

## FAQs
If users came across out of memory error, then try to reduce the batch size.

## Cite
[1] Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D. Plumbley. "Panns: Large-scale pretrained audio neural networks for audio pattern recognition." IEEE/ACM Transactions on Audio, Speech, and Language Processing 28 (2020): 2880-2894.

## Reference
[2] Gemmeke, J.F., Ellis, D.P., Freedman, D., Jansen, A., Lawrence, W., Moore, R.C., Plakal, M. and Ritter, M., 2017, March. Audio set: An ontology and human-labeled dataset for audio events. In IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 776-780, 2017

[3] Hershey, S., Chaudhuri, S., Ellis, D.P., Gemmeke, J.F., Jansen, A., Moore, R.C., Plakal, M., Platt, D., Saurous, R.A., Seybold, B. and Slaney, M., 2017, March. CNN architectures for large-scale audio classification. In 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 131-135, 2017

## External links
Other work on music transfer learning includes: <br>
https://github.com/jordipons/sklearn-audio-transfer-learning <br>
https://github.com/keunwoochoi/transfer_learning_music

## Audio tagging using pretrained models
Users can inference the tags of an audio recording using pretrained models without training. Details can be viewed at [scripts/0_inference.sh](scripts/0_inference.sh) First, downloaded one pretrained model from https://zenodo.org/record/3987831, for example, the model named "Cnn14_mAP=0.431.pth". Then, execute the following commands to inference this [audio](resources/R9_ZSCveAHg_7s.wav):
```
CHECKPOINT_PATH="Cnn14_mAP=0.431.pth"
wget -O $CHECKPOINT_PATH https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1
MODEL_TYPE="Cnn14"
CUDA_VISIBLE_DEVICES=0 python3 pytorch/inference.py audio_tagging \
    --model_type=$MODEL_TYPE \
    --checkpoint_path=$CHECKPOINT_PATH \
    --audio_path="resources/R9_ZSCveAHg_7s.wav" \
    --cuda
```

Then the result will be printed on the screen looks like:
```
Speech: 0.893
Telephone bell ringing: 0.754
Inside, small room: 0.235
Telephone: 0.183
Music: 0.092
Ringtone: 0.047
Inside, large room or hall: 0.028
Alarm: 0.014
Animal: 0.009
Vehicle: 0.008
embedding: (2048,)
```

If users would like to use 16 kHz model for inference, just do:
```
CHECKPOINT_PATH="Cnn14_16k_mAP=0.438.pth"   # Trained by a later code version, achieves higher mAP than the paper.
wget -O $CHECKPOINT_PATH https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth?download=1
MODEL_TYPE="Cnn14_16k"
CUDA_VISIBLE_DEVICES=0 python3 pytorch/inference.py audio_tagging \
    --sample_rate=16000 \
    --window_size=512 \
    --hop_size=160 \
    --mel_bins=64 \
    --fmin=50 \
    --fmax=8000 \
    --model_type=$MODEL_TYPE \
    --checkpoint_path=$CHECKPOINT_PATH \
    --audio_path='resources/R9_ZSCveAHg_7s.wav' \
    --cuda
```

## Sound event detection using pretrained models
Some of PANNs such as DecisionLevelMax (the best), DecisionLevelAvg, DecisionLevelAtt) can be used for frame-wise sound event detection. For example, execute the following commands to inference sound event detection results on this [audio](resources/R9_ZSCveAHg_7s.wav):

```
CHECKPOINT_PATH="Cnn14_DecisionLevelMax_mAP=0.385.pth"
wget -O $CHECKPOINT_PATH https://zenodo.org/record/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1
MODEL_TYPE="Cnn14_DecisionLevelMax"
CUDA_VISIBLE_DEVICES=0 python3 pytorch/inference.py sound_event_detection \
    --model_type=$MODEL_TYPE \
    --checkpoint_path=$CHECKPOINT_PATH \
    --audio_path="resources/R9_ZSCveAHg_7s.wav" \
    --cuda
```

The visualization of sound event detection result looks like:
<img src="resources/sed_R9_ZSCveAHg_7s.png">

Please see https://www.youtube.com/watch?v=QyFNIhRxFrY for a sound event detection demo.

For those users who only want to use the pretrained models for inference, we have prepared a **panns_inference** tool which can be easily installed by:
```
pip install panns_inference
```

Please visit https://github.com/qiuqiangkong/panns_inference for details of panns_inference.

