import os
import pickle
import numpy as np
import argparse
import time
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
 
from panns.utils.logging_utils import create_logging
from panns.data.mixup import Mixup, do_mixup
import panns.models
from panns.utils.pytorch_utils import move_data_to_device, count_parameters, count_flops
from panns.evaluate import evaluate
import panns.data.loaders


def train(*, train_indexes_hdf5_path,
          eval_indexes_hdf5_path,
          model_type,
          logs_dir=None,
          checkpoints_dir=None,
          statistics_dir=None,
          window_size=1024, hop_size=320, sample_rate=32000, clip_length=10000,
          fmin=50, fmax=14000, mel_bins=64,
          sampler='TrainSampler',
          augmentation=False, mixup_alpha=1.0,
          batch_size=32, learning_rate=1e-3, resume_iteration=0,
          resume_checkpoint_path=None, iter_max=1000000,
          cuda=False, classes_num=110,
          num_workers=8):
    """Train AudioSet tagging model. 

    Models are saved to and loaded from 'checkpoints' using torch.save/torch.load respectively.
    A checkpoint is a dictionary containng following keys:
        * iteration: counter of the iteration correponding to the checkpoint
        * model: state_dict of the model
        * sampler: state_dict of the sampler
        * statistics: list of statistics (average_precision and auc) for evaluation set at each iteration

    :param str train_indexes_hdf5_path: Path to hdf5 index of the train set
    :param str eval_indexes_hdf5_path: Path to hdf5 index of the evaluation set
    :param str model_type: Name of model to train (one of the model classes defined in models.py)
    :param str logs_dir: Directory to save the logs into (will be created if doesn't exist already), if None a directory 'logs' will be created in CWD  (default None)
    :param str checkpoints_dir: Directory to save neural net's checkpoints into (will be created if doesn't exist already), if None a directory 'checkpoints' will be created in CWD (default None)
    :param str statistics_dir: Directory to save evaluation statistics into (will be created if doesn't exist already), if None a directory 'statistics' will be created in CWD (default None) NOTE: statistics are also saved into checkpoints
    :param int window_size: Window size of filter to be used in training (default 1024)
    :param int hop_size: Hop size of filter to be used in traning (default 320)
    :param int sample_rate: Sample rate of the used audio clips; supported values are 32000, 16000, 8000 (default 32000)
    :param int clip_length: Length (in ms) of Audio clips user in dataset (default 10000)
    :param int fmin: Minimum frequency to be used when creating Logmel filterbank (default 50)
    :param int fmax: Maximum frequency to be used when creating Logmel filterbank (default 14000)
    :param int mel_bins: Amount of mel filters to use in the filterbank (default 64)
    :param str sampler: The sampler for the dataset to use for training ('TrainSampler' (default)|'BalancedTrainSampler'|'AlternateTrainSampler')
    :param bool augmentation: If True, use Mixup for data augmentation (default False)
    :param float mixup_alpha: If using augmentation, use this as alpha parameter for Mixup (default 1.0)
    :param int batch_size: Batch size to use for training/evaluation (default 32)
    :param float learning_rate: Learning rate to use in training (default 1e-3)
    :param int resume_iteration: If greater than 0, load a checkpoint and resume traning from this iteration (default 0)
    :param str resume_checkpoint_path: If resume_iteration is greater than 0, read a checkpoint to be resumed from this path (default None)
    :param int iter_max: Train until this iteration (default 1000000) 
    :param bool cuda: If True, try to use GPU for traning (default False)
    :param int classes_num: Amount of classes used in the dataset (default 110)
    :param int num_workers: Amount of workers to pass to torch.utils.data.DataLoader()
    :raises ValueError: if model_type or sampler not found among defined ones
    :raises ValueError: if resume_iteration is non-zero, but no resume_checkpoint_path given
    """

    device = torch.device('cuda') if (cuda and torch.cuda.is_available()) else torch.device('cpu')

    clip_samples = sample_rate*clip_length//1000
    
    if resume_iteration > 0 and resume_checkpoint_path is None:
        raise ValueError("resume_iteration is greater than 0, but no resume_checkpoint_path was given.")

    # Paths

    param_string = f"""sample_rate={sample_rate},window_size={window_size},\
hop_size={hop_size},mel_bins={mel_bins},fmin={fmin},fmax={fmax},model={model_type},\
sampler={sampler},augmentation={augmentation},batch_size={batch_size}"""

    workspace = os.getcwd()
    if checkpoints_dir is None:
        checkpoints_dir = os.path.join(workspace, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok)
    
    if statistics_dir is None:
        statistics_dir = os.path.join(workspace, 'statistics')
    os.makedirs(statistics_dir, exist_ok)

    if logs_dir is None:
        logs_dir = os.path.join(workspace, 'logs')
    create_logging(logs_dir, filemode='w')
    
    if 'cuda' in str(device):
        logging.info('Using GPU.')
        logging.info('GPU number: {}'.format(torch.cuda.device_count()))
        device = 'cuda'
    else:
        logging.info('Using CPU. Set --cuda flag to use GPU.')
        device = 'cpu'
    
    # Model
    if model_type in panns.models.__all__:
        Model = eval("panns.models."+model_type)
    else:
        raise ValueError(f"'{model_type}' is not among the defined models.")

    model = Model(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num)
     
    # Dataset will be used by DataLoader later. Dataset takes a meta as input 
    # and return a waveform and a target.
    dataset = panns.data.loaders.AudioSetDataset()

    # Train sampler
    if sampler in panns.data.loaders.__all__:
        Sampler = eval("panns.data.loaders."+sampler)
    else:
        raise ValueError(f"'{sampler}' is not among the defined samplers.")
     
    train_sampler = Sampler(
        indexes_hdf5_path=train_indexes_hdf5_path, 
        batch_size=batch_size * 2 if augmentation else batch_size)
    
    # Evaluate sampler
    eval_sampler = panns.data.loaders.EvaluateSampler(
        indexes_hdf5_path=eval_indexes_hdf5_path, batch_size=batch_size)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=train_sampler, collate_fn=panns.data.loaders.collate_fn, 
        num_workers=num_workers, pin_memory=True)
    
    eval_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=eval_sampler, collate_fn=panns.data.loaders.collate_fn, 
        num_workers=num_workers, pin_memory=True)

    if augmentation:
        mixup_augmenter = Mixup(mixup_alpha=mixup_alpha)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, 
        betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)

    train_bgn_time = time.time()
    
    # Resume training
    if resume_iteration > 0:
        logging.info('Loading checkpoint {}'.format(resume_checkpoint_path))
        checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        train_sampler.load_state_dict(checkpoint['sampler'])
        statistics = checkpoint['statistics']
        statistics = {k:v for k, v in statistics.items() if k < resume_iteration}
        iteration = checkpoint['iteration']

    else:
        iteration = 0
        statistics = {}
    
    # Parallel
    model = torch.nn.DataParallel(model)

    if device == 'cuda':
        model.to(device)
    
    
    for batch_data_dict in train_loader:
        """batch_data_dict: {
            'audio_name': (batch_size [*2 if mixup],), 
            'waveform': (batch_size [*2 if mixup], clip_samples), 
            'target': (batch_size [*2 if mixup], classes_num), 
            (ifexist) 'mixup_lambda': (batch_size * 2,)}
        """
        
        # Mixup lambda
        if augmentation:
            batch_data_dict['mixup_lambda'] = mixup_augmenter.get_lambda(
                batch_size=len(batch_data_dict['waveform']))
        else: 
            batch_data_dict['mixup_lambda'] = None

        # Move data to device
        for key in batch_data_dict.keys():
            batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)
        
        # Forward
        train_bgn_time = time.time()
        model.train()

        # 'embedding' is either embedding or framewise output, depends on model
        clipwise_output, embedding = model(batch_data_dict['waveform'], 
             batch_data_dict['mixup_lambda'])
        """clipwise_output: (batch_size, classes_num)"""

        if augmentation:
            target = do_mixup(batch_data_dict['target'], batch_data_dict['mixup_lambda'])
        else:
            target = batch_data_dict['target']

        # Loss
        loss = F.binary_cross_entropy(clipwise_output, target)

        # Backward
        loss.backward()
        logging.info(f"--- Iteration: {iteration}, training loss: {loss.item()}")
        
        optimizer.step()
        optimizer.zero_grad()
        
        train_fin_time = time.time()

        if iteration > 0 and iteration % 2000 == 0:
            # Evaluate

            eval_average_precision, eval_auc = evaluate(model, eval_loader)
                            
            statistics[iteration] = (eval_average_precision, eval_auc)

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info(
                f'--- Iteration: {iteration}, training time: {train_time:.3f} s, validate time: {validate_time:.3f} s, validate mAP: {np.mean(eval_average_precision):.3f}')

        
        # Save model/Stop learning
        if iteration % 100000 == 0 or iteration == iter_max:
            # Save model
            checkpoint = {
                'iteration': iteration, 
                'model': model.module.state_dict(), 
                'sampler': train_sampler.state_dict(),
                'statistics': statistics}

            checkpoint_name = "checkpoint_"+param_string+f",iteration={iteration}.pth"
            checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
                
            torch.save(checkpoint, checkpoint_path)
            statistics_name = "statistics_"+param_string+f",iteration={iteration}.pickle"
            statistics_path = os.path.join(statistics_dir, statistics_name)
            pickle.dump(statistics, open(statistics_path, 'wb'))
            logging.info(f'Model saved to {checkpoint_path}')
            if iteration == iter_max: break # Stop learning

        iteration += 1
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_indexes_hdf5_path', type=str, required=True,
                        help="Path to hdf5 index of the train set")
    parser.add_argument('--eval_indexes_hdf5_path', type=str, required=True,
                        help="Path to hdf5 index of the evaluation set")
    parser.add_argument('--model_type', type=str, required=True,
                        help="Name of model to train")
    parser.add_argument('--logs_dir', type=str, help="Directory to save the logs into")
    parser.add_argument('--checkpoints_dir', type=str,
                        help="Directory to save neural net's checkpoints into")
    parser.add_argument('--statistics_dir', type=str, help="Directory to save evaluation statistics into")
    parser.add_argument('--window_size', type=int, default=1024,
                        help="Window size of filter to be used in training (default 1024)")
    parser.add_argument('--hop_size', type=int, default=320,
                        help="Hop size of filter to be used in traning (default 320)")
    parser.add_argument('--sample_rate', type=int, default=32000,
                        help="Sample rate of the used audio clips; supported values are 32000, 16000, 8000 (default 32000)")
    parser.add_argument('--clip_length', type=int, default=10000,
                        help="Length (in ms) of audio clips used in the dataset (default 10000)")
    parser.add_argument('--fmin', type=int, default=50,
                        help="Minimum frequency to be used when creating Logmel filterbank (default 50)")
    parser.add_argument('--fmax', type=int, default=14000,
                        help="Maximum frequency to be used when creating Logmel filterbank (default 14000)")
    parser.add_argument('--mel_bins', type=int, default=64,
                        help="Amount of mel filters to use in the filterbank (default 64)")
    parser.add_argument('--sampler', type=str, default='TrainSampler', choices=['TrainSampler', 'BalancedTrainSampler', 'AlternateTrainSampler'],
                        help="The sampler for the dataset to use for training")
    parser.add_argument('--augmentation', action='store_true', default=False,
                        help="If set, use Mixup for data augmentation")
    parser.add_argument('--mixup_alpha', type=float, default=1.0,
                        help="If using augmentation, use this as alpha parameter for Mixup (default 1.0)")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size to use for training/evaluation (default 32)")
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help="Learning rate to use in training (default 1e-3)")
    parser.add_argument('--resume_iteration', type=int, default=0,
                        help="If greater than 0, load a checkpoint and resume traning from this iteration")
    parser.add_argument('--resume_checkpoint_path', type=str, default=None,
                        help="If --resume_iteration  is greater than zero, read a checkpoint from this path")
    parser.add_argument('--iter_max', type=int, default=1000000,
                        help="Train until this iteration (default 1000000)")
    parser.add_argument('--cuda', action='store_true', default=False,
                        help="If set, try to use GPU for traning")
    parser.add_argument('--classes_num', type=int, default=110,
                        help="Amount of classes used in the dataset (default 110)")
    parser.add_argument('--num_workers', type=int, default=8,
                        help="Amount of workers to pass to torch.utils.data.DataLoader (default 8)")
    
    args = parser.parse_args()

    train(train_indexes_hdf5_path=args.train_indexes_hdf5_path,
          eval_indexes_hdf5_path=args.eval_indexes_hdf5_path,
          model_type=args.model_type,
          logs_dir=args.logs_dir,
          checkpoints_dir=args.checkpoints_dir,
          statistics_dir=args.statistics_dir,
          window_size=args.window_size,
          hop_size=args.hop_size,
          sample_rate=args.sample_rate,
          clip_length=args.clip_length,
          fmin=args.fmin,
          fmax=args.fmax,
          mel_bins=args.mel_bins,
          sampler=args.sampler,
          augmentation=args.augmentation,
          mixup_alpha=args.mixup_alpha,
          batch_size=args.batch_size,
          learning_rate=args.learning_rate,
          resume_iteration=args.resume_iteration,
          resume_checkpoint_path=args.resume_checkpoint_path,
          iter_max=args.iter_max,
          cuda=args.cuda,
          classes_num=args.classes_num,
          num_workers=args.num_workers)
