import os
import pickle
import numpy as np
import argparse
import time
import logging

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
 
from panns.utils.logging_utils import create_logging
from panns.data.mixup import mixup_coefficients, mixup
import panns.models
from panns.evaluate import evaluate
from panns.data.dataset import AudioSetDataset


def train(*, hdf5_files_path_train,
          target_weak_path_train,
          hdf5_files_path_eval,
          target_weak_path_eval,
          model,
          logs_dir=None,
          checkpoints_dir=None,
          statistics_dir=None,
          augmentation=False, mixup_alpha=1.0,
          batch_size=32, learning_rate=1e-3, resume_iteration=0,
          resume_checkpoint_path=None, iter_max=1000000,
          cuda=False, num_workers=8):
    """Train AudioSet tagging model. 

    Models are saved to and loaded from 'checkpoints' using torch.save/torch.load respectively.
    A checkpoint is a dictionary containng following keys:
        * iteration: counter of the iteration correponding to the checkpoint
        * model: state_dict of the model
        * sampler: state_dict of the sampler
        * statistics: list of statistics (average_precision and auc) for evaluation set at each iteration

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

    if resume_iteration > 0 and resume_checkpoint_path is None:
        raise ValueError("resume_iteration is greater than 0, but no resume_checkpoint_path was given.")

    # Paths
    workspace = os.getcwd()
    if checkpoints_dir is None:
        checkpoints_dir = os.path.join(workspace, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    if statistics_dir is None:
        statistics_dir = os.path.join(workspace, 'statistics')
    os.makedirs(statistics_dir, exist_ok=True)

    if logs_dir is None:
        logs_dir = os.path.join(workspace, 'logs')
    create_logging(logs_dir, filemode='w')
    
    if 'cuda' in str(device):
        logging.info('Using GPU.')
        logging.info('GPU number: {}'.format(torch.cuda.device_count()))
    else:
        logging.info('Using CPU. Set --cuda flag to use GPU.')

    # Dataset will be used by DataLoader later. Dataset takes a meta as input
    # and return a waveform and a target.
    train_dataset = AudioSetDataset(hdf5_files_path_train, target_weak_path_train)
    eval_dataset = AudioSetDataset(hdf5_files_path_eval, target_weak_path_eval)

    # TODO add parameter pin_memory_device when torch 1.13 is supported
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers,
                                               persistent_workers=True,
                                               pin_memory=True)

    eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              persistent_workers=True,
                                              pin_memory=True)

    if augmentation:
        mixup_augmenter = mixup_coefficients(mixup_alpha=mixup_alpha,
                                             batch_size=batch_size)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, 
                           betas=(0.9, 0.999), eps=1e-08, weight_decay=0.,
                           amsgrad=True)

    # Resume training
    if resume_iteration > 0:
        logging.info('Loading checkpoint {}'.format(resume_checkpoint_path))
        checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(checkpoint['model'])
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

    for data, target in train_loader:
        # Mixup lambda
        mixup_lambda = next(mixup_augmenter) if augmentation else None

        # Forward
        model.train()
        train_bgn_time = time.time()

        clipwise_output, _ = model(data, mixup_lambda)

        target = mixup(target, mixup_lambda) if augmentation else target
        target = torch.tensor(target, device=device)
        # Loss
        loss = F.binary_cross_entropy(clipwise_output, target)

        # Backward
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        
        train_time = time.time() - train_bgn_time
        logging.info(f'--- Iteration: {iteration}, training time: '
                     f'{train_time:.3f} s, training loss: {loss.item()}')

        if iteration > 0 and iteration % 2000 == 0:
            # Evaluate
            val_begin_time = time.time()
            eval_average_precision, eval_auc = evaluate(model, eval_loader)
                            
            statistics[iteration] = (eval_average_precision, eval_auc)

            validate_time = time.time() - val_begin_time

            logging.info(
                f'--- Iteration: {iteration}, validate time:'
                f' {validate_time:.3f} s, validate mAP: {np.mean(eval_average_precision):.3f}')

        
        # Save model/Stop learning
        if iteration % 100000 == 0 or iteration == iter_max:
            # Save model
            checkpoint = {
                'iteration': iteration, 
                'model': model.module.state_dict(), 
                'statistics': statistics}

            checkpoint_name = f"checkpoint_iteration={iteration}.pth"
            checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
                
            torch.save(checkpoint, checkpoint_path)
            statistics_name = f"statistics_iteration={iteration}.pickle"
            statistics_path = os.path.join(statistics_dir, statistics_name)
            pickle.dump(statistics, open(statistics_path, 'wb'))
            logging.info(f'Model saved to {checkpoint_path}')
            if iteration == iter_max: break # Stop learning

        iteration += 1
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5_files_path_train', type=str, required=True,
                        help="Path to hdf5 file of the train split")
    parser.add_argument('--target_weak_path_train', type=str, required=True,
                        help="Path to the weak target array of the train split")
    parser.add_argument('--hdf5_files_path_eval', type=str, required=True,
                        help="Path to hdf5 file of the eval split")
    parser.add_argument('--target_weak_path_eval', type=str, required=True,
                        help="Path to the weak target array of the eval split")
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

    model = panns.models.load_model(args.model_type, args.sample_rate,
                                    args.window_size, args.hop_size,
                                    args.mel_bins, args.fmin, args.fmax,
                                    args.classes_num)

    train(hdf5_files_path_train=args.hdf5_files_path_train,
          target_weak_path_train=args.target_weak_path_train,
          hdf5_files_path_eval=args.hdf5_files_path_eval,
          target_weak_path_eval=args.target_weak_path_eval,
          model=model,
          logs_dir=args.logs_dir,
          checkpoints_dir=args.checkpoints_dir,
          statistics_dir=args.statistics_dir,
          augmentation=args.augmentation,
          mixup_alpha=args.mixup_alpha,
          batch_size=args.batch_size,
          learning_rate=args.learning_rate,
          resume_iteration=args.resume_iteration,
          resume_checkpoint_path=args.resume_checkpoint_path,
          iter_max=args.iter_max,
          cuda=args.cuda,
          num_workers=args.num_workers)
