import os
import sys
import pickle
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import time
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
 
from utils.file_utils import create_folder, get_filename
from utils.logger import create_logging
from utils.mixup import Mixup
from utils.stat_container import StatisticsContainer
from models import MODELS
from pytorch_utils import move_data_to_device, count_parameters, count_flops, 
    do_mixup, evaluate
from data_generator import AudioSetDataset, SAMPLERS, collate_fn
from losses import LOSS_FUNCS


def train(train_indexes_hdf5_path, eval_indexes_hdf5_path,
          logs_dirs=None, checkpoints_dir=None, statistics_path=None,
          window_size, hop_size, sample_rate,
          fmin, fmax, mel_bins, model_type, loss, sampler, augmentation,
          batch_size, learning_rate, resume_iteration, early_stop,
          accumulation_steps, cuda, filename, classes_num):
    """Train AudioSet tagging model. 

    Args:
      window_size: int
      hop_size: int
      mel_bins: int
      model_type: str
      loss: 'clip_bce'
      sampler: Name of the Sampler Class
      augmentation: True or False
      batch_size: int
      learning_rate: float
      resume_iteration: int
      early_stop: int
      accumulation_steps: int
      cuda: bool
    """

    device = torch.device('cuda') if (cuda and torch.cuda.is_available()) else torch.device('cpu')

    num_workers = 8
    clip_samples = sample_rate*10
    loss_func = LOSS_FUNCS[loss]

    # Paths

    workspace = os.getcwd()
    if checkpoints_dir is None:
        checkpoints_dir = os.path.join(workspace, 'checkpoints', filename, 
        'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
        'data_type={}'.format(data_type), model_type, 
        'loss_type={}'.format(loss), 'sampler={}'.format(sampler), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size))
    create_folder(checkpoints_dir)
    
    if statistics_path is None:
        statistics_path = os.path.join(workspace, 'statistics', filename, 
        'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
        'data_type={}'.format(data_type), model_type, 
        'loss_type={}'.format(loss), 'sampler={}'.format(sampler), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
        'statistics.pkl')
    create_folder(os.path.dirname(statistics_path))

    if logs_dir is None:
        logs_dir = os.path.join(workspace, 'logs', filename, 
        'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
        'data_type={}'.format(data_type), model_type, 
        'loss_type={}'.format(loss), 'sampler={}'.format(sampler), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size))
    create_logging(logs_dir, filemode='w')
    
    if 'cuda' in str(device):
        logging.info('Using GPU.')
        device = 'cuda'
    else:
        logging.info('Using CPU. Set --cuda flag to use GPU.')
        device = 'cpu'
    
    # Model
    Model = MODELS[model_type]
    model = Model(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num)
     
    # Dataset will be used by DataLoader later. Dataset takes a meta as input 
    # and return a waveform and a target.
    dataset = AudioSetDataset(sample_rate=sample_rate)

    # Train sampler
    Sampler = SAMPLERS[sampler]
     
    train_sampler = Sampler(
        indexes_hdf5_path=train_indexes_hdf5_path, 
        batch_size=batch_size * 2 if augmentation else batch_size)
    
    # Evaluate sampler
    eval_sampler = EvaluateSampler(
        indexes_hdf5_path=eval_indexes_hdf5_path, batch_size=batch_size)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=train_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)
    
    eval_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=eval_test_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    if augmentation:
        mixup_augmenter = Mixup(mixup_alpha=1.)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, 
        betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)

    train_bgn_time = time.time()
    
    # Resume training
    if resume_iteration > 0:
        resume_checkpoint_path = os.path.join(workspace, 'checkpoints', filename, 
            'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
            'data_type={}'.format(data_type), model_type, 
            'loss_type={}'.format(loss), 'sampler={}'.format(sampler), 
            'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
            '{}_iterations.pth'.format(resume_iteration))

        logging.info('Loading checkpoint {}'.format(resume_checkpoint_path))
        checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        train_sampler.load_state_dict(checkpoint['sampler'])
        statistics = pickle.load(open(statistics_path, 'rb'))
        statistics = statistics[0:resume_iteration+1]
        iteration = checkpoint['iteration']

    else:
        iteration = 0
    
    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if 'cuda' in str(device):
        model.to(device)
    
    time1 = time.time()
    
    for batch_data_dict in train_loader:
        """batch_data_dict: {
            'audio_name': (batch_size [*2 if mixup],), 
            'waveform': (batch_size [*2 if mixup], clip_samples), 
            'target': (batch_size [*2 if mixup], classes_num), 
            (ifexist) 'mixup_lambda': (batch_size * 2,)}
        """
        
        # Evaluate

        model.train(False)

        train_fin_time = time.time()

        eval_average_precision, eval_auc = evaluate(model, eval_loader)
                            
        logging.info('Validate test mAP: {:.3f}'.format(
            np.mean(eval_average_precision)))

        statistics.append((eval_average_precision, eval_auc))

        train_time = train_fin_time - train_bgn_time
        validate_time = time.time() - train_fin_time

        logging.info(
            'iteration: {}, train time: {:.3f} s, validate time: {:.3f} s'
                ''.format(iteration, train_time, validate_time))

        logging.info('------------------------------------')

        train_bgn_time = time.time()
        
        # Save model
        if iteration % 100000 == 0:
            checkpoint = {
                'iteration': iteration, 
                'model': model.module.state_dict(), 
                'sampler': train_sampler.state_dict()}

            checkpoint_path = os.path.join(
                checkpoints_dir, '{}_iterations.pth'.format(iteration))
                
            torch.save(checkpoint, checkpoint_path)
            pickle.dump(statistics, open(statistics_path, 'wb')
            logging.info('Model saved to {}'.format(checkpoint_path))
        
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
        model.train(True)

        batch_output_dict = model(batch_data_dict['waveform'], 
             batch_data_dict['mixup_lambda'])
        """{'clipwise_output': (batch_size, classes_num), ...}"""

        if augmentation:
            target = do_mixup(batch_data_dict['target'], batch_data_dict['mixup_lambda'])
        else:
            target = batch_data_dict['target']

        # Loss
        loss = loss_func(batch_output_dict,target)

        # Backward
        loss.backward()
        print(loss)
        
        optimizer.step()
        optimizer.zero_grad()
        
        if iteration % 10 == 0:
            print('--- Iteration: {}, train time: {:.3f} s / 10 iterations ---'\
                .format(iteration, time.time() - time1))
            time1 = time.time()
        
        # Stop learning
        if iteration == early_stop:
            break

        iteration += 1
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_indexes_hdf5_path", type=str, required=True)
    parser.add_argument('--eval_indexes_hdf5_path", type=str, required=True)
    parser.add_argument('--logs_dir', type=str)
    parser.add_argument('--checkpoints_dir', type=str)
    parser.add_argument('--statistics_path', type=str)
    parser.add_argument('--sample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=1024)
    parser.add_argument('--hop_size', type=int, default=320)
    parser.add_argument('--mel_bins', type=int, default=64)
    parser.add_argument('--fmin', type=int, default=50)
    parser.add_argument('--fmax', type=int, default=14000) 
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--loss', type=str, default='clip_bce', choices=['clip_bce'])
    parser.add_argument('--sampler', type=str, default='BalancedTrainSampler', choices=['TrainSampler', 'BalancedTrainSampler', 'AlternateTrainSampler'])
    parser.add_argument('--augmentation', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--resume_iteration', type=int, default=0)
    parser.add_argument('--early_stop', type=int, default=1000000)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--classes_num', type=int, default=110)
    
    args = parser.parse_args()
    filename = get_filename(__file__)

    train(workspace=args.workspace, data_type=args.data_type, fmin=args.fmin,
          fmax=args.fmax, sample_rate=args.sample_rate, window_size=args.window_size,
          hop_size=args.hop_size, mel_bins=args.mel_bins, model_type=args.model_type,
          loss=args.loss, sampler=args.sampler, augmentation=args.augmentation,
          batch_size=args.batch_size, learning_rate=args.learning_rate, cuda=args.cuda,
          resume_iteration=args.resume_iteration, early_stop=args.early_stop,
          classes_num=args.classes_num, filename=filename)
