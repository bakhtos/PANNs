import os
import pickle
import argparse
import time
import logging

import numpy as np
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
          mixup_alpha=None,
          batch_size=32, learning_rate=1e-3,
          iter_max=1000000,
          cuda=False, num_workers=8):
    """Train AudioSet tagging model. 

    Args:
        hdf5_files_path_train : str,
            Path to hdf5 compression of the train split of the dataset
        target_weak_path_train : str,
            Path to the .npy file containing weak target array of the
            train split of the dataset
        hdf5_files_path_eval : str,
            Path to hdf5 compression of the evaluation split of the dataset
        target_weak_path_eval : str,
            Path to the .npy file containing weak target array of the
            evaluation split of the dataset
        model : torch.nn.Module,
            Model to train (one of the model classes defined in panns.models.models)
        logs_dir : str,
            Directory to save the logs into (will be created if doesn't exist);
            if None, a directory 'logs' will be created in CWD (default None)
        checkpoints_dir : str,
            Directory to save model's checkpoints into (will be created if doesn't exist);
            if None a directory 'checkpoints' will be created in CWD (default None)
        statistics_dir : str,
            Directory to save evaluation statistics into (will be created if doesn't exist);
            if None a directory 'statistics' will be created in CWD (default None)
        mixup_alpha : float, Alpha parameter for Mixup;
            if None, mixup not used (default None)
        batch_size : int, Batch size to use for training/evaluation (default 32)
        learning_rate : float, Learning rate to use in training (default 1e-3)
        iter_max : bool, Train until this iteration (default 100000)
        cuda : bool, If True, use GPU for training (default False)
        num_workers : int, Amount of workers to pass to
            torch.utils.data.DataLoader() (default 32)
    """

    # Augmentation
    aug = mixup_alpha is not None

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

    def save_model(model, statistics, iteration,
                   checkpoints_dir=checkpoints_dir,
                   statistics_dir=statistics_dir):
        # Save model
        checkpoint_name = f"checkpoint_iteration={iteration}.pth"
        checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
        torch.save(model.module.state_dict(), checkpoint_path)
        logging.info(f'--- Iteration: {iteration}, Model saved to'
                     f' {checkpoint_path}')

        # Save statistics
        statistics_name = f"statistics_iteration={iteration}.pickle"
        statistics_path = os.path.join(statistics_dir, statistics_name)
        pickle.dump(statistics, open(statistics_path, 'wb'))
        logging.info(f'--- Iteration: {iteration}, Statistics saved to'
                     f' {statistics_path}')

    # Dataset
    train_dataset = AudioSetDataset(hdf5_files_path_train, target_weak_path_train)
    eval_dataset = AudioSetDataset(hdf5_files_path_eval, target_weak_path_eval)

    # TODO add parameter pin_memory_device when torch 1.13 is supported
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=num_workers,
                                               persistent_workers=True,
                                               pin_memory=True)

    eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              persistent_workers=True,
                                              pin_memory=True)

    mixup_augmenter = mixup_coefficients(mixup_alpha=mixup_alpha,
                                         batch_size=batch_size) if aug else None
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, 
                           betas=(0.9, 0.999), eps=1e-08, weight_decay=0.,
                           amsgrad=True)

    # Device
    if cuda:
        device = torch.device('cuda')
        logging.info('Using GPU.')
        logging.info('GPU number: {}'.format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
        model.to(device)
    else:
        device = torch.device('cpu')
        logging.info('Using CPU. Set --cuda flag to use GPU.')

    iteration = 0
    statistics = {}

    for data, target in train_loader:
        # Data augmentation
        mixup_lambda = next(mixup_augmenter) if aug else None
        target = mixup(target, mixup_lambda) if aug else target
        target = torch.tensor(target, device=device)

        train_bgn_time = time.time()

        # Train
        model.train()
        clipwise_output, _ = model(data, mixup_lambda)

        loss = F.binary_cross_entropy(clipwise_output, target)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        
        train_time = time.time() - train_bgn_time
        logging.info(f'--- Iteration: {iteration}, training time: '
                     f'{train_time:.3f} s, training loss: {loss.item()}')

        # Evaluate
        if iteration > 0 and iteration % 2000 == 0:
            val_begin_time = time.time()
            eval_average_precision, eval_auc = evaluate(model, eval_loader)
            validate_time = time.time() - val_begin_time

            statistics[iteration] = (eval_average_precision, eval_auc)

            logging.info(
                f'--- Iteration: {iteration}, validate time:'
                f' {validate_time:.3f} s, validate mAP: '
                f'{np.mean(eval_average_precision):.3f}')

        # Save model/Stop training
        if iteration == iter_max:
            save_model(model, statistics, iteration)
            break
        elif iteration % 100000 == 0:
            save_model(model, statistics, iteration)

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
    parser.add_argument('--statistics_dir', type=str,
                        help="Directory to save evaluation statistics into")
    parser.add_argument('--win_length', type=int, default=1024,
                        help="Window size of filter to be used in training ("
                             "default 1024)")
    parser.add_argument('--hop_size', type=int, default=320,
                        help="Hop size of filter to be used in training ("
                             "default 320)")
    parser.add_argument('--sample_rate', type=int, default=32000,
                        help="Sample rate of the used audio clips; supported "
                             "values are 32000, 16000, 8000 (default 32000)")
    parser.add_argument('--clip_length', type=int, default=10000,
                        help="Length (in ms) of audio clips used in the "
                             "dataset (default 10000)")
    parser.add_argument('--fmin', type=int, default=50,
                        help="Minimum frequency to be used when creating "
                             "Logmel filterbank (default 50)")
    parser.add_argument('--fmax', type=int, default=14000,
                        help="Maximum frequency to be used when creating "
                             "Logmel filterbank (default 14000)")
    parser.add_argument('--mel_bins', type=int, default=64,
                        help="Amount of mel filters to use in the filterbank "
                             "(default 64)")
    parser.add_argument('--mixup_alpha', type=float, default=None,
                        help="If using augmentation, use this as alpha "
                             "parameter for Mixup (default 1.0)")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size to use for training/evaluation ("
                             "default 32)")
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help="Learning rate to use in training (default 1e-3)")
    parser.add_argument('--resume_checkpoint_path', type=str, default=None,
                        help="Read a checkpoint from this path")
    parser.add_argument('--iter_max', type=int, default=1000000,
                        help="Train until this iteration (default 1000000)")
    parser.add_argument('--cuda', action='store_true', default=False,
                        help="If set, try to use GPU for training")
    parser.add_argument('--classes_num', type=int, default=110,
                        help="Amount of classes used in the dataset (default 110)")
    parser.add_argument('--num_workers', type=int, default=8,
                        help="Amount of workers to pass to "
                             "torch.utils.data.DataLoader (default 8)")
    
    args = parser.parse_args()

    model = panns.models.load_model(args.model_type, args.sample_rate,
                                    args.win_length, args.hop_size,
                                    args.mel_bins, args.fmin, args.fmax,
                                    args.classes_num,
                                    args.resume_checkpoint_path)

    train(hdf5_files_path_train=args.hdf5_files_path_train,
          target_weak_path_train=args.target_weak_path_train,
          hdf5_files_path_eval=args.hdf5_files_path_eval,
          target_weak_path_eval=args.target_weak_path_eval,
          model=model,
          logs_dir=args.logs_dir,
          checkpoints_dir=args.checkpoints_dir,
          statistics_dir=args.statistics_dir,
          mixup_alpha=args.mixup_alpha,
          batch_size=args.batch_size,
          learning_rate=args.learning_rate,
          iter_max=args.iter_max,
          cuda=args.cuda,
          num_workers=args.num_workers)
