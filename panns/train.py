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
 
from panns.logging import create_logging
from panns.models import load_model, model_parser
from panns.evaluate import evaluate
from panns.data.dataset import AudioSetDataset


def train(*, train_dataset,
          eval_dataset,
          model,
          checkpoints_dir=None,
          statistics_dir=None,
          label_type='weak',
          batch_size=32, learning_rate=1e-3,
          iter_max=1000000,
          cuda=False, num_workers=8):
    """Train AudioSet tagging model. 

    Args:
        model : torch.nn.Module,
            Model to train (one of the model classes defined in panns.models.models)
        train_dataset : torch.utils.data.Dataset,
            Dataset object which provides (data, target) batches for train split
        eval_dataset : torch.utils.data.Dataset,
            Dataset object which provides (data, target) batches for eval split
        checkpoints_dir : str,
            Directory to save model's checkpoints into (will be created if doesn't exist);
            if None a directory 'checkpoints' will be created in CWD (default None)
        statistics_dir : str,
            Directory to save evaluation statistics into (will be created if doesn't exist);
            if None a directory 'statistics' will be created in CWD (default None)
        label_type: str, Either 'weak' or 'strong' - which labels are used
                         to calculate loss
        batch_size : int, Batch size to use for training/evaluation (default 32)
        learning_rate : float, Learning rate to use in training (default 1e-3)
        iter_max : bool, Train until this iteration (default 100000)
        cuda : bool, If True, use GPU for training (default False)
        num_workers : int, Amount of workers to pass to
            torch.utils.data.DataLoader() (default 32)
    """

    # Paths
    workspace = os.getcwd()
    if checkpoints_dir is None:
        checkpoints_dir = os.path.join(workspace, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    if statistics_dir is None:
        statistics_dir = os.path.join(workspace, 'statistics')
    os.makedirs(statistics_dir, exist_ok=True)

    def save_model(model, iteration, checkpoints_dir):
        checkpoint_name = f"checkpoint_iteration={iteration}.pth"
        checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
        torch.save(model.module.state_dict(), checkpoint_path)
        logging.info(f'--- Iteration: {iteration}, Model saved to'
                     f' {checkpoint_path}')

    def save_statistics(statistics, iteration, statistics_dir):
        statistics_name = f"statistics_iteration={iteration}.pickle"
        statistics_path = os.path.join(statistics_dir, statistics_name)
        pickle.dump(statistics, open(statistics_path, 'wb'))
        logging.info(f'--- Iteration: {iteration}, Statistics saved to'
                     f' {statistics_path}')


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

    train_iter = iter(train_loader)
    for iteration in range(iter_max+1):
        try:
            data, target = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            data, target = next(train_iter)
        # Data augmentation
        data = data.to(device)
        target = target.to(device)

        train_bgn_time = time.time()

        # Train
        model.train()
        clipwise_output, segmentwise_output, framewise_output, embedding = \
            model(data)

        loss = F.binary_cross_entropy(clipwise_output, target) if label_type \
               == 'weak' else F.binary_cross_entropy(framewise_output, target)
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

            logging.info(
                f'--- Iteration: {iteration}, validate time:'
                f' {validate_time:.3f} s, validate mAP: '
                f'{np.mean(eval_average_precision):.3f}')

            save_statistics((eval_average_precision, eval_auc), iteration,
                            statistics_dir)

        # Save model/Stop training
        if iteration == iter_max:
            save_model(model, iteration, checkpoints_dir)
            break
        elif iteration % 100000 == 0:
            save_model(model, iteration, checkpoints_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(parents=[model_parser])
    files = parser.add_argument_group('Files', 'Arguments to specify paths '
                                               'to necessary files')
    files.add_argument('--hdf5_files_path_train', type=str, required=True,
                       help="Path to hdf5 file of the train split")
    files.add_argument('--target_path_train', type=str, required=True,
                       help="Path to the weak target array of the train split")
    files.add_argument('--hdf5_files_path_eval', type=str, required=True,
                       help="Path to hdf5 file of the eval split")
    files.add_argument('--target_path_eval', type=str, required=True,
                       help="Path to the weak target array of the eval split")
    files.add_argument('--logs_dir', type=str, help="Directory to save the "
                                                    "logs into")
    files.add_argument('--checkpoints_dir', type=str,
                       help="Directory to save neural net's checkpoints into")
    files.add_argument('--statistics_dir', type=str,
                       help="Directory to save evaluation statistics into")
    files.add_argument('--resume_checkpoint_path', type=str, default=None,
                       help="Read a checkpoint from this path")
    training = parser.add_argument_group("Training", "Parameters to customize "
                                                     "training")
    training.add_argument('--label_type', choices=['weak', 'strong'],
                          default='weak', help='Use weak/strong labels to '
                                               'calculate loss')
    training.add_argument('--batch_size', type=int, default=32,
                          help="Batch size to use for training/evaluation ("
                               "default 32)")
    training.add_argument('--learning_rate', type=float, default=1e-3,
                          help="Learning rate to use in training (default "
                               "1e-3)")
    training.add_argument('--iter_max', type=int, default=1000000,
                          help="Train until this iteration (default 1000000)")
    training.add_argument('--cuda', action='store_true', default=False,
                          help="If set, try to use GPU for training")
    training.add_argument('--num_workers', type=int, default=8,
                          help="Amount of workers to pass to "
                               "torch.utils.data.DataLoader (default 8)")

    args = parser.parse_args()

    spec_aug = args.spec_aug or args.no_spec_aug
    mixup_time = args.mixup_time or args.no_mixup_time
    mixup_freq = args.mixup_freq or args.no_mixup_freq
    dropout = args.dropout or args.no_dropout
    wavegram = args.wavegram or args.no_wavegram
    spectrogram = args.spectrogram or args.no_spectrogram
    center = args.center or args.no_center

    if args.label_type == 'strong':
        if args.decision_level is None:
            raise ValueError('Strong labels are used but no decision_level '
                             'function specified (--decision_level parameter)')

    model = load_model(model=args.model_type,
                       checkpoint=args.resume_checkpoint_path,
                       spec_aug=spec_aug, mixup_time=mixup_time,
                       mixup_freq=mixup_freq, dropout=dropout,
                       wavegram=wavegram, spectrogram=spectrogram,
                       decision_level=args.decision_level, center=center,
                       sample_rate=args.sample_rate,
                       win_length=args.win_length, hop_length=args.hop_length,
                       n_mels=args.n_mels, f_min=args.f_min, f_max=args.f_max,
                       pad_mode=args.pad_mode, top_db=args.top_db,
                       num_features=args.num_features,
                       embedding_size=args.embedding_size,
                       classes_num=args.classes_num)

    if args.logs_dir is None:
        args.logs_dir = os.path.join(os.getcwd(), 'logs')
    create_logging(args.logs_dir, filemode='w')

    train_dataset = AudioSetDataset(args.hdf5_files_path_train,
                                    args.target_path_train)
    eval_dataset = AudioSetDataset(args.hdf5_files_path_eval,
                                   args.target_path_eval)

    train(train_dataset=train_dataset,
          eval_dataset=eval_dataset,
          model=model,
          checkpoints_dir=args.checkpoints_dir,
          statistics_dir=args.statistics_dir,
          label_type=args.label_type,
          batch_size=args.batch_size,
          learning_rate=args.learning_rate,
          iter_max=args.iter_max,
          cuda=args.cuda,
          num_workers=args.num_workers)

    del train_dataset
    del eval_dataset
