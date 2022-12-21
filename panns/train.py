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
from panns.models import load_model, model_parser, TransferModel
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
          label_type='weak',
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
        label_type: str, Either 'weak' or 'strong' - which labels are used
                         to calculate loss
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
        clipwise_output, segmentwise_output, framewise_output, embedding = \
            model(data, mixup_lambda)

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

    parser = argparse.ArgumentParser(parents=[model_parser])
    files = parser.add_argument_group('Files', 'Arguments to specify paths '
                                               'to necessary files')
    files.add_argument('--hdf5_files_path_train', type=str, required=True,
                       help="Path to hdf5 file of the train split")
    files.add_argument('--target_weak_path_train', type=str, required=True,
                       help="Path to the weak target array of the train split")
    files.add_argument('--hdf5_files_path_eval', type=str, required=True,
                       help="Path to hdf5 file of the eval split")
    files.add_argument('--target_weak_path_eval', type=str, required=True,
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
    training.add_argument('--mixup_alpha', type=float, default=None,
                          help="If using augmentation, use this as alpha "
                               "parameter for Mixup (default 1.0)")
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
    transfer = parser.add_argument_group('Transfer', 'Transfer a pre-trained '
                                                     'model to another task')
    transfer.add_argument('--transfer', action='store_true', required=False,
                          help='If set, transfer the pre-trained model from a '
                               'checkpoint to a new task with new amount '
                               'of classes')
    transfer.add_argument('--classes_num_new', type=int, required=False,
                          help='New amount of classes for the transfer task')
    transfer.add_argument('--interpolate_ratio', type=int, default=32,
                          required=False, help='Ratio to interpolate '
                                               'framewise_output')
    freeze = transfer.add_mutually_exclusive_group()
    freeze.add_argument('--freeze_base', action='store_true', required=False,
                        help='If set, freeze the parameters of base model '
                             'during fine-tuning')
    freeze.add_argument('--no_freeze_base', action='store_false',
                        required=False, help='If set, do not freeze '
                                             'parameters of the base model '
                                             'during fine-tuning')
    freeze.add_argument('--clip_length', type=int, default=10000,
                        required=False, help='Length (in ms) of clips used in'
                                             'the dataset')
    args = parser.parse_args()

    spec_aug = args.spec_aug or args.no_spec_aug
    mixup_time = args.mixup_time or args.no_mixup_time
    mixup_freq = args.mixup_freq or args.no_mixup_freq
    dropout = args.dropout or args.no_dropout
    wavegram = args.wavegram or args.no_wavegram
    spectrogram = args.spectrogram or args.no_spectrogram
    center = args.center or args.no_center
    freeze = args.freeze_base or args.no_freeze_base

    if args.label_type == 'strong':
        if args.decision_level is None:
            raise ValueError('Strong labels are used but no decision_level '
                             'function specified (--decision_level parameter)')
    if args.transfer:
        if args.resume_checkpoint_path is None:
            raise ValueError('Transfer flag is set, but no model checkpoint '
                             'is provided')
        if args.classes_num_new is None:
            raise ValueError('Transfer flag is set, but number of new classes '
                             'is not provided')

    model = load_model(model=args.model_type,
                       checkpoint=args.resume_checkpoint_path,
                       spec_aug=spec_aug, mixup_time=mixup_time,
                       mixup_freq=mixup_freq, dropout=dropout,
                       wavegram=wavegram, spectrogram=spectrogram,
                       decision_level=args.decision_level, center=center,
                       win_length=args.win_length, hop_length=args.hop_length,
                       n_mels=args.n_mels, f_min=args.f_min, f_max=args.f_max,
                       pad_mode=args.pad_mode, top_db=args.top_db,
                       num_features=args.num_features,
                       embedding_size=args.embedding_size,
                       classes_num=args.classes_num)

    if args.transfer:
        frames_num = (args.clip_length // 1000) * (args.sample_rate //
                                                   args.hop_length) if \
                     args.decision_level is not None else None
        model = TransferModel(model, args.classes_num_new,
                              frames_num=frames_num,
                              interpolate_ratio=args.interpolate_ratio,
                              decision_level=args.decision_level,
                              freeze_base=freeze,
                              embedding_size=args.embedding_size)

    train(hdf5_files_path_train=args.hdf5_files_path_train,
          target_weak_path_train=args.target_weak_path_train,
          hdf5_files_path_eval=args.hdf5_files_path_eval,
          target_weak_path_eval=args.target_weak_path_eval,
          model=model,
          logs_dir=args.logs_dir,
          checkpoints_dir=args.checkpoints_dir,
          statistics_dir=args.statistics_dir,
          label_type=args.label_type,
          mixup_alpha=args.mixup_alpha,
          batch_size=args.batch_size,
          learning_rate=args.learning_rate,
          iter_max=args.iter_max,
          cuda=args.cuda,
          num_workers=args.num_workers)
