import os
from typing import Union

import torch
import torch.nn as nn
from torch.utils.data import dataloader

from platform_classification.common import ImagePreprocessing, get_checkpoint_name, is_aborted
from platform_classification.dataset import ZipDataset

from .routine import train, create_and_save_result
from .model import Model
from .utils import GlobalStep
from .sender import Sender

import shutil




def start_train(dataset_path: str, batch_size: int, num_workers: int, learning_rate: float, base_model: str,
                num_epoch: int, log_frequ: int, save_each_step: int, path_to_save: str,
                train_path_to_init: Union[str, None], use_file_cache: bool = False, min_loss : float = 0.01):

    sender = Sender()

    image_preprocessing = ImagePreprocessing()
    train_dataset = ZipDataset(dataset_path, image_preprocessing.get_train_prep(), use_file_cache)
    train_loader = dataloader.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                         num_workers=num_workers)

    global_step = GlobalStep()

    classes = train_dataset.classes
    class_to_idx = train_dataset.class_to_idx

    checkpoint_name = get_checkpoint_name(train_path_to_init, path_to_save)
    if checkpoint_name is not None:
        model = Model.load_from_checkpoint(checkpoint_name, global_step)
        train_dataset.restore_mapping(model.class_to_idx)
    else:
        model = Model(num_classes=len(train_dataset.classes), base_model_name=base_model,
                      classes=classes, class_to_idx=class_to_idx).cuda()

    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    output  = {'step': 0}
    for e in range(1, num_epoch + 1):
        output = train(train_loader=train_loader, model=model, criterion=criterion, optimizer=optimizer, epoch=e,
                       print_freq=log_frequ, global_step=global_step, save_each_step=save_each_step,
                       path_to_save=path_to_save, min_loss=min_loss, sender=sender)
        #minimal loss leached
        if is_aborted() or (output['step'] > 0 and output['losses'].avg <= min_loss):

            create_and_save_result(model=output['model'], global_step=global_step, loss=output['losses'],
                                   top1=output['top1'], top5=output['top5'],
                                   path_to_save=path_to_save)
            sender.send_train_progress(epoch=output['epoch'], loss_batch=output['losses'].val,
                                       loss_epoch=output['losses'].avg, step=global_step.step, saved=True, batch=output['step'])
            break
    else:
        #lets store latest result
        if output['step'] > 0 and not output['saved']:
            create_and_save_result(model=output['model'], global_step=global_step, loss=output['losses'],
                               top1=output['top1'], top5=output['top5'],
                               path_to_save=path_to_save)
            sender.send_train_progress(epoch=output['epoch'], loss_batch=output['losses'].val,
                                       loss_epoch=output['losses'].avg, step=global_step.step, saved=True, batch=output['step'])
    #clean file cache
    train_dataset.clean()