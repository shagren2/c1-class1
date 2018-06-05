import os
import shutil
import torch.nn as nn
from torch.utils.data import dataloader
import re

from platform_classification.common import ImagePreprocessing
from platform_classification.dataset import ZipDataset

from .model import Model
from .routine import validate
from .utils import GlobalStep



def start_validate(dataset_path: str, path_to_checkpoint_dir: str, val_path_to_save_best: str, batch_size: int,
                   num_workers: int, print_freq: int, use_file_cache: bool = False):
    image_preprocessing = ImagePreprocessing()
    val_dataset = ZipDataset(dataset_path, image_preprocessing.get_val_prep(), use_file_cache)
    val_loader = dataloader.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    criterion = nn.CrossEntropyLoss().cuda()

    global_step = GlobalStep()

    model_checkpoints = [x for x in os.listdir(path_to_checkpoint_dir) if x.endswith('.pth')]
    model_checkpoints = sorted(model_checkpoints, key=lambda x: int(re.findall(r'\d+(?=\.pth)', x)[0]))
    best_loss = float('inf')
    for model_path in model_checkpoints:
        step = 0
        steps = re.findall(r'\d+(?=\.pth)', model_path)
        if steps:
            step = int(steps[0])
        model = Model.load_from_checkpoint(os.path.join(path_to_checkpoint_dir, model_path), global_step)
        model.cuda()
        val_dataset.restore_mapping(model.class_to_idx)
        results = validate(val_loader=val_loader, model=model, criterion=criterion, print_freq=print_freq,
                           step=step)
        loss = results['losses']
        if loss < best_loss:
            shutil.copyfile(os.path.join(path_to_checkpoint_dir, model_path), val_path_to_save_best)
            best_loss = loss
    val_dataset.clean()
