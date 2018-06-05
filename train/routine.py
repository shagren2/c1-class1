import time
import os

import torch

from .model import Model
from .utils import (AverageMeter, accuracy, GlobalStep)
from .sender import Sender
from platform_classification.common import is_aborted


def create_and_save_result(model: Model, global_step: GlobalStep, loss: AverageMeter, top1: AverageMeter,
                           top5: AverageMeter, path_to_save: str):
    result = {
        'step': global_step.step,
        'top1': top1.avg,
        'top5': top5.avg,
        'loss': loss.avg,
        'state_dict': model.state_dict(),
        'num_classes': model.num_classes,
        'base_model_name': model.base_model_name,
        'classes': model.classes,
        'class_to_idx': model.class_to_idx,
    }
    torch.save(result, os.path.join(path_to_save, f'time_step_{int(time.time())}_global_step_{global_step.step}.pth'))


def train(train_loader, model, criterion, optimizer, epoch: int, print_freq: int, global_step: GlobalStep,
          save_each_step: int, path_to_save: str, min_loss: float, sender: Sender) -> dict:
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()

    result = {
        'model': model,
        'global_step': global_step,
        'losses': losses,
        'top1': top1,
        'top5': top5,
        'step': 0,
        'epoch': epoch,
        'saved': False
    }

    for i, (input, target) in enumerate(train_loader):
        result['saved'] = False
        result['step'] += 1
        # measure data loading time
        data_time.update(time.time() - end)

        input_var = torch.autograd.Variable(input.cuda())
        target_var = torch.autograd.Variable(target.cuda())

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data.cpu(), target, topk=(1, 2))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        global_step.increment()

        if losses.avg <= min_loss:
            break

        if is_aborted():
            break

        if global_step.step % save_each_step == 0:
            create_and_save_result(model=model, global_step=global_step, loss=losses, top1=top1, top5=top5,
                                   path_to_save=path_to_save)
            result['saved'] = True

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
        sender.send_train_progress(epoch=epoch, loss_batch=losses.val,
                                   loss_epoch=losses.avg, step=global_step.step, saved=result['saved'], batch=result['step'])
    return result


def validate(val_loader, model, criterion, print_freq: int, step: int):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    sender = Sender()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input.cuda(), volatile=True)
        target_var = torch.autograd.Variable(target.cuda(), volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data.cpu(), target, topk=(1, 2))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    sender.send_validate_progress(loss_checkpoint=losses.avg, step=step)
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return {'top1': top1.avg, 'top5': top5.avg, 'losses': losses.avg}
