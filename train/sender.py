import json


class Sender:
    def __init__(self, prefix='__nmprogress'):
        self.prefix = prefix

    def send_loss(self, loss):
        """
        @legacy
        :param loss:
        :return:
        """
        self.send('loss={}'.format(loss))

    def send(self, message):
        """
        Send any string to platform
        :param str:
        :return:
        """
        print('[{}]{}'.format(self.prefix, message))

    def send_object(self, obj):
        """
        Send any object to platform
        :param obj:
        :return:
        """
        message = json.dumps(obj)
        self.send(message)

    def send_train_progress(self, epoch: int, loss_batch: float, loss_epoch: float, step: int, saved: bool, batch: int):
        """
        Send train progress
        :param epoch:
        :param loss_batch:
        :param loss_epoch:
        :return:
        """
        report = {'epoch': epoch, 'loss_batch': loss_batch, 'loss_epoch': loss_epoch, 'step': step, 'saved': saved, 'batch': batch}
        self.send_object(report)

    def send_validate_progress(self, loss_checkpoint: float, step: int):
        """
        Send validate progress
        :param epoch:
        :param loss_batch:
        :param loss_epoch:
        :return:
        """
        report = {'loss_checkpoint': loss_checkpoint, 'step': step}
        self.send_object(report)
