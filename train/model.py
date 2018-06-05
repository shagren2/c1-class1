from typing import List

import torch
import torch.nn as nn
from torchvision import models
from .utils import GlobalStep



BASE_MODELS = {
    'res_net_18': models.resnet18,
    'res_net_34': models.resnet34,
    'res_net_50': models.resnet50,
    'res_net_101': models.resnet101,
    'res_net_152': models.resnet152,

    'densenet121': models.densenet121,
    'densenet169': models.densenet169,
    'densenet201': models.densenet201,
    'densenet161': models.densenet161,

    'squeezenet1_1': models.squeezenet1_1,

    'vgg16': models.vgg16,
    'vgg19': models.vgg19,
}

LAST_FEATURES_SIZE = 1000


def get_base_model(base_model_name):
    assert base_model_name in BASE_MODELS
    base_model = BASE_MODELS[base_model_name](pretrained=True)
    return base_model


class Model(nn.Module):

    @classmethod
    def load_from_checkpoint(cls, path_to_checkpoint: str, global_step: GlobalStep = None):
        checkpoint = torch.load(path_to_checkpoint)

        num_classes = checkpoint['num_classes']
        base_model_name = checkpoint['base_model_name']
        classes = checkpoint['classes']
        class_to_idx = checkpoint['class_to_idx']
        state_dict = checkpoint['state_dict']
        if global_step is not None:
            global_step.restore(checkpoint['step'])
        model = cls(num_classes=num_classes, base_model_name=base_model_name, classes=classes,
                    class_to_idx=class_to_idx)
        model.load_state_dict(state_dict)
        return model

    def __init__(self, num_classes: int, base_model_name: str, classes: List[str], class_to_idx: dict):
        super(Model, self).__init__()
        self.base_model_name = base_model_name
        self.num_classes = num_classes
        self.base_model = get_base_model(base_model_name)

        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(LAST_FEATURES_SIZE, num_classes)
        self.classes = classes
        self.class_to_idx = class_to_idx

    def forward(self, x):
        x = self.base_model(x)
        x = self.relu(x)
        x = self.fc(x)
        return x
