from decouple import config

import torch
from torchvision.datasets.folder import default_loader
import torch.nn.functional as F

from platform_classification.train import Model
from platform_classification.common import ImagePreprocessing

TOP_N = config('TOP_N', cast=int)


class InferenceModel:
    def __init__(self, path_to_checkpoint: str):

        self.model = Model.load_from_checkpoint(path_to_checkpoint).cuda()
        self.image_preprocessing_val = ImagePreprocessing().get_inference_prep()
        self.model.eval()

    def classify(self, image_path):
        image = default_loader(image_path)
        image = self.image_preprocessing_val(image)
        image = image.unsqueeze_(0)
        image = torch.autograd.Variable(image.cuda())
        result = self.model(image)

        softmax_out = F.softmax(result).data.cpu().numpy()[0]
        top_index = softmax_out.argsort()[-TOP_N:][::-1]
        result = [{'name': self.model.classes[idx], 'conf': float(softmax_out[idx])} for idx in top_index]
        return result


