import os
import re
from torchvision import transforms
import Augmentor
import random
import signal

# TODO remove this fix
LAST_WEIGHT = 'TAKELATEST'


_aborted = False

def abort(signum, frame):
    global _aborted
    _aborted = True

def is_aborted():
    global _aborted
    return _aborted


signal.signal(signal.SIGUSR1, abort)


def find_last_file(path_to_all_weight):
    files = [x for x in os.listdir(path_to_all_weight) if x.endswith('.pth')]
    files = sorted(files, key=lambda x: int(re.findall('\d+', x)[0]))
    return os.path.join(path_to_all_weight, files[-1])


def get_checkpoint_name(checkpoint_name, path_to_all_weight):
    if checkpoint_name is None:
        return None

    if checkpoint_name == LAST_WEIGHT:
        try:
            checkpoint_name = find_last_file(path_to_all_weight)
        except LookupError:
            checkpoint_name = None

    return checkpoint_name


class ImagePreprocessing:

    def __init__(self):
        random.seed()

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        #train processing
        augumentor_pipeline = Augmentor.Pipeline()
        augumentor_pipeline.random_erasing(probability=0.3, rectangle_area=0.15)
        augumentor_pipeline.skew(probability=0.3)
        augumentor_pipeline.crop_random(probability=0.3, percentage_area=0.9)
        augumentor_pipeline.flip_random(probability=0.5)
        augumentor_pipeline.rotate_random_90(probability=0.5)
        augumentor_pipeline.shear(probability=0.2, max_shear_left=7, max_shear_right=7)
        # random_distortion()
        def _fixed_augumentor_transform(image):
            """
            Internal callback for Augumentor transformations handler

            Augumentor Pipeline torch_transform() is broken. It`s fixed callback, @see https://github.com/mdbloice/Augmentor/issues/102
            """

            for operation in augumentor_pipeline.operations:
                r = random.random()
                if r <= operation.probability:
                    image = [image]
                    image = operation.perform_operation(image)
                    image = image[0]
            return image
        self._train_load = transforms.Compose([
            # transforms.Resize((224 * 2, 224 * 2)),
        ])
        self._train_augument = transforms.Compose([
            _fixed_augumentor_transform,
        ])
        self._train_prepare = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])

        #validate processing
        self._validate_load = transforms.Compose([
            transforms.Resize((224, 224)),
        ])
        self._validate_augument = transforms.Compose([
        ])
        self._validate_prepare = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        #inference processing
        self._inference_prep = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])


    def get_train_prep(self):
        return {'load': self._train_load, 'augument': self._train_augument, 'prepare': self._train_prepare}

    def get_val_prep(self):
        return {'load': self._validate_load, 'augument': self._validate_augument, 'prepare': self._validate_prepare}

    def get_inference_prep(self):
        return self._inference_prep