import json
import zipfile
from collections import defaultdict
import os
from PIL import Image
import shutil
import hashlib


MAIN_JSON_FILE_NAME = 'annotations.json'


def parse_annotations(annotations):
    id_to_ann = defaultdict(list)
    for ann in annotations:
        id_to_ann[ann['image_id']].append(ann)
    return id_to_ann


def find_classes(categories):
    classes = []
    class_to_idx = {}
    mapping = {}
    for idx, cat in enumerate(categories):
        classes.append(cat['name'])
        class_to_idx[cat['name']] = idx
        mapping[cat['id']] = idx
    return classes, class_to_idx, mapping



class ZipDataset:
    def get_zip_for_this_process(self):
        pid = os.getpid()
        if pid not in self.zips:
            self.zips[pid] = zipfile.ZipFile(self.zip_path)
        return self.zips[pid]

    def clean(self):
        if self.use_file_cache and os.path.isdir(self.tmp_dir):
            shutil.rmtree(self.tmp_dir, True)

    def _item_load(self, image):
        """
        Initial load and preparing dataset item

        Load item from zip archive and make initial preparing
        Result can be cached

        :param image: string Path to image in size zip archive
        :return: PIL image(numpy)
        """

        img = None

        if self.use_file_cache:
            item_hash = hashlib.md5(image['file_name'].encode('utf-8')).hexdigest()
            cached_file_name = os.path.join(self.tmp_dir, item_hash + '.rgb')
            if os.path.exists(cached_file_name):
                img = Image.open(cached_file_name)

        if img is None:
            with self.get_zip_for_this_process().open(image['file_name']) as f:
                    with Image.open(f) as img:
                        img = img.convert('RGB')
            img = self.transform['load'](img)
            if self.use_file_cache:
                img.save(cached_file_name)
        return img

    def _item_augument(self, img):
        """
        Augument prepared and loaded image

        :param img: PIL image
        :return: PIL image
        """
        img = self.transform['augument'](img)
        return img

    def _item_prepare(self, img):
        """
        Prepare image for pytorch

        :param img: PIL image
        :return: Tensor
        """
        tensor = self.transform['prepare'](img)
        return tensor

    def __init__(self, zip_path, transform=None, use_file_cache=False):
        self.use_file_cache = use_file_cache
        if self.use_file_cache:
            self.tmp_dir = '/tmp/platform-cache'
            self.clean()
            os.mkdir(self.tmp_dir)

        self.zips = {}
        self.zip_path = zip_path
        
        with self.get_zip_for_this_process().open(MAIN_JSON_FILE_NAME) as myfile:
            coco = json.load(myfile)

        self.coco = coco
        self.images = self.coco['images']
        self.annotations = parse_annotations(self.coco['annotations'])

        classes, class_to_idx, mapping= find_classes(self.coco['categories'])
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.mapping = mapping

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = self.images[item]
        ann = self.annotations[image['id']]

        img = self._item_load(image)
        img = self._item_augument(img)
        # img.save("/tmp/aug_" + str(time.time()) + ".png")

        tensor = self._item_prepare(img)

        category_id = ann[0]['category_id']

        return tensor, self.mapping[category_id]

    def restore_mapping(self, class_to_idx):
        categories = self.coco['categories']
        self.mapping = dict()
        for idx, cat in enumerate(categories):
            stored_idx = class_to_idx[cat['name']]
            self.mapping[cat['id']] = stored_idx
