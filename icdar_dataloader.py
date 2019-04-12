import sys
import os
import torch
import numpy as np
import random
import csv
import skimage.io
import skimage.transform
import skimage.color
import skimage

from torch.utils.data import Dataset, DataLoader
from torchvision import utils
from torch.utils.data.sampler import Sampler
from dateutil.parser import parse as parse_date
from PIL import Image


class IcdarDataset(Dataset):
    @staticmethod
    def resolve_label(content):
        try:
            # Number class
            _ = float(content)
            return 0
        except:
            pass
        try:
            # Date class
            _ = parse_date(content)
            return 1
        except:
            pass
        # Text class
        return 2
        
    def __getitem__(self, idx):
        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.img_paths)

    def __init__(self, data_path, transform):
        self.data_path = data_path
        self.transform = transform
        self.classes = {'Numbers': 0, 'Date': 1, 'Text': 2}

        files = os.listdir(data_path)
        img_files = filter(lambda x: x.endswith('.jpg'), files)
        img_paths = map(lambda x: os.path.join(data_path, x), img_files)
        self.img_paths = list(img_paths)

        self.label_paths = []
        for path in self.img_paths:
            im_id = os.path.split(path)[-1].split('.')[0]
            label_path = os.path.join(data_path, '%s.txt' % im_id)
            self.label_paths.append(label_path)

    def load_image(self, image_index):
        img = skimage.io.imread(self.img_paths[image_index])

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32)/255.0

    def load_annotations(self, image_index):
        label_path = self.label_paths[image_index]
        annotations = []
        with open(label_path, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                parts = line.split(',', maxsplit=8)
                xmin, ymin, _, _, xmax, ymax, _, _, c = parts
                lb = self.resolve_label(c)
                annot = [
                    int(xmin), int(ymin), 
                    int(xmax), int(ymax), lb
                ]
                annotations.append(annot)
        annotations = np.asarray(annotations, dtype=np.float)
        return annotations

    def collate(self, data):
        imgs = [s['img'] for s in data]
        annots = [s['annot'] for s in data]
        scales = [s.get('scale', 1.0) for s in data]
            
        widths = [int(s.shape[0]) for s in imgs]
        heights = [int(s.shape[1]) for s in imgs]
        batch_size = len(imgs)

        max_width = np.array(widths).max()
        max_height = np.array(heights).max()

        padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

        for i in range(batch_size):
            img = imgs[i]
            padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

        max_num_annots = max(annot.shape[0] for annot in annots)
        
        if max_num_annots > 0:
            annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

            if max_num_annots > 0:
                for idx, annot in enumerate(annots):
                    #print(annot.shape)
                    if annot.shape[0] > 0:
                        annot_padded[idx, :annot.shape[0], :] = annot
        else:
            annot_padded = torch.ones((len(annots), 1, 5)) * -1

        padded_imgs = padded_imgs.permute(0, 3, 1, 2)
        return padded_imgs.float(), annot_padded

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def num_classes(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)
