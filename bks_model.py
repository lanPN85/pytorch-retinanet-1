import numpy as np

import csv_eval

from model import ResNet, BasicBlock, Bottleneck
from losses import FocalLoss
from anchors import Anchors


SCORE_THRES = 0.05
MAX_DETECTIONS = 100
IOU_THRES=0.5


class UserModel(ResNet):
    def __init__(self):
        super().__init__(3, Bottleneck, [3, 4, 6, 3])
        # Custom anchor
        self.anchors = Anchors(ratios=[0.125, 0.25, 0.5, 1, 2])

    def loss(self, output, target):
        classification, regression, anchors = output
        closs, rloss = FocalLoss()(classification, regression, anchors, target)
        closs = closs.mean()
        rloss = rloss.mean()
        return closs + rloss


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from bks_dataset import UserDataset

    model = UserModel()
    ds = UserDataset('data/icdar-task1-train')
    loader = DataLoader(ds, collate_fn=ds.collate, batch_size=2)
    batch = next(iter(loader))
    data, target = batch
    out = model(data)
    print(out)
    print(model.metrics(out, target))
