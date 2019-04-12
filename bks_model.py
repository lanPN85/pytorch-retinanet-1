import numpy as np

import csv_eval

from model import ResNet, BasicBlock, Bottleneck
from losses import FocalLoss


SCORE_THRES = 0.05
MAX_DETECTIONS = 100
IOU_THRES=0.5


class UserModel(ResNet):
    def __init__(self):
        super().__init__(3, Bottleneck, [3, 4, 6, 3])

    def loss(self, output, target):
        classification, regression, anchors = output
        closs, rloss = FocalLoss()(classification, regression, anchors, target)
        closs = closs.mean()
        rloss = rloss.mean()
        return closs + rloss

    def _get_detections(self, outputs):
        classification, regression, anchors = output
        all_detections = [[None for i in range(3)] for j in range(classification.shape[0])]

        for output in outputs:
            scores, labels, boxes = self.infer(output)
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            boxes  = boxes.cpu().numpy()

            boxes /= scale
            indices = np.where(scores > SCORE_THRES)[0]

            if indices.shape[0] > 0:
                # select those scores
                scores = scores[indices]

                # find the order with which to sort the scores
                scores_sort = np.argsort(-scores)[:MAX_DETECTIONS]

                # select detections
                image_boxes      = boxes[indices[scores_sort], :]
                image_scores     = scores[scores_sort]
                image_labels     = labels[indices[scores_sort]]
                image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

                # copy detections to all_detections
                for label in range(3):
                    all_detections[index][label] = image_detections[image_detections[:, -1] == label, :-1]
            else:
                # copy detections to all_detections
                for label in range(3):
                    all_detections[index][label] = np.zeros((0, 5))
        return all_detections        

    def _get_annotations(self, targets):
        batch_size = targets.shape[0]
        all_annotations = [[None for i in range(3)] for j in range(batch_size)]

        for i in range(batch_size):
            # load the annotations
            # copy detections to all_annotations
            target = targets[i]
            for label in range(3):
                all_annotations[i][label] = target[target[:, 4] == label, :4].copy()

        return all_annotations

    def metrics(self, output, target):
        dc = {}

        all_detections = self._get_detections(output)
        all_annotations = self._get_annotations(target)

        for label in range(3):
            false_positives = np.zeros((0,))
            true_positives  = np.zeros((0,))
            scores          = np.zeros((0,))
            num_annotations = 0.0

            for i in range(output.shape[0]):
                detections           = all_detections[i][label]
                annotations          = all_annotations[i][label]
                num_annotations     += annotations.shape[0]
                detected_annotations = []

                for d in detections:
                    scores = np.append(scores, d[4])

                    if annotations.shape[0] == 0:
                        false_positives = np.append(false_positives, 1)
                        true_positives  = np.append(true_positives, 0)
                        continue

                    overlaps            = csv_eval.compute_overlap(np.expand_dims(d, axis=0), annotations)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap         = overlaps[0, assigned_annotation]

                    if max_overlap >= IOU_THRES and assigned_annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives  = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives  = np.append(true_positives, 0)

            # no annotations -> AP for this class is 0 (is this correct?)
            if num_annotations == 0:
                average_precisions[label] = 0, 0
                continue

            # sort by score
            indices         = np.argsort(-scores)
            false_positives = false_positives[indices]
            true_positives  = true_positives[indices]

            # compute false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives  = np.cumsum(true_positives)

            # compute recall and precision
            recall    = true_positives / num_annotations
            precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

            # compute average precision
            average_precision  = csv_eval._compute_ap(recall, precision)
            dc['mAP@%d' % label] = average_precision

        return dc


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
