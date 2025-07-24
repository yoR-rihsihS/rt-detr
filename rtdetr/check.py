import torch
from scipy.optimize import linear_sum_assignment

from .box_ops import box_iou, box_cxcywh_to_xyxy

@torch.no_grad()
def checking(results, targets, iou_threshold=0.33):
    """ 
    Computes the number of correct matches between predictions and targets.
    The matching is done using the Hungarian algorithm on the pairwise IoU matrix.
    Args:
        - results: list of dicts, such that len(results) == batch_size
        see the output specification of the RTDETRPostprocessor
            keys:
                - boxes (Tensor): shape (num_pred_objects, 4)
                - scores (Tensor): shape (num_pred_objects, 1)
                - labels (Tensor): shape (num_pred_objects, 1) contains labels of the predicted objects
        - targets: list of dicts, such that len(targets) == batch_size.
            keys:
                - boxes (Tensor): shape (num_objects, 4)
                - labels (Tensor): shape (num_objects)
    Returns:
        - total_targets (int): total number of ground truth objects
        - total_preds (int): total number of predicted objects
        - correct_matches (int): number of correct predictions
        - total_iou (float): sum of IoU scores for bounding boxes of matched objects
    """
    total_targets = 0
    total_preds = 0
    correct_matches = 0
    total_iou = 0

    for res, tgt in zip(results, targets):
        # ground truths
        gt_boxes = tgt["boxes"]            # (num_tgt, 4)
        gt_labels = tgt["labels"]          # (num_tgt)
        num_tgt = gt_boxes.size(0)
        total_targets += num_tgt

        # preds
        pred_boxes = res["boxes"]           # (num_pred, 4)
        pred_labels = res["labels"]         # (num_pred)
        num_pred = pred_boxes.size(0)
        total_preds += num_pred

        if num_tgt == 0 or num_pred == 0:
            continue  # nothing to match

        # Pairwise IoU and Hungarian matching
        ious, unions = box_iou(box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(gt_boxes)) # [num_pred, num_tgt]

        # zero‐out any IoU where labels differ
        label_eq = pred_labels[:, None] == gt_labels[None, :] # [num_pred, num_tgt]
        ious = ious * label_eq

        cost = -ious.cpu().numpy()
        row_idx, col_idx = linear_sum_assignment(cost)

        for r, c in zip(row_idx, col_idx):
            iou = ious[r, c].item()
            if iou < iou_threshold:
                continue
            total_iou += iou
            correct_matches += 1

    return total_targets, total_preds, correct_matches, total_iou