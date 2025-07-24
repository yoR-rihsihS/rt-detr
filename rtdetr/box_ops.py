import torch
from torchvision.ops.boxes import box_area

def box_cxcywh_to_xyxy(x):
    """
    Convert boxes from (x_center, y_center, width, height) to (x_min, y_min, x_max, y_max).
    Args:
        - x (Tensor): shape (..., 4).
    Returns:
        - b (Tensor): shape (..., 4).
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x):
    """
    Convert boxes from (x_min, y_min, x_max, y_max) to (x_center, y_center, width, height).
    Args:
        - x (Tensor): shape (..., 4).
    Returns:
        - b (Tensor): shape (..., 4).
    """
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def box_iou(boxes1, boxes2):
    """
    Compute the pair-wise IoU (and Union) of two sets of boxes in (x_min, y_min, x_max, y_max) format.
    Args:
        - boxes1 (Tensor): shape (N, 4)
        - boxes2 (Tensor): shape (M, 4)
    Returns:
        - iou (Tensor): shape (N, M)
        - union (Tensor): shape (N, M)
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    """
    Compute the pair-wise Generalized IoU of two sets of boxes in (x_min, y_min, x_max, y_max) format.
    Args:
        - boxes1 (Tensor): shape (N, 4)
        - boxes2 (Tensor): shape (M, 4)
    Returns:
        - generalized iou (Tensor): shape (N, M)
    """
    # degenerate boxes results in inf / nan
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area