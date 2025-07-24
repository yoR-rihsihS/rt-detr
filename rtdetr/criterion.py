import torch
import torch.nn as nn
import torchvision.ops as ops
import torch.nn.functional as F 

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class SetCriterion(nn.Module):
    """ This class computes the loss for RTDETR
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, matcher, weight_dict, losses, num_classes, share_matched_indices=False, alpha=0.25, gamma=2.0):
        """ Create the criterion
        Args:
            - num_classes (int): Number of object categories, omitting the special no-object category.
            - matcher (Module): Computes a matching between targets and model outputs
            - weight_dict (dict): Contains as key the names of the losses and as values their relative weight.
            - losses [List]: Losses to be applied. See get_loss for list of available losses.
            - share_matched_indices (bool): Whether to share matched indices between different losses.
            - alpha (float): Focal loss parameter.
            - gamma (float): Focal loss parameter.
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.num_classes = num_classes
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma

    def focal_loss(self, outputs, targets, indices, num_boxes):
        """
        Computes the losses related to the label classification error.
        The target labels are expected to be a tensor of dim [num_target_boxes] containing the class labels.
        """
        assert "pred_logits" in outputs, "Model output does not contain pred_logits"
        idx = self._get_src_permutation_idx(indices)
        src_logits = outputs["pred_logits"]

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1].float()
        loss = ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        # weight_dict is expected to contain the key "loss_label"
        return {f"loss_label": loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Computes the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs, "Model output does not contain pred_boxes"
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]

        target_boxes = torch.cat([t["boxes"][J] for t, (_, J) in zip(targets, indices)], dim=0)

        loss_l1 = F.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))

        # weight_dict is expected to contain the keys "loss_l1" and "loss_giou"
        return {"loss_l1": loss_l1.sum() / num_boxes, "loss_giou": loss_giou.sum() / num_boxes}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        function_map = {
            "boxes": self.loss_boxes,
            "focal" : self.focal_loss,
        }
        assert loss in function_map, f"do you really want to compute {loss} loss?"
        return function_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        """ 
        This performs the loss computation.
        Args:
            - outputs: dict of tensors, see the output specification of the model for the format.
            - targets: list of dicts, such that len(targets) == batch_size.
                The expected keys in each dict depends on the losses applied, see loss doc.
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if "aux" not in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["boxes"]) for t in targets)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f"_aux_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of cdn auxiliary losses. For rtdetr
        if "dn_aux_outputs" in outputs:
            assert "dn_meta" in outputs, "Denoising Metadata not found in output"
            indices = self.get_cdn_matched_indices(outputs["dn_meta"], targets)
            num_boxes = num_boxes * outputs["dn_meta"]["dn_num_group"]

            for i, aux_outputs in enumerate(outputs["dn_aux_outputs"]):
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """
        Computes the matched indices for the denoising auxiliary losses.
        """
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t["boxes"]) for t in targets]
        device = targets[0]["boxes"].device
        
        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), torch.zeros(0, dtype=torch.int64,  device=device)))
        
        return dn_match_indices

