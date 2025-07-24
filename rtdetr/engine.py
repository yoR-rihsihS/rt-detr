import torch
from torch.amp import autocast

from .check import checking

def train_one_epoch(model, criterion, data_loader, optimizer, scaler, postprocessor, device, max_norm=0):
    count = 0
    running_loss = 0
    total_gt_objects = 0
    total_pred_objects = 0
    total_correct_preds = 0
    total_iou = 0
    model.train()
    criterion.train()
    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        with autocast(device_type=device, cache_enabled=True):
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)
        
        loss = sum(loss_dict.values())
        
        scaler.scale(loss).backward()
        if max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        count += 1
        if count % 100 == 0:
            print(f"{count} iterations done", end='\r')

        num_gt_objects, num_pred_objects, num_correct_preds, iou = checking(postprocessor(outputs), targets)
        total_gt_objects += num_gt_objects
        total_pred_objects += num_pred_objects
        total_correct_preds += num_correct_preds
        total_iou += iou

    precision = total_correct_preds / total_pred_objects if total_pred_objects > 0 else 0.0
    recall = total_correct_preds / total_gt_objects if total_gt_objects > 0 else 0.0
    mean_iou = total_iou / total_correct_preds if total_correct_preds > 0 else 0.0

    metrics = {
        "loss": running_loss / len(data_loader),
        "precision": precision,
        "recall": recall,
        "mean_iou": mean_iou,
        "total_gt_objects": total_gt_objects,
        "total_pred_objects": total_pred_objects,
        "total_correct_preds": total_correct_preds,
    }
     
    return metrics


@torch.no_grad()
def evaluate(model, criterion, data_loader, postprocessor, device):
    running_loss = 0
    total_gt_objects = 0
    total_pred_objects = 0
    total_correct_preds = 0
    total_iou = 0
    model.eval()
    criterion.eval()
    with torch.no_grad():
        for samples, targets in data_loader:
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with autocast(device_type=device, cache_enabled=True):
                outputs = model(samples)
                loss_dict = criterion(outputs, targets)
            
            loss = sum(loss_dict.values())

            running_loss += loss.item()

            num_gt_objects, num_pred_objects, num_correct_preds, iou = checking(postprocessor(outputs), targets)
            total_gt_objects += num_gt_objects
            total_pred_objects += num_pred_objects
            total_correct_preds += num_correct_preds
            total_iou += iou

    precision = total_correct_preds / total_pred_objects if total_pred_objects > 0 else 0.0
    recall = total_correct_preds / total_gt_objects if total_gt_objects > 0 else 0.0
    mean_iou = total_iou / total_correct_preds if total_correct_preds > 0 else 0.0

    metrics = {
        "loss": running_loss / len(data_loader),
        "precision": precision,
        "recall": recall,
        "mean_iou": mean_iou,
        "total_gt_objects": total_gt_objects,
        "total_pred_objects": total_pred_objects,
        "total_correct_preds": total_correct_preds,
    }
     
    return metrics