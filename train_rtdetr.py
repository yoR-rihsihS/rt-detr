import os
import json
import argparse
import pickle

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from torch.amp import GradScaler

from rtdetr import RTDETR, train_one_epoch, evaluate, SetCriterion, HungarianMatcher, RTDETRPostProcessor
from rtdetr import COCODataset, collate_fn, get_train_transforms, get_valid_transforms

DEVICE = "cuda:1"

def save_file(history, path):
    with open(path, 'wb') as file:
        pickle.dump(history, file)

def load_file(path):
    with open(path, 'rb') as file:
        history = pickle.load(file)
    return history

def print_metrics(metrics, epoch, mode):
    print(f"Epoch {epoch} - {mode} Metrics:")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"  Total GT Objects: {metrics['total_gt_objects']}")
    print(f"  Total Pred Objetcd: {metrics['total_pred_objects']}")
    print(f"  Total Correct Predictions: {metrics['total_correct_preds']}")

def main(cfg):
    model = RTDETR(
        num_classes = cfg['num_classes'],
        backbone_model = cfg['backbone_model'],
        hidden_dim = cfg['hidden_dim'], 
        nhead = cfg['nhead'], 
        ffn_dim = cfg['ffn_dim'], 
        num_encoder_layers = cfg['num_encoder_layers'],
        expansion_factor= cfg['expansion_factor'],
        aux_loss = cfg['aux_loss'],
        num_queries = cfg['num_queries'],
        num_decoder_points = cfg['num_decoder_points'],
        num_denoising = cfg['num_denoising'],
        num_decoder_layers = cfg['num_decoder_layers'],
        dropout = cfg['dropout'],
        multi_scale= cfg['multi_scale'],
        num_bottleneck_blocks= cfg['num_bottleneck_blocks'],
    )
    model.to(DEVICE)

    matcher = HungarianMatcher(
        weight_dict = cfg['matcher_weight_dict'],
        num_classes = cfg['num_classes'],
    )
    criterion = SetCriterion(
        matcher = matcher,
        weight_dict = cfg['criterion_weight_dict'],
        losses = cfg['compute_losses'],
        num_classes = cfg['num_classes'],
    )

    num_parameters = sum(p.numel() for p in model.parameters())
    print("Number of parameters =", num_parameters)

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters =", num_parameters)

    backbone_params = list(model.backbone.parameters())
    encoder_params = list(model.encoder.parameters())
    decoder_params = list(model.decoder.parameters())

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': cfg['learning_rate_backbone']},
        {'params': encoder_params, 'lr': cfg['learning_rate']},
        {'params': decoder_params, 'lr': cfg['learning_rate']}
    ], weight_decay=cfg['weight_decay'])

    ms_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg["steps"], gamma=cfg['gamma'])
    scaler = GradScaler()

    history = {"train": [], "val": []}

    train_set = COCODataset(image_dir='./data/train2017/', annot_path='./data/annotations/instances_train2017.json', transforms=get_train_transforms())
    print("Total Samples in Train Set:", len(train_set))
    val_set = COCODataset(image_dir='./data/val2017/', annot_path='./data/annotations/instances_val2017.json', transforms=get_valid_transforms())
    print("Total Samples in Validation Set:", len(val_set))

    train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True, num_workers=8, persistent_workers=True, collate_fn=collate_fn, prefetch_factor=10, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg['batch_size'], shuffle=False, num_workers=2, persistent_workers=True, collate_fn=collate_fn, prefetch_factor=10, pin_memory=True)
    
    output_processor = RTDETRPostProcessor(num_classes=cfg['num_classes'], num_queries=cfg['num_queries'])   

    if os.path.exists(f"./saved/{cfg['model_name']}_checkpoint.pth"):
        history = load_file(f"./saved/{cfg['model_name']}_history.pkl")
        checkpoint = torch.load(f"./saved/{cfg['model_name']}_checkpoint.pth", map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        ms_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    for epoch in range(cfg['epochs']):
        if len(history['train']) > epoch:
            print_metrics(history['train'][epoch], epoch+1, "Train")
            print_metrics(history['val'][epoch], epoch+1, "Validation")
            print()
            continue

        train_metrics = train_one_epoch(model, criterion, train_loader, optimizer, scaler, output_processor, DEVICE, max_norm=0.1)
        print_metrics(train_metrics, epoch+1, "Train")
        val_metrics = evaluate(model, criterion, val_loader, output_processor, DEVICE)
        print_metrics(val_metrics, epoch+1, "Validation")

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        print()
        ms_scheduler.step()

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": ms_scheduler.state_dict(),
        }, f"./saved/{cfg['model_name']}_checkpoint.pth")
        save_file(history, f"./saved/{cfg['model_name']}_history.pkl")

        if epoch % 10 == 9:
            torch.save(model.state_dict(), f"./saved/{cfg['model_name']}_{epoch+1}.pth")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="RTDETR Training")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()
    config = json.load(open(args.config, "r"))
    main(config)