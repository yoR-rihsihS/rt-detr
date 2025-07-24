import os
import cv2
import json
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors

class COCODataset(Dataset):
    def __init__(self, image_dir, annot_path, transforms):
        self.image_dir = image_dir
        self.transforms = transforms

        with open(annot_path, 'r') as f:
            coco = json.load(f)

        self.image_id_to_file_name = {img["id"]: img["file_name"] for img in coco["images"]}

        self.image_id_to_annotations = {}
        for annot in coco["annotations"]:
            img_id = annot["image_id"]
            if img_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[img_id] = []
            self.image_id_to_annotations[img_id].append({
                "bbox": annot["bbox"],
                "category_id": annot["category_id"],
            })

        self.ids = list(self.image_id_to_file_name.keys())
        self.cat_id_to_label = {cat["id"]: idx for idx, cat in enumerate(coco["categories"])}
        # There are total 80 categories, apparently COCO dataset's category ids start from 1 to 90, with some gaps.

        del coco # to free memory

    def __len__(self):
        return len(self.ids)
    
    def is_valid_box(self, box):
        """Check if a bounding box is valid."""
        return len(box) == 4 and all(i >=0 for i in box) and box[2] > 0 and box[3] > 0

    def __getitem__(self, index):
        img_id = self.ids[index]
        file_name = self.image_id_to_file_name[img_id]
        file_path = os.path.join(self.image_dir, file_name)

        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        anns = self.image_id_to_annotations.get(img_id, [])

        boxes = []
        labels = []

        for ann in anns:
            if self.is_valid_box(ann["bbox"]):
                boxes.append(ann["bbox"])
                labels.append(self.cat_id_to_label[ann["category_id"]])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        boxes = tv_tensors.BoundingBoxes(boxes, format="XYWH", canvas_size=(image.shape[0], image.shape[1]))
        image, boxes = self.transforms(image, boxes)
        boxes = boxes.as_subclass(torch.Tensor)

        # Get image size after augmentation
        _, H, W = image.shape

        # Convert to [cx, cy, w, h] normalized
        boxes[:, 0] = (boxes[:, 0] + boxes[:, 2] / 2) / W  # cx
        boxes[:, 1] = (boxes[:, 1] + boxes[:, 3] / 2) / H  # cy
        boxes[:, 2] = boxes[:, 2] / W                      # w
        boxes[:, 3] = boxes[:, 3] / H                      # h

        target = {
            'boxes': boxes,
            'labels': torch.tensor(labels, dtype=torch.int64),
        }

        return image, target