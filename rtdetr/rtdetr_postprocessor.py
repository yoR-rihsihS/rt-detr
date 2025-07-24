import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class RTDETRPostProcessor(nn.Module):
    def __init__(self, num_classes, num_queries):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
    
    @torch.no_grad()
    def forward(self, outputs, top_k=None, score_thresh=0.33):
        if top_k is None:
            top_k = self.num_queries

        if top_k > self.num_queries:
            top_k = self.num_queries
            print(f"top_k : {top_k} is larger than num_queries, model can only return upto {self.num_queries} queries.")

        boxes = outputs["pred_boxes"]   # shape: (bs, numq, 4)
        logits = outputs["pred_logits"] # shape: (bs, numq, num_classes)

        scores = F.sigmoid(logits).max(-1).values   # shape: (bs, numq)
        topk_scores, topk_indices = scores.topk(top_k, dim=1)   # shape: (bs, top_k)
        topk_bboxes = boxes.gather(dim=1, index=topk_indices.unsqueeze(-1).expand(-1, -1, boxes.shape[-1])) # shape: (bs, top_k, 4)

        labels = logits.argmax(dim=-1)  # (bs, numq)
        topk_labels = labels.gather(1, topk_indices) # (bs, top_k)

        bs = boxes.shape[0]
        results = []
        for b in range(bs):
            valid_mask = topk_scores[b].squeeze(-1) >= score_thresh  # (k,)
            results.append({
                "boxes": topk_bboxes[b][valid_mask], 
                "scores": topk_scores[b][valid_mask], 
                "labels": topk_labels[b][valid_mask],
            })

        return results