import numpy as np
import torch.nn as nn 
import torch.nn.functional as F

from .backbone import BackBone
from .hybrid_encoder import HybridEncoder
from .decoder import RTDETRDecoder

class RTDETR(nn.Module):
    def __init__(self,
                 num_classes, backbone_model, hidden_dim, nhead, ffn_dim, num_encoder_layers, aux_loss, expansion_factor,
                 num_queries, num_decoder_points, num_denoising, num_decoder_layers, num_bottleneck_blocks, dropout=0.1, multi_scale=None):
        super().__init__()
        self.multi_scale = multi_scale
        self.num_classes = num_classes
        self.backbone = BackBone(model=backbone_model, num_levels=3)
        self.in_channels = [128, 256, 512] if backbone_model in ["resnet18", "resnet34"] else [512, 1024, 2048]
        self.feat_strides = [8, 16, 32]

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.dim_feedforward = ffn_dim
        self.dropout = dropout
        self.num_encoder_layers = num_encoder_layers
        self.expansion_factor = expansion_factor
        self.num_bottleneck_blocks = num_bottleneck_blocks
        
        self.encoder = HybridEncoder(
            in_channels = self.in_channels,
            hidden_dim = self.hidden_dim,
            nhead = self.nhead,
            dim_feedforward = self.dim_feedforward,
            dropout = self.dropout,
            num_encoder_layers = self.num_encoder_layers,
            expansion_factor = self.expansion_factor,
            num_blocks = self.num_bottleneck_blocks
        )

        self.aux_loss = aux_loss
        self.num_queries = num_queries
        self.num_decoder_points = num_decoder_points
        self.num_denoising = num_denoising
        self.num_decoder_layers = num_decoder_layers

        self.decoder = RTDETRDecoder(
            num_classes = self.num_classes,
            d_model = self.hidden_dim,
            num_queries = self.num_queries,
            feat_channels = len(self.in_channels) * [hidden_dim],
            num_levels = len(self.in_channels),
            num_points = self.num_decoder_points,
            nhead = self.nhead,
            num_layers = self.num_decoder_layers,
            dim_feedforward = self.dim_feedforward,
            dropout = self.dropout,
            num_denoising = self.num_denoising,
            label_noise_ratio = 0.5,
            box_noise_scale = 1.0,
            aux_loss = self.aux_loss,
        )
        
    def forward(self, x, targets=None):
        if self.multi_scale is not None and self.training:
            size = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[size, size], mode='bilinear', align_corners=False)

        x = self.backbone(x)
        x = self.encoder(x)     
        x = self.decoder(x, targets)
        return x