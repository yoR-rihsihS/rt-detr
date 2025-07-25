import copy
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from .utils import get_activation

class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=True, act="relu"):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size, stride, padding=(kernel_size-1)//2 if padding is None else padding, bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
    
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=1.0, act="relu"):
        super(Bottleneck, self).__init__()
        hidden_channels = int(out_channels * expansion_factor)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, kernel_size=3, stride=1, act=act)
        self.conv2 = ConvNormLayer(hidden_channels, hidden_channels, kernel_size=1, stride=1, act=act)
        self.conv3 = ConvNormLayer(hidden_channels, out_channels, kernel_size=3, stride=1, act=None)
        self.act = get_activation(act)
         
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = ConvNormLayer(in_channels, out_channels, kernel_size=1, stride=1, act=None)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.act(x + residual)

class FusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=3, expansion_factor=1.0, act="silu"):
        super(FusionBlock, self).__init__()
        self.conv1 = ConvNormLayer(in_channels, out_channels, kernel_size=1, stride=1, act=act)
        self.conv2 = ConvNormLayer(in_channels, out_channels, kernel_size=1, stride=1, act=act)
        self.bottlenecks = nn.Sequential(*[
            Bottleneck(out_channels, out_channels, expansion_factor, act=act) for _ in range(num_blocks)
        ])
            
    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return x_1 + x_2

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation="gelu", normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before
        self.activation = get_activation(activation)

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None):
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)
        if self.norm is not None:
            output = self.norm(output)
        return output

class HybridEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, nhead, dim_feedforward, num_blocks, expansion_factor=1.0, dropout=0.1, num_encoder_layers=1):
        super().__init__()
        assert num_encoder_layers > 0, "Number of encoder layers must be greater than 0"

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_encoder_layers = num_encoder_layers
        
        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1),
                    nn.BatchNorm2d(hidden_dim),
                )
            )

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation="gelu")
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act="silu"))
            self.fpn_blocks.append(
                FusionBlock(hidden_dim * 2, hidden_dim, num_blocks, expansion_factor, act="silu")
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act="silu")
            )
            self.pan_blocks.append(
                FusionBlock(hidden_dim * 2, hidden_dim, num_blocks, expansion_factor, act="silu")
            )

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        
        # encoder
        h, w = proj_feats[-1].shape[2:]
        # flatten [B, C, H, W] to [B, HxW, C]
        src_flatten = proj_feats[-1].flatten(2).permute(0, 2, 1)
        pos_embed = self.build_2d_sincos_position_embedding(w, h, self.hidden_dim).to(src_flatten.device)
        memory = self.encoder(src_flatten, pos_embed=pos_embed)
        proj_feats[-1] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
            inner_outs[0] = feat_high
            upsample_feat = F.interpolate(feat_high, size=feat_low.shape[-2:], mode='nearest')
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_high], dim=1))
            outs.append(out)

        return outs