import torch
import torch.nn as nn


from .factory import register_model,create_backbone
from lib.utils import get_loss_fn
from .utils import *
import numpy as np

__all__ = ['RNALoc']


class mLATTICE(nn.Module):
    def __init__(self, backbone, feat_dim, cfg, device=None):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.device = torch.device('cuda') if device is None else device
        self.text_feats = torch.tensor(np.load(cfg.embed_path), dtype=torch.float32).to(self.device)
        self.blocks = nn.ModuleList([Block(dim=feat_dim, num_heads=cfg.num_heads) for _ in range(cfg.depth)])
        if cfg.embed_type == 'bert':
            self.text_feats = torch.tensor(np.load(cfg.embed_path), dtype=torch.float32).to(self.device)
        elif cfg.embed_type == 'onehot':
            self.text_feats = torch.eye(cfg.num_classes).to(self.device)
        else:
            self.text_feats = torch.rand(cfg.num_classes, feat_dim).to(self.device)
        self.depth = cfg.depth
        self.text_head_ = Head(feat_dim, cfg.num_classes)
        #self.text_linear = nn.Linear(cfg.num_classes, feat_dim, bias=False)
        #self.mlp = nn.Linear(768, 9)
        self.attention = LowRankBilinearAttention(feat_dim, feat_dim, feat_dim)

        self.criterion = get_loss_fn(cfg)

        self.net1 = nn.Sequential(
            nn.Conv1d(feat_dim, feat_dim, 1, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True)
        )
        self.net2 = nn.Sequential(
            nn.Conv1d(feat_dim, feat_dim, 1, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, att_mask,token_type_ids,y=None):
        seqfeat, att_token = self.backbone(x,att_mask,token_type_ids) # [8,token,768]

        # cls [8,768]
        tfeat = torch.stack([self.text_feats for _ in range(x.shape[0])], dim=0)
        if self.cfg.embed_type == 'onehot':
            tfeat = self.text_linear(tfeat).to(self.device)
        for i in range(self.depth):
            tfeat, attn = self.blocks[i](seqfeat, tfeat)
            # tfeat [8,9,768]

        _alpha = self.attention(seqfeat, tfeat) # [8,9,token]
        f = self.net1(seqfeat.transpose(1, 2)).transpose(1, 2)
        _x = torch.bmm(_alpha, f)
        _x = self.net2(_x.transpose(1, 2)).transpose(1, 2) # [8,9,768]

        logits = self.text_head_(_x)
        loss = 0
        if self.training:
            loss = self.criterion(logits, y)

        return {
            'logits': logits,
            'attn_label': attn,  # [B,H,9,9]
            'loss': loss,
            'alpha': _alpha, # [B,9,token]
            'att_token':att_token,
        }



@register_model
def RNALoc(cfg):
    backbone, feat_dim = create_backbone(cfg.arch)
    model = mLATTICE(backbone, feat_dim, cfg)
    return model

