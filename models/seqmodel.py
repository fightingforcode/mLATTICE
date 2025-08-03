#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel
import logging

logger = logging.getLogger(__name__)
from .factory import register_backbone

__all__ = ['BERTClass']

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(BASE_DIR, 'DNAbert2_attention')

class BERTClass(nn.Module):
    def __init__(self, dp1, dp2):
        super(BERTClass, self).__init__()
        self.bert_model = AutoModel.from_pretrained(path,trust_remote_code=True, output_attentions=True)
        self.sequential = torch.nn.Sequential(torch.nn.Dropout(dp1),
                                              torch.nn.Linear(768, 1024),
                                              torch.nn.Dropout(dp2),
                                              torch.nn.Linear(1024, 768)
                                              )

    def forward(self, input_ids, attn_mask, token_type_ids):
        out,cls,attn_weight,attention_prob = self.bert_model(input_ids, attention_mask=attn_mask, token_type_ids=token_type_ids)

        hidden_states = torch.stack(list(out), dim=0)

        hidden_states = self.sequential(hidden_states)

        return hidden_states, attention_prob[-1] # last layer attention [8, 12, 1059, 1059]


@register_backbone(feat_dim=768)
def seq_encoder(**kwargs):
    return BERTClass(dp1=0.1, dp2=0.1)




