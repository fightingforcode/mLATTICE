import os

import numpy as np
import yaml
import argparse
from argparse import Namespace
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
import pandas as pd
warnings.filterwarnings("ignore")
import torch
import torch.nn.functional as F
from models.factory import create_model
from lib.metrics import *
plt.rcParams.update({'font.size': 30})
from Bio import SeqIO
torch.backends.cudnn.benchmark = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(BASE_DIR, 'DNAbert2_attention')


class Pipeline(object):
    def __init__(self, cfg):
        super(Pipeline, self).__init__()
        self.model = create_model(cfg.model, cfg=cfg)
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.labels = [line.strip() for line in open(cfg.label_path)]

    @torch.no_grad()
    def run(self):
        model_dict = torch.load(self.cfg.ckpt_best_path)
        # print(model_dict.keys())
        if list(model_dict.keys())[0].startswith('module'):
            model_dict = {k[7:]: v for k, v in model_dict.items()}
        self.model.load_state_dict(model_dict)
        self.model.cuda()
        print('loading best checkpoint success')
        ids = []
        all_y_score = []
        for record in SeqIO.parse(self.cfg.input_path, "fasta"):
            description = record.description
            ids.append(description[1:])
            seq = str(record.seq)
            if len(seq) > 6000:
                seq = seq[:3000] + seq[-3000:]
            if seq:
                inputs = self.tokenizer(
                    seq,
                    add_special_tokens=True,
                    return_token_type_ids=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                # 推断（预测）
                with torch.no_grad():
                    self.model.eval()
                    outputs = self.model(inputs["input_ids"].cuda(), inputs["attention_mask"].cuda(), inputs["token_type_ids"].cuda())
                    scores = torch.sigmoid(outputs['logits']).cpu().numpy()[0]
                    all_y_score.append(scores)

            all_y_score = np.array(all_y_score)
            all_y_pred = np.where(all_y_score > 0.5, 1, 0)
            output_path = "output_result"

            if not os.path.isdir(output_path):
                os.makedirs(output_path)
            pd.DataFrame(
                all_y_score,
                columns = ['Exosome','Nucleus','Nucleoplasm','Chromatin','Nucleolus','Cytosol','Membrane','Ribosome','Cytoplasm'],
                index=ids,
            ).to_csv(os.path.join(output_path, "score.csv"))
            pd.DataFrame(
                all_y_pred,
                columns = ['Exosome','Nucleus','Nucleoplasm','Chromatin','Nucleolus','Cytosol','Membrane','Ribosome','Cytoplasm'],
                index=ids,
            ).to_csv(os.path.join(output_path, "pred.csv"))
            print('output result success')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-dir', type=str, default='./bestmodel/')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--input-path', type=str, default='./test.fasta')

    args = parser.parse_args()
    cfg_path = os.path.join(args.exp_dir, 'config.yaml')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError('config file not found in the {}!'.format(cfg_path))
    cfg = yaml.load(open(cfg_path, 'r'),Loader=yaml.FullLoader)
    cfg = Namespace(**cfg)
    cfg.ckpt_best_path = os.path.join(args.exp_dir, 'checkpoints','best_model.pth')
    cfg.threshold = args.threshold
    cfg.input_path = args.input_path
    evaluator = Pipeline(cfg)

    evaluator.run()
