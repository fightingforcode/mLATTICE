import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


logger = logging.getLogger(__name__)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(BASE_DIR, 'DNAbert2_attention')

class MLDataset(Dataset):
    def __init__(self, cfg,data_path,is_train = True):
        super(MLDataset, self).__init__()
        self.is_train = is_train
        self.tokenizer = AutoTokenizer.from_pretrained(path,trust_remote_code=True)
        self.labels = [line.strip() for line in open(cfg.label_path)]
        self.num_classes = len(self.labels)
        self.label2id = {label: i for i, label in enumerate(self.labels)}

        self.data = []
        self.seq = []
        with open(data_path, 'r') as fr:
            for line in fr.readlines():
                seqs, label = line.strip().split('\t')
                seqs = self.truncate_sequence(seqs, 5000)
                label = [self.label2id[l] for l in label.split(',')]
                self.data.append(seqs)
                self.seq.append(label)
        self.max_len = self.getmaxtokenizerlen(self.data)
    def getmaxtokenizerlen(self,X):
        maxlen = 0
        for each in X:
            templen = self.tokenizer(each, return_tensors='pt')["input_ids"].shape[1]
            if templen > maxlen:
                maxlen = templen
            # print(templen)
        return maxlen
    def truncate_sequence(self,sequence, max_length):
        if len(sequence) <= max_length:
            return sequence
        else:
            return sequence[-max_length:]

    def __getitem__(self, index):

        seq, label = self.data[index],self.seq[index]

        # one-hot encoding for label
        target = np.zeros(self.num_classes).astype(np.float32)
        target[label] = 1.0

        inputs = self.tokenizer(  # .encode_plus
            seq,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        motif = -1
        if not self.is_train:
            motif = self.tokenizer.decode(inputs['input_ids'].flatten())
        return {
            'features': inputs["input_ids"].flatten().cuda(),
            'attention_mask': inputs['attention_mask'].flatten().cuda(),
            'token_type_ids': inputs["token_type_ids"].flatten().cuda(),
            'labels': torch.FloatTensor(target),
            'motif': motif if motif else -1
        }

    def __len__(self):
        return len(self.data)


