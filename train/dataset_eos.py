'''
The main part is quoted from the following source:

@article{honda2019smiles,
    title={SMILES Transformer: Pre-trained Molecular Fingerprint for Low Data Drug Discovery},
    author={Shion Honda and Shoi Shi and Hiroki R. Ueda},
    year={2019},
    eprint={1911.04738},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
'''

import random
from re import A
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from enumerator import SmilesEnumerator
from utils import split

PAD = 0
MAX_LEN = 512

class Randomizer(object):

    def __init__(self):
        self.sme = SmilesEnumerator()
    
    def __call__(self, sm):
        sm_r = self.sme.randomize_smiles(sm) # Random transoform
        if sm_r is None:
            sm_spaced = split(sm) # Spacing
        else:
            sm_spaced = split(sm_r) # Spacing
        sm_split = sm_spaced.split()
        if len(sm_split)<=MAX_LEN - 2:
            return sm_split # List
        else:
            return split(sm).split()


def truncate_pad(line, num_steps, padding_token):
        """Truncate or pad text sequences"""
        if len(line) > num_steps:
            return line[:num_steps] # 截断
        return line + [padding_token] * (num_steps - len(line)) # 填充

class Seq2seqDataset(Dataset):

    def __init__(self, smiles, vocab, seq_len=512, transform=Randomizer()):
        self.smiles = smiles
        self.vocab = vocab
        self.seq_len = seq_len
        self.transform = transform

    def __len__(self):
        return len(self.smiles)



    def __getitem__(self, item):
        sm = self.smiles[item]
        content = [self.vocab.stoi.get(token, self.vocab.unk_index) for token in sm]

        array = torch.tensor([truncate_pad(l, self.seq_len, self.vocab.pad_index) for l in content])
        valid_len = (array !=  self.pad_index).type(torch.int32).sum(1)

        return array, valid_len


class Seq2seqDataset_len(Dataset):

    def __init__(self, smiles, vocab, seq_len=256, transform=Randomizer()):
        self.smiles = smiles
        self.vocab = vocab
        self.seq_len = seq_len
        self.transform = transform

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, item):
        sm = self.smiles[item]
        #sm = self.transform(sm) # List
        content = [self.vocab.stoi.get(token, self.vocab.unk_index) for token in sm]
        if len(content) > self.seq_len -2 : 
            content = content[:254]
        X = content + [self.vocab.eos_index]

        padding = [self.vocab.pad_index]*(self.seq_len - len(X))
        X.extend(padding)
        valid_len = (torch.tensor(X) != self.vocab.pad_index).type(torch.int32).sum(1)
        return torch.tensor(X), valid_len

