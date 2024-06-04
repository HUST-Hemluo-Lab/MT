#Used to obtain molecular fingerprints

from copy import copy
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils import data
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from rdkit.Chem.Pharm2D import Generate
from torch.utils.data import DataLoader
from copy import deepcopy
from dataset_eos import Seq2seqDataset
from BuildVocab import WordVocab 
from utils import split
import translation_smi4decoder_with_mask

def tokenize_nmt(text):
    """tokenize data"""
    source, target = [], []
    for i in tqdm(range(len(text)),desc='tokenize_nmt进度:'):
        line = text[i].strip('\n').strip(' ')
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

def build_array_nmt(lines, vocab, num_steps):
    """Convert text sequences into small batches"""
    A,masks = [],[]
    for i in range(len(lines)):
        line = lines[i]
        content = [vocab.stoi.get(token, vocab.unk_index) for token in line]
        if len(content) > num_steps -2 : 
            content = content[:254]
        X = content + [vocab.eos_index]
        masks.append([False for _ in range(len(X))] + [True for _ in range(num_steps - len(X))])
        padding = [vocab.pad_index]*(num_steps - len(X))
        X.extend(padding)
        A.append(deepcopy(X))
    return torch.tensor(A), torch.tensor(masks)

def get_dataset(source, src_vocab, seq_len):
    source, src_valid_len = build_array_nmt(source, src_vocab, num_steps=seq_len) 
    dataset = (source, src_valid_len)
    return dataset

def load_array(data_arrays, batch_size, is_train=True): #@save
    """Construct a data iterator"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

#Canoncial smiles 
def SmileToCanon(file,smiles):
    SMILES = []
    if file is not None:
        smiles = pd.read_csv('%s' %(file))['ids']
    
    for smi in smiles:
        SMI = Chem.CanonSmiles(smi)
        SMILES.append(SMI)
    return SMILES

def get_vec6(ids,data,model_weight,hidden_size,n_head,n_layer):

    smi_vocab_path = '../data/pubchem/smi_vocab1.pkl'
    feature_dir = 'data/%s/' %(data,)
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)  
    smi_corpus_path = 'data/%s/smi_corpus_con.txt' %(data,)
    feature_path = 'data/%s/vec_2_translate.csv' %(data,)

    with  open(smi_corpus_path, 'w') as f:
            for i in tqdm(range(len(ids))):
                if ids[i] is None:
                    pass
                else:
                    sm = ids[i]
                    f.write(split(sm) +'\n')
    print('Built smi corpus file!')

    if os.path.exists(smi_corpus_path):
        with open(smi_corpus_path,'r') as f: 
            text = f.readlines()

    src_vocab = WordVocab.load_vocab(smi_vocab_path)
    trfm = translation_smi4decoder_with_mask.TrfmSeq2seq(len(src_vocab), hidden_size, 2260, len(src_vocab),124,842,n_head,n_layer)
    
    trfm.load_state_dict(torch.load(model_weight))  
    source = []
    for sm in text:
        sm = sm.split()
        if len(sm)>256:
            sm = sm[:256]
            source.append(sm)
        else:
            source.append(sm)

    data = get_dataset(source, src_vocab,seq_len=256)
    loader = load_array(data, batch_size=8, is_train=False)

    trfm.eval()

    X_en = []
    with torch.no_grad():
        for b, (sm,sm_valid_len) in enumerate(tqdm(loader)):
            bos = torch.tensor([src_vocab.sos_index] * sm.shape[0]).reshape(-1, 1)
            dec_input_sm = torch.cat([bos, sm[:, :-1]], 1)
            sm = torch.t(dec_input_sm)
            hidden = trfm.encode(sm,src_mask=None,src_key_padding_mask=sm_valid_len) # (T,B,V) 
            X_en.append(hidden)

    vec_encode = pd.DataFrame()

    en = X_en
    for i in tqdm(range(len(en))):
        vec = pd.DataFrame(en[i])
        vec_encode = vec_encode._append(vec,ignore_index=True)
    del en,X_en


    feature = pd.concat([pd.DataFrame(ids),vec_encode],axis=1)
    feature.to_csv(feature_path,index=False)
    return vec_encode

