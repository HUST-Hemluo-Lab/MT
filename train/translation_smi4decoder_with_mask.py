'''
partial of the content is quoted from the following source but with major changes:

@article{honda2019smiles,
    title={SMILES Transformer: Pre-trained Molecular Fingerprint for Low Data Drug Discovery},
    author={Shion Honda and Shoi Shi and Hiroki R. Ueda},
    year={2019},
    eprint={1911.04738},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
'''
import math
import gc 
import os
import random
from typing_extensions import Self
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import copy
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm
from IPython import display
from torch.autograd import Variable
from BuildVocab import WordVocab
from dataset_eos import Seq2seqDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


PAD = 0
UNK = 1
EOS = 2
SOS = 3
MASK = 4


class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建⼀个⾜够⻓的 `P`
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)
    def forward(self, X):
        X = X.permute(1,0,2)
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X.permute(1,0,2))


class TrfmSeq2seq(nn.Module):
    def __init__(self, in_size, hidden_size, out_size1, out_size2,out_size3, out_size4,nheads,n_layers, dropout=0.1):
        super(TrfmSeq2seq, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(in_size, hidden_size)
        self.embed_tgt = nn.Embedding(out_size1,hidden_size)
        self.embed_inchi = nn.Embedding(out_size3,hidden_size)
        self.embed_pubchemfp = nn.Embedding(out_size4,hidden_size)            ##增加decoder
        self.pe = PositionalEncoding(hidden_size, dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nheads, dim_feedforward=hidden_size, dropout=dropout, activation = "relu")
        encoder_norm = nn.LayerNorm(hidden_size)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, norm = encoder_norm)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=nheads, dim_feedforward=hidden_size, dropout=dropout, activation = "relu")
        decoder_norm = nn.LayerNorm(hidden_size)
        self.decoder1 = nn.TransformerDecoder(decoder_layer, num_layers=n_layers, norm=decoder_norm)
        self.decoder2 = nn.TransformerDecoder(decoder_layer, num_layers=n_layers, norm=decoder_norm)
        self.decoder3 = nn.TransformerDecoder(decoder_layer, num_layers=n_layers, norm=decoder_norm)
        self.decoder4 = nn.TransformerDecoder(decoder_layer, num_layers=n_layers, norm=decoder_norm)  ##增加decoder

        self.out1 = nn.Linear(hidden_size, out_size1)
        self.out2 = nn.Linear(hidden_size, out_size2)
        self.out3 = nn.Linear(hidden_size, out_size3)
        self.out4 = nn.Linear(hidden_size, out_size4)                ##增加decoder

    def forward(self, src, tgt, src_de, inchi, pubchemfp,
                src_mask=None,tgt_mask_1=None, tgt_mask_2=None, tgt_mask_3=None, tgt_mask_4=None, memory_mask=None, 
                src_key_padding_mask=None, tgt_key_padding_mask=None, inchi_key_padding_mask=None, pubchemfp_key_padding_mask=None, memory_key_padding_mask=None):
        # src: (T,B)
        embedded_src = self.embed(src)  # (T,B,H)
        embedded_src = self.pe(embedded_src) # (T,B,H) 位置编码
        embedded_tgt = self.embed_tgt(tgt)  # (T,B,H)
        embedded_tgt = self.pe(embedded_tgt)
        embedded_src_de = self.embed(src_de)  # (T,B,H)
        embedded_src_de = self.pe(embedded_src_de) # (T,B,H) 位置编码
        embedded_inchi = self.embed_inchi(inchi)  # (T,B,H)
        embedded_inchi = self.pe(embedded_inchi) # (T,B,H) 位置编码
        embedded_pubchemfp = self.embed_pubchemfp(pubchemfp)  # (T,B,H)
        embedded_pubchemfp  = self.pe(embedded_pubchemfp ) # (T,B,H) 位置编码      

        encode_out = self.encoder(embedded_src,mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        hidden_pharm = self.decoder1(embedded_tgt,encode_out,tgt_mask = tgt_mask_1,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask) # (T,B,H)  
        hidden_smi= self.decoder2(embedded_src_de,encode_out,tgt_mask = tgt_mask_2,
                                    tgt_key_padding_mask=src_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask)
        hidden_inchi= self.decoder3(embedded_inchi,encode_out,tgt_mask = tgt_mask_3,
                                    tgt_key_padding_mask=inchi_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask)
        hidden_pubchemfp= self.decoder4(embedded_pubchemfp,encode_out,tgt_mask = tgt_mask_4,
                                    tgt_key_padding_mask=pubchemfp_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask)                 ##增加decoder

        out_pharm = self.out1(hidden_pharm) # (T,B,V)  #线性层
        out_pharm = F.log_softmax(out_pharm, dim=2) # (T,B,V)  v= vocab_size

        out_smi = self.out2(hidden_smi) # (T,B,V)  #线性层
        out_smi = F.log_softmax(out_smi, dim=2) # (T,B,V)  v= vocab_size

        out_inchi = self.out3(hidden_inchi) # (T,B,V)  #线性层
        out_inchi = F.log_softmax(out_inchi, dim=2) # (T,B,V)  v= vocab_size

        out_pubchemfp = self.out4(hidden_pubchemfp) # (T,B,V)  #线性层
        out_pubchemfp = F.log_softmax(out_pubchemfp, dim=2) # (T,B,V)  v= vocab_size 

        return out_pharm,out_smi,out_inchi,out_pubchemfp 



    def _encode1(self, src, src_mask, src_key_padding_mask):
        # src: (T,B)
        embedded = self.embed(src)  # (T,B,H)
        embedded = self.pe(embedded) # (T,B,H)
        output = embedded
        for i in range(self.encoder.num_layers - 1):
            output = self.encoder.layers[i](output,src_mask=src_mask,
                                                src_key_padding_mask = src_key_padding_mask)  # (T,B,H)
        penul = output.detach().numpy()    #取倒数第二层
        output = self.encoder.layers[-1](output,src_mask=src_mask,
                                                src_key_padding_mask = src_key_padding_mask)  # (T,B,H)
        if self.encoder.norm:
            output = self.encoder.norm(output) # (T,B,H)
        output = output.detach().numpy()  #取最后一层
        return np.hstack([np.mean(output, axis=0), np.max(output, axis=0), np.mean(penul, axis=0), np.max(penul, axis=0)]) # (B,4H)


    
    def encode(self, src, src_mask, src_key_padding_mask):
        # src: (T,B)
        batch_size = src.shape[1]
        if batch_size<=100:
            return self._encode1(src, src_mask, src_key_padding_mask)
        else: # Batch is too large to load
            print('There are {:d} molecules. It will take a little time.'.format(batch_size))
            st,ed = 0,100
            out = self._encode1(src[:,st:ed]) # (B,4H)
            while ed<batch_size:
                st += 100
                ed += 100
                out = np.concatenate([out, self._encode1(src[:,st:ed])], axis=0)
            return out


def evaluate(model, test_loader, tgt_vocab, src_vocab, inchi_vocab, pubchemfp_vocab):
    model.eval()
    total_loss = 0
    for b,(sm, X_valid_len, fp, Y_valid_len, inchi, I_valid_len, pubchemfp, P_valid_len) in enumerate(tqdm(test_loader)): ##增加decoder
        bos = torch.tensor([tgt_vocab.sos_index] * fp.shape[0]).reshape(-1, 1)
        dec_input_fp = torch.cat([bos, fp[:, :-1]], 1) #2D pharmacophore fingerprint
        dec_input_sm = torch.cat([bos, sm[:, :-1]], 1)
        dec_input_inchi = torch.cat([bos,inchi[:, :-1]], 1)
        dec_input_pubchemfp = torch.cat([bos,pubchemfp[:, :-1]], 1) 
 
        sm = torch.t(sm.cuda()) # (T,B)
        fp = torch.t(fp.cuda())
        inchi = torch.t(inchi.cuda())
        pubchemfp = torch.t(pubchemfp.cuda())

        dec_input_fp = torch.t(dec_input_fp.cuda()) # (T,B)   
        dec_input_sm =torch.t(dec_input_sm.cuda())
        dec_input_inchi =torch.t(dec_input_inchi.cuda())
        dec_input_pubchemfp =torch.t(dec_input_pubchemfp.cuda())
     

        tgt_mask_1 = generate_square_subsequent_mask(dec_input_fp.shape[0],device)
        tgt_mask_2 = generate_square_subsequent_mask(dec_input_sm.shape[0],device)
        tgt_mask_3 = generate_square_subsequent_mask(dec_input_inchi.shape[0],device)
        tgt_mask_4 = generate_square_subsequent_mask(dec_input_pubchemfp.shape[0],device)
     


        with torch.no_grad():
            output_pharm,output_smi,output_inchi,output_pubchemfp = model(sm,dec_input_fp,dec_input_sm,dec_input_inchi,dec_input_pubchemfp,         
                                            src_mask=None, 
                                            tgt_mask_1=tgt_mask_1,
                                            tgt_mask_2=tgt_mask_2,
                                            tgt_mask_3=tgt_mask_3, 
                                            tgt_mask_4=tgt_mask_4,                                         
                                            src_key_padding_mask = X_valid_len.cuda(), 
                                            tgt_key_padding_mask = Y_valid_len.cuda(),
                                            inchi_key_padding_mask = I_valid_len.cuda(),   
                                            pubchemfp_key_padding_mask = P_valid_len.cuda(),                   
                                            memory_key_padding_mask = X_valid_len.cuda()) 

        loss_pharm = F.nll_loss(output_pharm.view(-1, len(tgt_vocab)),
                               fp.contiguous().view(-1),
                               ignore_index=PAD) 
        loss_smi = F.nll_loss(output_smi.view(-1, len(src_vocab)),
                               sm.contiguous().view(-1),
                               ignore_index=PAD)
        loss_inchi = F.nll_loss(output_inchi.view(-1, len(inchi_vocab)),
                               inchi.contiguous().view(-1),
                               ignore_index=PAD)
        loss_pubchemfp = F.nll_loss(output_pubchemfp.view(-1, len(pubchemfp_vocab)),
                               pubchemfp.contiguous().view(-1),
                               ignore_index=PAD)
        loss = 30*loss_pharm + 10*loss_smi + 30*loss_inchi + 20*loss_pubchemfp
        total_loss += loss.item()
    return total_loss / len(test_loader)

def tokenize_nmt(text):
    """tokenize data"""
    source, target, inchi, pubchemfp = [], [], [], []
    for i in tqdm(range(len(text)),desc='tokenize_nmt进度:'):
        line = text[i].strip('\n').strip(' ')
        parts = line.split('\t')
        if len(parts) == 4:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
            inchi.append(parts[2].split(' '))
            pubchemfp.append(parts[3].split(' '))
    return source, target, inchi, pubchemfp

def generate_square_subsequent_mask(sz, device):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

def build_array_nmt(lines, vocab, num_steps):
    """Convert text sequences into small batches."""
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
        A.append(copy.deepcopy(X))
    return torch.tensor(A), torch.tensor(masks)

def load_array(data_arrays, batch_size, is_train=True): 
    """Construct a data iterator"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def truncate_pad(line, num_steps, padding_token):
    """Truncate or pad text sequences."""
    if len(line) > num_steps:
        return line[:num_steps] # 截断
    return line + [padding_token] * (num_steps - len(line)) # 填充

def data_split(full_list, ratio, shuffle=False):
    """
    数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio:     子列表1
    :param shuffle:   子列表2
    :return:
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2

def get_dataset(text, src_vocab, tgt_vocab, inchi_vocab, pubchemfp_vocab, seq_len):
    source, target, inchi_target,pubchemfp_target= tokenize_nmt(text)
    target, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps=seq_len) 
    source, src_valid_len = build_array_nmt(source, src_vocab, num_steps=seq_len) 
    inchi_target, inchi_valid_len = build_array_nmt(inchi_target, inchi_vocab, num_steps=seq_len) 
    pubchemfp_target, pubchemfp_valid_len = build_array_nmt(pubchemfp_target, pubchemfp_vocab, num_steps=seq_len)
    dataset = (source, src_valid_len, target, tgt_valid_len, inchi_target, inchi_valid_len, pubchemfp_target, pubchemfp_valid_len)
    return dataset

def train_smi(smi_vocab_path,pharm_vocab_path,inchi_vocab_path,pubchemfp_vocab_path,
    smi_pharm_inchi_pubchemfp_path,size,
    seq_len=256,n_epoch=100,num_workers=0,batch_size=32,
    n_hidden=256,n_heads=4,n_layers=2,lr=1e-4):
    print('Loading dataset...')

    src_vocab = WordVocab.load_vocab(smi_vocab_path) 
    tgt_vocab = WordVocab.load_vocab(pharm_vocab_path)  
    inchi_vocab = WordVocab.load_vocab(inchi_vocab_path)
    pubchemfp_vocab = WordVocab.load_vocab(pubchemfp_vocab_path)

    #Model design
    model = TrfmSeq2seq(len(src_vocab),n_hidden, len(tgt_vocab), len(src_vocab), len(inchi_vocab), len(pubchemfp_vocab), n_heads,n_layers).cuda()  #sm 2 pharm
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(model)
    print('Total parameters:', sum(p.numel() for p in model.parameters()))

    log_dir_ = 'logs_model/logs_train'
    log_dir = 'logs_model/trfm_model.pth' 
    if not os.path.isdir(log_dir_):
                    os.makedirs(log_dir_)
    if os.path.exists(log_dir):
        checkpoint = torch.load(log_dir)
        model.load_state_dict(checkpoint['model'])
        
        start_epoch = checkpoint['epoch']
        print('加载 epoch {} 成功！'.format(start_epoch))
    else:
        start_epoch = 0
        print('无保存模型，将从头开始训练！')

   #Read file
    f = open(smi_pharm_inchi_pubchemfp_path ,"r")   
    text = f.readlines()    
    f.close()  
    text_sub= random.sample(text,size)
    train_text, test_text = data_split(text_sub, 0.8, shuffle=True)
    print('Train size:',len(train_text))
    train = get_dataset(train_text, src_vocab, tgt_vocab, inchi_vocab, pubchemfp_vocab, seq_len)
    test = get_dataset(test_text, src_vocab, tgt_vocab, inchi_vocab, pubchemfp_vocab, seq_len)
    del text
    gc.collect()  #Release resources

    print('Dataloader...')
    train_loader = load_array(train, batch_size, is_train=True)
    test_loader = load_array(test, batch_size, is_train=False)
    del  train, test
    gc.collect()    #Release resources

  
    #Calculated loss   
    best_loss = None
    for e in range(start_epoch,n_epoch):
        for b,(sm, X_valid_len, fp, Y_valid_len, inchi, I_valid_len,pubchemfp, P_valid_len) in enumerate(tqdm(train_loader)):
            bos = torch.tensor([tgt_vocab.sos_index] * fp.shape[0]).reshape(-1, 1)
            dec_input_fp = torch.cat([bos, fp[:, :-1]], 1)
            dec_input_sm = torch.cat([bos, sm[:, :-1]], 1)
            dec_input_inchi = torch.cat([bos, inchi[:, :-1]], 1)
            dec_input_pubchemfp = torch.cat([bos, pubchemfp[:, :-1]], 1)
 
            sm = torch.t(sm.cuda()) # (T,B)
            fp = torch.t(fp.cuda())
            inchi = torch.t(inchi.cuda())
            pubchemfp = torch.t(pubchemfp.cuda())
            dec_input_fp = torch.t(dec_input_fp.cuda()) # (T,B)   
            dec_input_sm =torch.t(dec_input_sm.cuda())
            dec_input_inchi =torch.t(dec_input_inchi.cuda())
            dec_input_pubchemfp =torch.t(dec_input_pubchemfp.cuda())

            # Mask
            tgt_mask_1 = generate_square_subsequent_mask(dec_input_fp.shape[0],device)
            tgt_mask_2 = generate_square_subsequent_mask(dec_input_sm.shape[0],device)
            tgt_mask_3 = generate_square_subsequent_mask(dec_input_inchi.shape[0],device)
            tgt_mask_4 = generate_square_subsequent_mask(dec_input_pubchemfp.shape[0],device)

            optimizer.zero_grad()
            output_pharm,output_smi,output_inchi,output_pubchemfp = model(sm,dec_input_fp,dec_input_sm,dec_input_inchi,dec_input_pubchemfp,
                                            src_mask=None,  # Encoder的注意力Mask输入，这部分其实对于Encoder来说是没有用的
                                            tgt_mask_1=tgt_mask_1,
                                            tgt_mask_2=tgt_mask_2,
                                            tgt_mask_3=tgt_mask_3,
                                            tgt_mask_4=tgt_mask_4,
                                            src_key_padding_mask = X_valid_len.cuda(), 
                                            tgt_key_padding_mask = Y_valid_len.cuda(),
                                            inchi_key_padding_mask = I_valid_len.cuda(),
                                            pubchemfp_key_padding_mask = P_valid_len.cuda(),
                                            memory_key_padding_mask = X_valid_len.cuda()) 
     
            loss_pharm = F.nll_loss(output_pharm.view(-1, len(tgt_vocab)),
                               fp.contiguous().view(-1),
                               ignore_index=PAD) 
            loss_smi = F.nll_loss(output_smi.view(-1, len(src_vocab)),
                               sm.contiguous().view(-1),
                               ignore_index=PAD)
            loss_inchi = F.nll_loss(output_inchi.view(-1, len(inchi_vocab)),
                               inchi.contiguous().view(-1),
                               ignore_index=PAD)
            loss_pubchemfp = F.nll_loss(output_pubchemfp.view(-1, len(pubchemfp_vocab)),
                               pubchemfp.contiguous().view(-1),
                               ignore_index=PAD)
            loss = 30*loss_pharm + 10*loss_smi + 30*loss_inchi + 20*loss_pubchemfp       

            #total loss
            running_loss = loss.sum()
            loss.sum().backward()
            optimizer.step()



            if b%500==0:
      
                print('Train {:3d}: iter {:5d} | loss {:.6f} | ppl {:.6f}'.format(e, b, loss.item(), math.exp(loss.item())))
            if b%5000==0:
                loss = evaluate(model, test_loader, tgt_vocab,src_vocab,inchi_vocab,pubchemfp_vocab)    


                print('Val {:3d}: iter {:5d} | loss {:.6f} | ppl {:.6f}'.format(e, b, loss, math.exp(loss)))
                # Save the model if the validation loss is the best we've seen so far.
                if not best_loss or loss < best_loss:
                    print("[!] saving model...")
                    torch.save(model.state_dict(), './trfm_new_%d_%d.pkl' % (e,b))
                    best_loss = loss
        state = {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':e}
        torch.save(state,'logs_model/trfm_model.pth')
        torch.save(model.state_dict(),'%s/trfm_model_%s.pkl' %(log_dir_,e,) )

if __name__ == "__main__":
    pharm_vocab_path = '../data/pubchem/pharm_vocab1.pkl'
    smi_vocab_path = '../data/pubchem/smi_vocab1.pkl'
    inchi_vocab_path = '../data/pubchem/inchi_vocab1.pkl'
    pubchemfp_vocad_path = '../data/pubchem/pubfp_vocab1.pkl'
    smi_pharm_inchi_pubchemfp_filepath = '../data/pubchem/smi_pharm_inchi_pubchem_corpus2.txt'
    try:
        train_smi(smi_vocab_path=smi_vocab_path,
                pharm_vocab_path=pharm_vocab_path,
                inchi_vocab_path=inchi_vocab_path,
                pubchemfp_vocab_path=pubchemfp_vocad_path,
                smi_pharm_inchi_pubchemfp_path=smi_pharm_inchi_pubchemfp_filepath,
                size=3000000,
                seq_len=256,n_heads=4,n_hidden=256,batch_size=128,n_layers=2)
    except KeyboardInterrupt as e:
        print("[STOP]", e)

