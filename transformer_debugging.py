#D:\work\FPGA\zipcpu\medium_articles-main\transformer_debugging.py
#In this notebook, I tried various visualizations for getting better understanding of architecture
#This notebook is a simplified transformer architecture,
#uses 1 multiheadattention layer. because if u use more than 1 , visualizations are not human readable
#but of course model is better.
#I will create another notebook for multihead
#I get the source from below URL
#https://github.com/bentrevett/pytorch-seq2seq/blob/master/6 - Attention is All You Need.ipynb
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import spacy
import numpy as np
import matplotlib
import random
import math
import time
import random
from yellowbrick.text import TSNEVisualizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import display
#these are initial parameters you can change but, u have to play with configurations
#so best is keeping these simple,and testing as is
HID_DIM = 64
ENC_LAYERS = 1
DEC_LAYERS = 1
ENC_HEADS = 1
DEC_HEADS = 1
ENC_PF_DIM = 128
DEC_PF_DIM = 128
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
def get_head_count():
    return ENC_HEADS
def get_row_count():
    return 1
def get_col_count():
    return 1
#utility class to log all intermediate steps. here saving all tensors send to add_info
#then we can query these tensors. When we want to have a seperate logger, create new instance.
class StepLogger():
    def __init__(self,capacity):
        self.tensor_datas = {}        
        self.capacity = capacity
        self.added_labels = []
    def add_info(self,tensor_data,tensor_label):
        if tensor_label not in self.added_labels:
            self.added_labels.append( tensor_label )
        if tensor_label in self.tensor_datas.keys():
            current_arr = self.tensor_datas.get(tensor_label)
            if len(current_arr) < self.capacity:
                current_arr = self.tensor_datas.get(tensor_label, [])
                current_arr.append(tensor_data)
        else:
            self.tensor_datas[tensor_label] = [tensor_data]
    def get_default_summary(self,show_data=False,summary_count=1):
        self.get_summary(self.added_labels,show_data,show_info=True,summary_count=summary_count)
    def get_summary(self,labels,show_data=False,show_info=False,summary_count=1):
        if show_info:
            print("summary_count",summary_count,"   self.capacity ",self.capacity)
        count = 0
        values = []
        for i in range(summary_count):
            #print(i," ------------------------------------------------")
            for l in labels:
                if i < len(self.tensor_datas.get(l)):
                    label_data = self.tensor_datas.get(l)[i]
                    values.append(label_data)
                    if show_info:
                        print(l)
                    if torch.is_tensor(label_data):
                        if show_info:
                            print( list(label_data.size() ) )
                    if not show_data and not torch.is_tensor(label_data):
                        if show_info:
                            print(label_data)
                    if show_data:    
                        print(label_data)
        return values              
loggers = {}                    
current_logger = StepLogger(100)   
def add_new_logger(logger_label):
    global current_logger
    if logger_label in loggers.keys() :
        #print("using existing logger")
        current_logger = loggers[logger_label]
    else:    
        loggers[logger_label]  = StepLogger(50)
        current_logger = loggers[logger_label]
def add_infos(datas,labels,labels_prefix=""):
    #print( id(current_logger))
    for i in range(len(datas)):
        final_label = labels_prefix+"@"+labels[i]
        current_logger.add_info(datas[i],final_label.strip())
#add_info   tensor_data,tensor_label      currentLogger.get_default_summary(show_data=False)
data_pipeline = []
pipeline_enabled = True
#do no want to log during eval mode
def disable_pipeline():
    global pipeline_enabled
    pipeline_enabled = False
def enable_pipeline():
    global pipeline_enabled
    pipeline_enabled = True
#sometimes i want to log some values selectively by enable/disable
#u can check how do i do use this at code, my intention is to collect
#some intermediate values
def add_pipeline_info(label,data):
    if pipeline_enabled:
        data_pipeline.append( (label,data) )
#my dummy variables for enabling/disabling some parts of network
APPLY_EVERYTHING = 1
APPLY_ONLY_TARGET_WITHOUT_SCALE = 2
APPLY_ONLY_TARGET_EMBEDDING = 3
APPLY_ONLY_SOURCE_EMBEDDING = 4
APPLY_ONLY_POS_EMBEDDING = 5
APPLY_ONLY_TARGET_WITH_SCALE = 6
#7 missing :) 9 also missing
APPLY_NOT_ATTENTION = 8
APPLY_NOT_DECODER_SELFATTENTION = 10
APPLY_ONLY_NORM_DECODER_SELFATTENTION = 11
APPLY_ONLY_DECODER_SELFATTENTION = 12
APPLY_ONLY_DROPOUT_DECODER_SELFATTENTION = 13
APPLY_NOT_POSITIONWISE = 14
#---------- Attention Types
#according to these values i create attention vectors and change the work flow of network
ATTENTION_DEFAULT = 1
ATTENTION_EQUAL_DISTRIBUTED = 2
ATTENTION_AT_BEGINNING = 3
ATTENTION_AT_END = 4
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
def tokenize(text):
    return [tok for tok in text.split(" ")]
SRC = Field(tokenize = tokenize, init_token = '<sos>', eos_token = '<eos>', lower = True, batch_first = True)
TRG = Field(tokenize = tokenize, init_token = '<sos>', eos_token = '<eos>', lower = True, batch_first = True)
#read csv file
tabular_set = torchtext.data.TabularDataset(path='eng_de.csv', format='csv',fields=[('src', SRC),('trg', TRG)])
#dump a sample
print(tabular_set[0].src)
print(tabular_set[0].trg)
train_data, valid_data, test_data = tabular_set ,tabular_set, tabular_set
#all of our vocublary is used at least 2 times,  check the csv file
SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device=",device)
#our tranining size is so small so batch size will also be small
BATCH_SIZE = 8
#create an iterator
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), 
     batch_size = BATCH_SIZE,
     sort_key = lambda x:  len(x.src),
     device = device)
class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length = 100):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
            for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
    def forward(self, src, src_mask,attention_type):
        add_infos([src],["src_source"],"Encoder") #log vectors
        add_pipeline_info(label="Encoder->attention",data=src) #log vectors
        batch_size = src.shape[0]
        src_len = src.shape[1]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        #print("src.device=",src.device)
        #print("src=",src)
        #print("self.tok_embedding=",self.tok_embedding)
        #src_embedding = self.tok_embedding(src) 
        src_embedding = self.tok_embedding.to(device)(src) 
        add_infos([src,pos,src_embedding],["src","pos","src_embedding"],"Encoder")   
        src = self.dropout((src_embedding * self.scale) + self.pos_embedding(pos))                
        for layer in self.layers:
            src = layer(src, src_mask,attention_type)
        #log vectors
        add_infos([src],["src_final"],"Encoder")            
        return src
class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device,"encoder")
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)
    def forward(self, src, src_mask,attention_type):                        
        #self attention
        _src, sattention = self.self_attention(src, src, src,attention_type, src_mask) #return x, attention
        #dropout, residual connection and layer norm
        src2 = self.self_attn_layer_norm(src + self.dropout(_src))
        ###At the end attention is learning a new representation for a sentence,here _src is a helper representation
        add_pipeline_info(label="EncoderSSS"+"@SSS",data=[src,_src,src2])  
        #positionwise feedforward
        _src2 = self.positionwise_feedforward(src2)
        #dropout, residual and layer norm
        src3 = self.ff_layer_norm(src + self.dropout(_src2))
        add_infos([_src,src2,_src2,src3,src_mask,sattention],
            ["_src","src2","_src2","src3","src_mask","sattention"],"EncoderLayer")
        return src3
class MultiHeadAttentionLayer(nn.Module):
    #since this layer is common while logging there must be a label to differantiate labels
    def __init__(self, hid_dim, n_heads, dropout, device,layer_label):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        #!!! head_dim is encoding per head so ,hid_dim must divide n_heads
        self.head_dim = hid_dim // n_heads
        self.layer_label = layer_label
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        add_infos([self.hid_dim,self.n_heads,self.head_dim],["hid_dim","n_heads","head_dim"],self.layer_label)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
    def forward(self, query, key, value,attention_type, mask = None):
        batch_size = query.shape[0]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)        
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)        
        add_infos([Q,K,V],["Q","K","V"],self.layer_label)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale               
        if mask is not None:
            #print("energy.shape=",energy.shape)
            #print("mask.shape=",mask.shape)
            #print("energy=",energy)
            #print("mask=",mask)
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = torch.softmax(energy, dim = -1)                
        ### attention is a weight over dimensions
        if attention_type==ATTENTION_EQUAL_DISTRIBUTED: #normally distributed literally means no attention
            attention = torch.tensor(np.full(attention.shape, 1/attention.shape[3]),dtype=torch.float32 )
            attention = attention.to(device) #added by ly
        #just pay attention to beginning
        if attention_type == ATTENTION_AT_BEGINNING:           
            if attention.shape[3] > 2: 
                real_shape = attention.shape
                att1 = torch.tensor(np.full((real_shape[0], real_shape[1], real_shape[2], 3), 1/3) ,dtype=torch.float32 )
                att2 = torch.tensor(np.full((real_shape[0], real_shape[1], real_shape[2], real_shape[3]-3),
                    1/7) ,dtype=torch.float32 )
                att3 = torch.cat( (att1,att2),3 )
                attention = torch.softmax(att3, dim = -1)
                attention = attention.to(device) #added by ly
        #just pay attention to end         
        if attention_type == ATTENTION_AT_END :           
            if attention.shape[3] > 2: 
                real_shape = attention.shape                
                att1 = torch.tensor(np.full((real_shape[0], real_shape[1], 
                    real_shape[2], real_shape[3] - 1), 0) ,dtype=torch.float32 )
                att2 = torch.tensor(np.full((real_shape[0], real_shape[1], real_shape[2], 1), 1) ,dtype=torch.float32 )
                att3 = torch.cat( (att1,att2),3 )
                attention = torch.softmax(att3, dim = -1)
                attention = attention.to(device)  #added by ly
        #print("attention.device=",attention.device)
        #print("V.device=",V.device)
        x1 = torch.matmul(self.dropout(attention), V)        
        add_pipeline_info(label=self.layer_label+"@QKV",data=[Q,K,V,mask,energy,attention,mask,x1])        
        x2 = x1.permute(0, 2, 1, 3).contiguous()                
        x3 = x2.view(batch_size, -1, self.hid_dim)                
        x4 = self.fc_o(x3)        
        add_infos([energy,mask,attention],["energy","mask","attention"],self.layer_label)
        add_infos([x1,x2,x3,x4],["x1","x2","x3","x4"],self.layer_label)
        return x4, attention
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)        
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):                
        x1 = self.dropout(torch.relu(self.fc_1(x)))                
        x2 = self.fc_2(x1)        
        add_infos([x1,x2],["x1","x2"],"PositionwiseFeedforwardLayer")        
        return x2
class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length = 100):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
            for _ in range(n_layers)])
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
    def forward(self, trg, enc_src, trg_mask, src_mask,attention_type=1,flow_type=1):        
        add_pipeline_info(label="Decoder_original->trg",data=trg)         
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]        
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        #trg_embedding   = self.tok_embedding(trg)
        trg_embedding   = self.tok_embedding.to(device)(trg)
        if flow_type == APPLY_EVERYTHING : #apply everything (default)
            trg = self.dropout( (trg_embedding * self.scale) + self.pos_embedding(pos))
        elif flow_type == APPLY_ONLY_TARGET_WITHOUT_SCALE: #apply only target embedding wihtout scale
            trg = self.dropout( trg_embedding )
        elif flow_type == APPLY_ONLY_TARGET_EMBEDDING: #apply without dropout
            trg = trg_embedding
        elif flow_type == APPLY_ONLY_SOURCE_EMBEDDING: 
            trg = enc_src    #just give encoder embedding,very bad embedding,expect worse!
        elif flow_type == APPLY_ONLY_POS_EMBEDDING: # apply only positional embedding
            trg = self.pos_embedding(pos)        
        elif flow_type == APPLY_ONLY_TARGET_WITH_SCALE: #apply only target embedding with scale
            trg = self.dropout( (trg_embedding * self.scale) )
        else:#THIS IS DEFAULT PATH, it is same with APPLY_EVERYTHING
            trg = self.dropout( (trg_embedding * self.scale) + self.pos_embedding(pos))
        trg_before_attention = trg 
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask,attention_type,flow_type)
        if flow_type == APPLY_NOT_ATTENTION : #revert back to vector before attention
            trg = trg_before_attention
        output = self.fc_out(trg)
        add_infos([pos,trg,output,attention,trg_embedding],
            ["pos","trg","output","attention","trg_embedding"],"Decoder")
        return output, attention
class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device,"decoder_self")
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout,
            device,"decoder_encoder_attention")
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)
    def forward(self, trg, enc_src, trg_mask, src_mask,attention_type,flow_type):
        #self attention
        add_pipeline_info(label="Decoder_self->attention",data=trg) 
        if flow_type == APPLY_NOT_DECODER_SELFATTENTION:
            _trg1 = trg
        else:
            _trg1, _ = self.self_attention(trg, trg, trg,attention_type, trg_mask)
        #dropout, residual connection and layer norm
        trg2 = self.self_attn_layer_norm(trg + self.dropout(_trg1))
        #encoder attention
        add_pipeline_info(label="Decoder->encoder_attention",data=trg2) 
        _trg2, attention = self.encoder_attention(trg2, enc_src, enc_src,attention_type, src_mask)
        if flow_type == APPLY_ONLY_NORM_DECODER_SELFATTENTION:
            trg3 = self.enc_attn_layer_norm(trg2) #only apply decoder self attention with layer
        elif flow_type == APPLY_ONLY_DECODER_SELFATTENTION: #only apply decoder self attention
            trg3 = trg2    
        elif flow_type == APPLY_ONLY_DROPOUT_DECODER_SELFATTENTION: # only apply encoder attention
            trg3 = self.dropout(_trg2)        
        else:    
            #dropout, residual connection and layer norm
            trg3 = self.enc_attn_layer_norm(trg2 + self.dropout(_trg2))
        #positionwise feedforward
        if flow_type == APPLY_NOT_POSITIONWISE:
            _trg3 = trg3
        else:    
            _trg3 = self.positionwise_feedforward(trg3)
        #dropout, residual and layer norm
        trg4 = self.ff_layer_norm(trg3 + self.dropout(_trg3))
        add_infos([_trg1,trg2,_trg2,trg3,_trg3,trg4,attention,trg_mask, src_mask],
           ["_trg1","trg2","_trg2","trg3","_trg3","trg4","attention","trg_mask", "src_mask"],"DecoderLayer")
        return trg4, attention
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask
    def forward(self, src, trg,attention_type):        
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask,attention_type)
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask,attention_type)
        add_infos([src_mask,trg_mask,enc_src,output,attention,src,trg],
            ["src_mask","trg_mask","enc_src","output","attention","src","trg"],"Seq2Seq")        
        return output, attention
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
enc = Encoder(INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device)
dec=Decoder(OUTPUT_DIM,HID_DIM,DEC_LAYERS,DEC_HEADS,DEC_PF_DIM,DEC_DROPOUT,device)
SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The model has {count_parameters(model):,} trainable parameters')
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
#dump model architecture
model.apply(initialize_weights)
LEARNING_RATE = 0.0005
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)
def train(model, iterator, optimizer, criterion, clip,attention_type):
    model.train()    
    epoch_loss = 0    
    for i, batch in enumerate(iterator):        
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()
        add_pipeline_info(label="train"+"@srctrg",data=[src,trg[:,:-1]]) 
        output, _ = model(src, trg[:,:-1],attention_type)        
        output_dim = output.shape[-1]                
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)
def evaluate(model, iterator, criterion,attention_type):
    model.eval()    
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output, _ = model(src, trg[:,:-1],attention_type)            
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
import os
import gc
N_EPOCHS = 30
CLIP = 1
best_valid_loss = float('inf')
add_new_logger("training")
model_save_path = "D:/work/FPGA/zipcpu/medium_articles-main/tut6-model_en_de.pt"
model_to_train = model
for epoch in range(N_EPOCHS):
    gc.collect()
    if os.path.isfile(model_save_path) :
        print("loading model",model_save_path)
        #If you want to save the model enable below,do it for big datasets
        #model_to_train.load_state_dict(torch.load(model_save_path))
    start_time = time.time()
    enable_pipeline()
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP,attention_type=1)
    disable_pipeline()
    valid_loss = evaluate(model, valid_iterator, criterion,attention_type=1)    
    end_time = time.time()    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)    
    if valid_loss < best_valid_loss:
        print("saving model",model_save_path," with loss ",valid_loss)
        best_valid_loss = valid_loss
        torch.save(model_to_train.state_dict(), model_save_path)
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')    
import pandas as pd
import seaborn as sns
#Utility method for translating a sentence and then displaying Attention and Logits
#Logits = the vector of raw (non-normalized) predictions that a classification model generates, 
#which is ordinarily then passed to a normalization function(softmax). The softmax function
# generates a vector of (normalized) probabilities with one value for each possible class.
def translate_info(src,draw_charts,attention_type=1,dump_result=True,flow_type=1):
    if isinstance(src,str):
        src = src.split(" ")
    logger_name = "".join(src)+str( random.randint(1,180001) )
    add_new_logger(logger_name)
    if dump_result:
        print(f'source = {src}')
    translation, attention,all_outputs = translate_sentence(src, SRC, TRG, model, device,
        attention_type=attention_type,flow_type=flow_type)
    if dump_result:
        print(f'predicted target = {translation}')
    output_words = translation
    indexes = []
    for index_w,w in enumerate(translation):
        if index_w == 0:
            indexes.append("sos->"+w.replace("<","").replace(">",""))
        else:    
            indexes.append(translation[index_w -1 ]+"->"+w.replace("<","").replace(">",""))        
    indexes = [ str(i+1)+")"+index for i,index in enumerate(indexes) ]
    #for better display i do not include "unk","pad",    
    all_outputs = np.array(all_outputs)[:,2:]
    df2 = pd.DataFrame(all_outputs,columns=["sos","eos"]+TRG.vocab.itos[4:],index=indexes)
    pd.options.display.float_format = '{:,.2f}'.format
    cm = sns.light_palette("blue", as_cmap=True)
    df2.style.format("{:.2%}")
    df2.style.set_table_attributes("style='display:inline'").set_caption('Logits for translation')
    styled_df = df2.style.background_gradient(cmap=cm).set_precision(2)   
    pd.options.display.float_format = '{:,.2f}'.format
    if draw_charts:
        print("Logits")
        display(styled_df)    
        display_attention(src, indexes, attention,n_heads=get_head_count(),
            n_rows = get_row_count(), n_cols = get_col_count())
    return loggers[logger_name],all_outputs,translation
def get_color(word):
    cmap = plt.get_cmap('viridis')
    distinct_words = SRC.vocab.itos[4:]
    colors = cmap(np.linspace(0, 1, len(distinct_words)))
    color_map = { w:colors[i] for i,w in enumerate(distinct_words) }
    for distinct_word in distinct_words:
        if  word.find(distinct_word) == 0: #if starstwith
            return color_map[distinct_word]
    return color_map[distinct_words[0]]
#map the input on a 3d space. #sometimes we see better positioning in 2d sometimes in 3d so i do both
def map_on_3d(data_2d,words,color_mapping:None,exceptional_words=[],
        use_word_colors=False,fig_size=[19,11]):
    pass
    '''
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(fig_size[0], fig_size[1])
    ax = plt.axes(projection ="3d")
    #scatter result words
    ax.scatter3D(data_2d[:, 0], data_2d[:, 1], data_2d[:, 2])
    #put an annotation on x,y cordinates for words
    for i, word in enumerate(words):
        if color_mapping is not None:
            ax.text(data_2d[i, 0], data_2d[i, 1], data_2d[i, 2],word,color=color_mapping[i] )
        elif word in exceptional_words:
            ax.text(data_2d[i, 0], data_2d[i, 1], data_2d[i, 2],word,color='#0000ff' )
        elif use_word_colors:
            ax.text(data_2d[i, 0], data_2d[i, 1], data_2d[i, 2],word,color=get_color(word))
        elif "eat" in word:
            ax.text(data_2d[i, 0], data_2d[i, 1], data_2d[i, 2],word,color='#0000ff')
        elif "drink" in word:
            ax.text(data_2d[i, 0], data_2d[i, 1], data_2d[i, 2],word,color='#00ff00')
        elif "read" in word:
            ax.text(data_2d[i, 0], data_2d[i, 1], data_2d[i, 2],word,color='#ff0000')     
        else:
            ax.text(data_2d[i, 0], data_2d[i, 1], data_2d[i, 2],word,color='#ff00dd')
    plt.show()
    '''
#map the input on a 2d space      
def map_on_2d(data_2d,words,color_mapping:None,exceptional_words=[],
        use_word_colors=False,fig_size=[19,11]):
    pass
    '''
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(fig_size[0], fig_size[1])
    #scatter result words
    plt.scatter(data_2d[:, 0], data_2d[:, 1])        
    #put an annotation on x,y cordinates for words
    for i, word in enumerate(words):
        if color_mapping is not None:
            plt.annotate(word, xy=(data_2d[i, 0], data_2d[i, 1]),color=color_mapping[i] )
        elif word in exceptional_words:
            plt.annotate(word, xy=(data_2d[i, 0], data_2d[i, 1]),color='#0000ff' )
        elif use_word_colors:
            plt.annotate(word, xy=(data_2d[i, 0], data_2d[i, 1]),color=get_color(word))
        elif "eat" in word:
            plt.annotate(word, xy=(data_2d[i, 0], data_2d[i, 1]),color='#0000ff')
        elif "drink" in word:
            plt.annotate(word, xy=(data_2d[i, 0], data_2d[i, 1]),color='#00ff00')
        elif "read" in word:
            plt.annotate(word, xy=(data_2d[i, 0], data_2d[i, 1]),color='#ff0000')     
        else:
            plt.annotate(word, xy=(data_2d[i, 0], data_2d[i, 1]),color='#ff00dd')
    plt.show() 
    '''     
def encoding_to_sentence(encodings,language):
    return [encoding_to_word(i,language) for i in encodings ]   
def encoding_to_word(embedding_index,language):
    return language.vocab.itos[embedding_index]     
#show a dataframe with style
def draw_df(data,columns,indexes):
    #df2 = pd.DataFrame(data,columns=columns,index=indexes)
    df2 = pd.DataFrame(data)
    cm = sns.light_palette("blue", as_cmap=True)
    styled_df = df2.style.background_gradient(cmap=cm).set_precision(2)
    display(styled_df)
def translate_sentence(sentence, src_field, trg_field, model, device,attention_type, max_len = 10,flow_type=1):
    model.eval()
    if isinstance(sentence, str):
        nlp = spacy.load('de')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)
    #generate all encoder state
    with torch.no_grad():
        #print("src_tensor.device=",src_tensor.device)
        #print("src_mask.device=",src_mask.device)
        enc_src = model.encoder(src_tensor, src_mask,attention_type)
    #add Start of sentence token for target    
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    ### hold output of linear layer at everytime step generate 1 output
    all_outputs = []
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)
        #generate next word
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask,attention_type,flow_type)
        #all_outputs.append(output[:, -1:, :].numpy().flatten())
        all_outputs.append(output[:, -1:, :].cpu().numpy().flatten())
        pred_token = output.argmax(2)[:,-1].item()
        trg_indexes.append(pred_token)
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    return trg_tokens[1:], attention,all_outputs
def display_attention(sentence, translation, attention, n_heads = get_head_count(), 
        n_rows = get_row_count(), n_cols = get_col_count()):
    pass
    '''
    assert n_rows * n_cols == n_heads
    figsize = (5,5) if n_cols == 1 else (20,15)
    fig = plt.figure(figsize=figsize) 
    for i in range(n_heads):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        _attention = attention.squeeze(0)[i].cpu().detach().numpy()
        cax = ax.matshow(_attention, cmap='bone')
        ax.tick_params(labelsize=12)
        ax.set_xticklabels(['']+['<sos>']+[t.lower() for t in sentence]+['<eos>'], rotation=45)
        ax.set_yticklabels(['']+translation)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()
    plt.close()
    '''  
_ = translate_info("we can eat apple",attention_type=ATTENTION_DEFAULT,draw_charts=True)
_ = translate_info("eat apple can i",attention_type=ATTENTION_DEFAULT,draw_charts=True)
_ = translate_info("i can eat apple",attention_type=ATTENTION_DEFAULT,draw_charts=True)
#at konnen -> attention is at apfel
#at apfel attention is at eat
_ = translate_info("i want to eat apple",attention_type=ATTENTION_EQUAL_DISTRIBUTED,draw_charts=True)
#since no real attention network cannot predict well
_ = translate_info("i want to eat apple",attention_type=ATTENTION_AT_BEGINNING,draw_charts=True)
_ = translate_info("i eat apple",attention_type=ATTENTION_AT_BEGINNING,draw_charts=True)
#_ = translate_info("i can read book",attention_type=ATTENTION_AT_END,draw_charts=True)
_ = translate_info("i want to eat apple",attention_type=ATTENTION_AT_BEGINNING,draw_charts=True)
#utility class for future visualizations
class DebugStepWord:
    def __init__(self,sentence,translation):
        self.q = None
        self.k = None
        self.v = None
        self.energy = None
        self.attention = None
        self.mask = None
        self.x1 = None
        self.sentence = sentence
        self.translation = translation
        self.original_embedding = None 
        self.word_vector = None #only filled if done at world level
        self.word_translation = None #only filled if done at world level
        self.word_embeddding  = None #only filled if done at world level
    def summary(self):
        print("q",self.q.shape)
        print("k",self.k.shape)
        print("v",self.v.shape)
        print("attention",self.attention.shape)
        print("energy",self.energy.shape)
        print("mask",self.mask.shape)
        print("x1",self.x1.shape)
        print("x1",self.x1.shape)
        print("word_translation",self.word_translation)
        print("translation",self.translation)
#for all sentences generate a map
all_sentences = {}
for i, batch in enumerate(train_iterator):
    src = batch.src
    trg = batch.trg
    for i in range( (src.shape[0]) ):
        all_sentences[" ".join(encoding_to_sentence(src[i],SRC))] = " ".join(encoding_to_sentence(trg[i],TRG))
def find_src_sentence_from_trg(trg_sentence):
    for src in all_sentences.keys():
        if all_sentences[src].find( trg_sentence) >= 0:
            return src
    assert 1 == 2 #must not reach here
#last word is trimmed so here find full version of sentence    
def find_fullsentence_from_trg(trg_sentence):
    for src in all_sentences.keys():
        if all_sentences[src].find( trg_sentence) >= 0:
            return all_sentences[src]
    assert 1 == 2    
assert find_fullsentence_from_trg("ich essen apfel") == "<sos> ich essen apfel <eos> <pad>"
#prepare intermediate vector maps. #traverse the vectors generated for data_pipeline
#and generate DebugStepWord object for each, here i know the order of addition
#of these vectors so i create object on QKV and then update. #here i log these for sentences
steps_decoder_bysentence = []
LANG = TRG
last_sentence_encoding_batch = None
last_sentence_original_batch = None
for i in range(len(data_pipeline)):
    current_label = data_pipeline[i][0]
    if "decoder_encoder_attention@QKV" == current_label: # TODO encoder@QKV
         sentence_batch = last_sentence_encoding_batch
         batch_len = sentence_batch.shape[0]   
         for sent_index in range(batch_len):  
            sentence = sentence_batch[sent_index]
            data_part = data_pipeline[i][1]
            original_embedding = last_sentence_original_batch[sent_index]
            translation = " ".join( encoding_to_sentence(original_embedding,TRG) )
            dstep = DebugStepWord(sentence,translation)
            dstep.original_embedding = original_embedding 
            dstep.q = data_pipeline[i][1][0][sent_index].squeeze().detach()
            dstep.k = data_pipeline[i][1][1][sent_index].squeeze().detach()
            dstep.v = data_pipeline[i][1][2][sent_index].squeeze().detach()
            dstep.energy = data_pipeline[i][1][4][sent_index].squeeze().detach()
            dstep.attention = data_pipeline[i][1][5][sent_index].squeeze().detach()
            dstep.mask = data_pipeline[i][1][6][sent_index].squeeze().detach()
            dstep.x1 = data_pipeline[i][1][7][sent_index].squeeze().detach()
            steps_decoder_bysentence.append(dstep)
    if "Decoder->encoder_attention" == current_label: #TODO Encoder->attention
         last_sentence_encoding_batch = data_pipeline[i][1]
    if "Decoder_original->trg" == current_label: #TODO Encoder->attention
         last_sentence_original_batch = data_pipeline[i][1]    
assert N_EPOCHS * len(steps_decoder_bysentence) * 36
print( " epochs: ",N_EPOCHS , " log length : ", len(steps_decoder_bysentence) )
# num epoch x 36 sentence  
assert steps_decoder_bysentence[0].v.shape[0] == 7
assert steps_decoder_bysentence[0].v.shape[1] == HID_DIM 
#now by using same object, i create these maps for word also
steps_decoder_byword = []
for step in steps_decoder_bysentence:
    word_count = step.q.shape[0]
    for i in range(word_count):        
        word_step = DebugStepWord(step.sentence,step.translation)    
        #print("step.q",step.q.shape)
        word_step.q = step.q[i]
        word_step.k = step.k[i]
        word_step.v = step.v[i]
        word_step.energy = step.energy[i]
        word_step.attention = step.attention[i]
        word_step.mask = step.mask[i]
        word_step.x1 = step.x1[i]
        word_step.sentence = step.sentence
        word_step.translation = step.translation
        word_step.word_vector = step.sentence[i]
        word_step.word_embeddding = step.original_embedding[i]
        #print("step.sentence.squeeze()[i]",word_step.word_embeddding )
        word_step.word_translation = encoding_to_word(word_step.word_embeddding,LANG)
        steps_decoder_byword.append( word_step )
assert steps_decoder_byword[0].q.shape[0] == 64 #indexes must be single dimension
assert steps_decoder_byword[0].word_vector.shape[0] == 64 #indexes must be single dimension
def get_by_sentence(search_sentence,steps):
    return [ step for step in steps if step.translation.find(search_sentence) >=0 ]
def get_by_word(search_sentence,keywords,steps_word):
    steps_for_sample = get_by_sentence(search_sentence,steps_word)
    keyword_map = {}
    for step in steps_for_sample :
        if step.word_translation in keywords :
            if step.word_translation in keyword_map.keys():
                keyword_map[step.word_translation] = keyword_map[step.word_translation]+[step]
            else:    
                keyword_map[step.word_translation] = [step]
    return keyword_map
#for a target sentence(german) find intermediate steps
def steps_for_target(search_sentence):
    steps_for_sample = get_by_sentence(search_sentence,steps_decoder_bysentence)
    for index,step in enumerate(steps_for_sample):
        if index == 0 or (index+1) % 10 == 0:
            print("training epoch :",index+1)
            #attention = step.attention.detach().numpy()
            attention = step.attention.detach().cpu().numpy() #modified by ly
            splits = step.translation.split(" ")
            splits = [s.replace("<","").replace(">","") for s in splits]
            splits = [ splits[i]+"->"+splits[i+1]  for i in range(len(splits)-1)] + [splits[len(splits)-1]+"->eos"]            
            splits = [ str(index+1)+")"+s for index,s in enumerate(splits)]
            src_sentence_splits = find_src_sentence_from_trg(step.translation).split(" ")
            src_sentence_splits = [s.replace("<","").replace(">","").replace("pad","pad"+str(index)) 
                for index,s in enumerate(src_sentence_splits)]            
            draw_df(np.array(attention),columns=src_sentence_splits,indexes=splits)
#steps_for_target("wir essen apfel")
#steps_for_target("wir mochten apfel essen")
steps_for_target("wir konnen apfel essen")
#includes : list of vectors to add Q,K,V,A
#get steps for stated vectors
def map_word_by_sentence(steps_for_sample,includes="QV",ranks=[1,15,N_EPOCHS]):
    Xs = []
    Ys = []
    for key in steps_for_sample.keys():
        for index,step in enumerate(steps_for_sample[key]):
            #print("index:",index)            
            if "Q" in includes:
                Xs.append( step.q.cpu().numpy())
                Ys.append( key+"@Q@"+str(index+1) )
            if "K" in includes:
                Xs.append( step.k.cpu().numpy())
                Ys.append( key+"@K@"+str(index+1) )
            if "V" in includes:
                Xs.append( step.v.cpu().numpy())
                Ys.append( key+"@V@"+str(index+1) )
            if "X" in includes:
                Xs.append( step.x1.cpu().numpy())
                Ys.append( key+"@X1@"+str(index+1) )     
    return filter_by_indexes(Xs,Ys,ranks)
def filter_by_indexes(Xs,Ys,ranks):
    Xnew = []
    Ynew = []
    for val,key in zip(Xs,Ys):        
        item_rank = key.split("@")[2]
        if int(item_rank) in ranks:
            Xnew.append(val)
            Ynew.append(key)
    return Xnew,Ynew    
def filter_by_key(Xs,Ys,keys):
    X_new,Y_new = [],[]
    for val,key in zip(Xs,Ys):
        if key.split("@")[0] in keys:
            X_new.append(val)
            Y_new.append(key)
    return X_new,Y_new        
def get_as_map(Xs,Ys,map_keys):
    xy_map = {}
    for val,key in zip(Xs,Ys):
        if key in map_keys:
            xy_map[key]=val
    return xy_map       
#altough wir and essen are same words, their embedding changed over time
#mapping on 2d changes over time, and network learns to differentiate better on final encoding
#final encoding is on 30th step. Here mapping is with wir and essen
search_sentence = "wir konnen apfel essen"
steps_for_sample = get_by_word(search_sentence,["essen","wir"],steps_decoder_byword)
X_s1,Y_s1 = map_word_by_sentence(steps_for_sample,includes="QKVX",ranks=[1,15,30])
search_sentence2 = "wir konnen brot essen"
steps_for_sample2 = get_by_word(search_sentence2,["essen","wir"],steps_decoder_byword)
X_s2,Y_s2 = map_word_by_sentence(steps_for_sample2,includes="QKVX",ranks=[1,15,30])
Xs = X_s1 + X_s2
Ys = [ s for s in Y_s1] + [ s for s in Y_s2]
color_mapping = [ "#0000ff" for s in Y_s1] + [ "#ff0000" for s in Y_s2]
pca = PCA(n_components=2)
X_transformed = pca.fit_transform(np.array(Xs))
map_on_2d(np.array(X_transformed),Ys,color_mapping=color_mapping,use_word_colors=False,fig_size=[15,15]) 
pca = PCA(n_components=3)
X_transformed = pca.fit_transform(np.array(Xs))
map_on_3d(np.array(X_transformed),Ys,color_mapping=color_mapping,use_word_colors=False,fig_size=[15,15]) 
#show only essen
search_sentence = "wir konnen apfel essen"
steps_for_sample = get_by_word(search_sentence,["essen"],steps_decoder_byword)
X_s1,Y_s1 = map_word_by_sentence(steps_for_sample,includes="QKVX",ranks=[1,15,30])
search_sentence2 = "wir konnen brot essen"
steps_for_sample2 = get_by_word(search_sentence2,["essen"],steps_decoder_byword)
X_s2,Y_s2 = map_word_by_sentence(steps_for_sample2,includes="QKVX",ranks=[1,15,30])
Xs = X_s1 + X_s2
Ys = [ s for s in Y_s1] + [ s for s in Y_s2]
color_mapping = [ "#0000ff" for s in Y_s1] + [ "#ff0000" for s in Y_s2]
pca = PCA(n_components=2)
X_transformed = pca.fit_transform(np.array(Xs))
map_on_2d(np.array(X_transformed),Ys,color_mapping=color_mapping,use_word_colors=False,fig_size=[12,12]) 
pca = PCA(n_components=3)
X_transformed = pca.fit_transform(np.array(Xs))
map_on_3d(np.array(X_transformed),Ys,color_mapping=color_mapping,use_word_colors=False,fig_size=[12,12]) 
test_logger,all_outputs,aa = translate_info(tabular_set[0].src,draw_charts=False,dump_result=False)
trg4 = test_logger.get_summary(labels=["DecoderLayer@trg4"],show_data=False,summary_count=10)
trg4[4].shape
trg4[len(trg4) -1 ].squeeze().shape
print("aa=",aa)
#most sentences have a translation of length 5 ,so for those sentences, check the trg4 vector
#for all steps of a translation and return the vector for index
#since all steps, expected words have a list, i give filter words,to only get target words
def get_translate_trg_vector(index,attention_type,filter_words):
    sentences = [ " ".join( tabular_set[i].src) for i in range(len(tabular_set)) ]
    filtered_sentences = []    
    sentences_embeddings = []
    step_output_words = []
    for i in range(len(tabular_set)):
        test_logger,all_outputs,output_words = translate_info(
            tabular_set[i].src,attention_type=attention_type,draw_charts=False,dump_result=False)
        trg4 = test_logger.get_summary(labels=["DecoderLayer@trg4"],show_data=False,summary_count=10)
        trg_all = trg4[len(trg4) -1 ].squeeze()
        if trg_all.shape[0] > index : #because of length of sentence,positions do no match just do this for same size items
            if output_words[index] in filter_words:
                sentences_embeddings.append(trg_all[index].cpu().numpy().flatten())
                filtered_sentences.append( sentences[i] )            
                step_output_words.append(output_words[index])
    return sentences_embeddings,filtered_sentences,step_output_words
#show multiple items in 2d with different colors
def show_embeddings(invalid_embeddings,invalid_sentences,
        valid_sentences_embeddings,valid_sentences,fig_size=[18,8]):    
    pass
    '''
    Xs = valid_sentences_embeddings
    Ys = valid_sentences
    #I want to mark sentences 
    if len(invalid_sentences)> 0 :
        invalid_sentences = [ "<<<"+s+">>>" for s in invalid_sentences]
        Xs = Xs+ [embed for embed in invalid_embeddings]
        Ys = Ys + invalid_sentences 
    pca = PCA(n_components=2)
    result = pca.fit_transform(np.array(Xs))
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(fig_size[0], fig_size[1])
    #scatter result words
    plt.scatter(result[:, 0], result[:, 1])
    words = list(Ys)
    #put an annotation on x,y cordinates for words, color according to verb
    for i, word in enumerate(words):
        if word in invalid_sentences:
            if "eat" in word:
                plt.annotate(word.replace("<<<","").replace(">>>","!!!"), xy=(result[i, 0], result[i, 1]),color='#dd00dd')
            elif "drink" in word:
                plt.annotate(word.replace("<<<","").replace(">>>","]]]"), xy=(result[i, 0], result[i, 1]),color='#dd44dd')
            else:
                plt.annotate(word.replace("<<<",""), xy=(result[i, 0], result[i, 1]),color='#dd88dd')
        elif "eat" in word:
            plt.annotate(word, xy=(result[i, 0], result[i, 1]),color='#0000ff')
        elif "drink" in word:
            plt.annotate(word, xy=(result[i, 0], result[i, 1]),color='#00ff00')
        elif "read" in word:
            plt.annotate(word, xy=(result[i, 0], result[i, 1]),color='#ff0000')     
        else:
            plt.annotate(word, xy=(result[i, 0], result[i, 1]),color='#ff00dd')
    plt.show()
    '''
#ATTENTION_DEFAULT #ATTENTION_EQUAL_DISTRIBUTED
#for i in range(5):
#    print(i)
#    valid_sentences_embeddings,filtered_sentences = get_translate_trg_vector(
#        i ,ATTENTION_EQUAL_DISTRIBUTED)
#    show_on_vis_multi([],[],valid_sentences_embeddings,filtered_sentences,fig_size=[18,18])    
#for i in range(5):
#    print(i)
#    valid_sentences_embeddings,filtered_sentences = get_translate_trg_vector(i,ATTENTION_DEFAULT)
#    invalid_sentences_embeddings,invalid_filtered_sentences = \
#        get_translate_trg_vector(i,ATTENTION_EQUAL_DISTRIBUTED)
#    show_on_vis_multi([],[],valid_sentences_embeddings,filtered_sentences,fig_size=[18,18])    
#    show_on_vis_multi(invalid_sentences_embeddings,invalid_filtered_sentences,
#        valid_sentences_embeddings,filtered_sentences)
#invalid_embeddings,invalid_sentences,step_output_words=get_translate_trg_vector(
#    2,ATTENTION_EQUAL_DISTRIBUTED)
#step_output_words
# SHOW EMBEDDINGS BY TARGET, SUBJECT
word_index_target = 0
filter_words = ["ich","wir"]
v_embeddings,v_sentences,v_outputs= get_translate_trg_vector( 
    word_index_target,ATTENTION_DEFAULT,filter_words)
inv_embeddings,inv_sentences,inv_outputs=get_translate_trg_vector( 
    word_index_target,ATTENTION_EQUAL_DISTRIBUTED,filter_words)
show_embeddings([],[],v_embeddings,v_sentences,fig_size=[12,6])    
show_embeddings(inv_embeddings,inv_sentences,v_embeddings,v_sentences,fig_size=[12,6])
# SHOW EMBEDDINGS BY TARGET, MODAL VERBS
word_index_target = 1
filter_words = ["konnen","mochten"]
v_embeddings,v_sentences,v_outputs= get_translate_trg_vector( 
    word_index_target,ATTENTION_DEFAULT,filter_words)
inv_embeddings,inv_sentences,inv_outputs=get_translate_trg_vector( 
    word_index_target,ATTENTION_EQUAL_DISTRIBUTED,filter_words)
show_embeddings([],[],v_embeddings,v_sentences,fig_size=[12,6])    
show_embeddings(inv_embeddings,inv_sentences,v_embeddings,v_sentences,fig_size=[12,6])
# SHOW EMBEDDINGS BY TARGET, OBJECTS
word_index_target = 2
filter_words = ["brot","apfel","bier","wasser","buch","zeitung"]
v_embeddings,v_sentences,v_outputs= get_translate_trg_vector( 
    word_index_target,ATTENTION_DEFAULT,filter_words)
inv_embeddings,inv_sentences,inv_outputs=get_translate_trg_vector( 
    word_index_target,ATTENTION_EQUAL_DISTRIBUTED,filter_words)
show_embeddings([],[],v_embeddings,v_sentences,fig_size=[12,6])    
show_embeddings(inv_embeddings,inv_sentences,v_embeddings,v_sentences,fig_size=[12,6])
# SHOW EMBEDDINGS BY TARGET, VERBS
word_index_target = 3
filter_words = ["essen","lesen","trinken"]
v_embeddings,v_sentences,v_outputs= get_translate_trg_vector( 
    word_index_target,ATTENTION_DEFAULT,filter_words)
inv_embeddings,inv_sentences,inv_outputs=get_translate_trg_vector( 
    word_index_target,ATTENTION_EQUAL_DISTRIBUTED,filter_words)
show_embeddings([],[],v_embeddings,v_sentences,fig_size=[12,6])    
show_embeddings(inv_embeddings,inv_sentences,v_embeddings,v_sentences,fig_size=[12,6])
word_index_target = 4
filter_words = ["<eos>"]
v_embeddings,v_sentences,v_outputs= get_translate_trg_vector(word_index_target,ATTENTION_DEFAULT,filter_words)
inv_embeddings,inv_sentences,inv_outputs=get_translate_trg_vector(
    word_index_target,ATTENTION_EQUAL_DISTRIBUTED,filter_words)
#print(v_outputs)
#print(inv_outputs)
show_embeddings([],[],v_embeddings,v_sentences,fig_size=[12,12])    
show_embeddings(inv_embeddings,inv_sentences,v_embeddings,v_sentences,fig_size=[12,12])
#test all flow types
#flow_types = [1,2,3,4,5,6,8,10,11,12,13,14]
flow_types = [1,2,3,5,6,8,10,11,12,13,14]
apply_map = {}
apply_map[APPLY_EVERYTHING] = "APPLY_EVERYTHING"
apply_map[APPLY_NOT_DECODER_SELFATTENTION] = "APPLY_NOT_DECODER_SELFATTENTION"
apply_map[APPLY_ONLY_NORM_DECODER_SELFATTENTION] = \
    "APPLY_ONLY_NORM_DECODER_SELFATTENTION"
apply_map[APPLY_ONLY_DECODER_SELFATTENTION] = "APPLY_ONLY_DECODER_SELFATTENTION"
apply_map[APPLY_ONLY_DROPOUT_DECODER_SELFATTENTION] = \
    "APPLY_ONLY_DROPOUT_DECODER_SELFATTENTION"
apply_map[APPLY_NOT_POSITIONWISE] = "APPLY_NOT_POSITIONWISE"
apply_map[APPLY_ONLY_TARGET_WITHOUT_SCALE] = "APPLY_ONLY_TARGET_WITHOUT_SCALE"
apply_map[APPLY_ONLY_TARGET_EMBEDDING] = "APPLY_ONLY_TARGET_EMBEDDING"
apply_map[APPLY_ONLY_SOURCE_EMBEDDING] = "APPLY_ONLY_SOURCE_EMBEDDING"
apply_map[APPLY_ONLY_POS_EMBEDDING] = "APPLY_ONLY_POS_EMBEDDING"
apply_map[APPLY_ONLY_TARGET_WITH_SCALE] = "APPLY_ONLY_TARGET_WITH_SCALE"
apply_map[APPLY_NOT_ATTENTION] = "APPLY_NOT_ATTENTION"
#Open above to check all types enable/disable
for flow_type in flow_types:
    print("------------- flow type : ",apply_map[flow_type])
    _,_,_ = translate_info("we want to eat apple",attention_type=1,draw_charts=True,flow_type=flow_type)
    _,_,_ = translate_info("we can eat apple",attention_type=1,draw_charts=True,flow_type=flow_type)
_,_,_ = translate_info("we want to eat apple",attention_type=1,draw_charts=True,
                       flow_type=APPLY_NOT_POSITIONWISE)
#_,_,_ = translate_info("we can eat apple",attention_type=1,draw_charts=True,
#                       flow_type=APPLY_NOT_POSITIONWISE)
#_,_,_ = translate_info("we eat apple",attention_type=1,draw_charts=True,flow_type=APPLY_NOT_POSITIONWISE)
_ = translate_info("i can read book",attention_type=ATTENTION_DEFAULT,draw_charts=True)
#Open to see other samples
#_ = translate_sentence("i can read book",attention_type=ATTENTION_AT_END,draw_charts=True)
#_ = translate_sentence("i want to read apple",attention_type=ATTENTION_DEFAULT,draw_charts=True)
#_ = translate_sentence("i want to eat book",attention_type=ATTENTION_DEFAULT,draw_charts=True)
#_ = translate_sentence("i can eat book",attention_type=ATTENTION_DEFAULT,draw_charts=True)
#attention_types = [1,2,3,4]
#for attention_type in attention_types:
#    _ = translate_sentence("i can eat bread",attention_type=attention_type,draw_charts=True)
#_ = translate_sentence("we beer can read",attention_type=1,draw_charts=True)
_ = translate_info("apple eat can we",attention_type=ATTENTION_DEFAULT,draw_charts=True)
### altough order is not correct network can correct it, but this is a bit overfitting
###and it wont happen if we use a big set.
#Open to see other samples
#_ = test_sentence("apple eat we can",attention_type=ATTENTION_DEFAULT,draw_charts=True)
#_ = test_sentence("apple eat can we",attention_type=1,draw_charts=True)
#when no object distinctive, if we put all verbs lesen is chosen over others.
_ = translate_info("eat drink read we can",attention_type=ATTENTION_DEFAULT,draw_charts=True)
#show self attentions on encoder layer
#we have 36 sentences, change below values to see different sentence over training
#Change range indexes to see others..
for index in range(25,27):
    test_logger,all_outputs,_ = translate_info(tabular_set[index].src,draw_charts=False,attention_type=1)
    encoded_attention = test_logger.get_summary(["EncoderLayer@sattention"],show_data=False)        
    output_words = ["sos"]+tabular_set[index].src  +["eos"]
    for head_index in range(encoded_attention[0].shape[1]):
        all_outputs0 = encoded_attention[0][:,head_index,:,:]
        draw_df(all_outputs0.squeeze().cpu().numpy(),output_words,output_words)
last_sentence_original_batch = None
#collect encoder data
steps_encoder_bysentence = []
LANG = SRC          
last_sentence_encoding_batch = None
for i in range(len(data_pipeline)):
    current_label = data_pipeline[i][0]
    if "encoder@QKV" == current_label: 
         sentence_batch = last_sentence_original_batch
         batch_len = sentence_batch.shape[0]   
         #assert batch_len == len(data_pipeline[i][1] )         
         for sent_index in range(batch_len):  
            sentence = sentence_batch[sent_index]
            data_part = data_pipeline[i][1]
            original_embedding = last_sentence_original_batch[sent_index]
            translation = " ".join( encoding_to_sentence(original_embedding,LANG) )
            dstep = DebugStepWord(sentence,translation)
            dstep.original_embedding = original_embedding 
            dstep.q = data_pipeline[i][1][0][sent_index].squeeze().detach()
            dstep.k = data_pipeline[i][1][1][sent_index].squeeze().detach()
            dstep.v = data_pipeline[i][1][2][sent_index].squeeze().detach()
            dstep.energy = data_pipeline[i][1][4][sent_index].squeeze().detach()
            dstep.attention = data_pipeline[i][1][5][sent_index].squeeze().detach()
            dstep.mask = data_pipeline[i][1][6][sent_index].squeeze().detach()
            dstep.x1 = data_pipeline[i][1][7][sent_index].squeeze().detach()
            steps_encoder_bysentence.append(dstep)
    if "Encoder->attention" == current_label: 
         last_sentence_original_batch = data_pipeline[i][1]
assert list(steps_encoder_bysentence[0].q.shape) == [7,64]
assert N_EPOCHS * len(steps_encoder_bysentence) * 36
print( " epochs: ",N_EPOCHS , " log length : ", len(steps_encoder_bysentence) )
assert steps_encoder_bysentence[0].v.shape[0] == 7
assert steps_encoder_bysentence[0].v.shape[1] == HID_DIM 
#collect stats by word
steps_encoder_byword = []
for step in steps_encoder_bysentence:
    word_count = step.q.shape[0]
    for i in range(word_count):        
        word_step = DebugStepWord(step.sentence,step.translation)    
        #print("step.q",step.q.shape)
        word_step.q = step.q[i]
        word_step.k = step.k[i]
        word_step.v = step.v[i]
        word_step.energy = step.energy[i]
        word_step.attention = step.attention[i]
        word_step.mask = step.mask[i]
        word_step.x1 = step.x1[i]
        word_step.sentence = step.sentence
        word_step.translation = step.translation
        word_step.word_vector = step.sentence[i]
        word_step.word_embeddding = step.original_embedding[i]
        #print("step.sentence.squeeze()[i]",word_step.word_embeddding )
        word_step.word_translation = encoding_to_word(word_step.word_embeddding,LANG)
        steps_encoder_byword.append( word_step )
assert steps_encoder_byword[0].q.shape[0] == 64 
#VECTORS CHANGING OVER TIME, change ranks or indexes to see different combinations
#altough "eat" and "we" are same words, their encodings differ in space,
search_sentence = "we want to eat apple" #"eat","we","apple","want"
steps_for_sample = get_by_word(search_sentence,["eat"],steps_encoder_byword)
X_s1,Y_s1 = map_word_by_sentence(steps_for_sample,includes="QKVX",ranks=[1,15,30])
search_sentence2 = "we want to eat bread"
steps_for_sample2 = get_by_word(search_sentence2,["eat"],steps_encoder_byword)
X_s2,Y_s2 = map_word_by_sentence(steps_for_sample2,includes="QKVX",ranks=[1,15,30])
Xs = X_s1 + X_s2
Ys = [ s for s in Y_s1] + [ s for s in Y_s2]
color_mapping = [ "#0000ff" for s in Y_s1] + [ "#ff0000" for s in Y_s2]
pca = PCA(n_components=2)
X_transformed = pca.fit_transform(np.array(Xs))
map_on_2d(np.array(X_transformed),Ys,color_mapping=color_mapping,use_word_colors=False,fig_size=[12,8]) 
pca = PCA(n_components=3)
X_transformed = pca.fit_transform(np.array(Xs))
map_on_3d(np.array(X_transformed),Ys,color_mapping=color_mapping,use_word_colors=False,fig_size=[12,12]) 
###when we are checking encoder self attention, we see that q,v,k is less saturated,more clustered
#because no other source of input,or effect.  #collect related vectors
def collect_sentence_data(sentences,vector_name,indexes,keywords,use_last_vector=False,dump_result=False):
    print("vector_name",vector_name)
    Xs = []
    Ys = []
    for sentence_index,sentence in enumerate(sentences):
        test_logger,_,translation = translate_info(sentence,
            attention_type=1,draw_charts=False,dump_result=dump_result)
        x4 = test_logger.get_summary([vector_name],show_data=False,summary_count=10)
        if use_last_vector:
            x4 = x4[len(x4)-1]
        for i,index in enumerate(indexes):
            #if use_last_vector it is decoder so we must use result from translation
            if use_last_vector:
                key = translation[index]
            else:    
                key = keywords[i]
            val = x4[0].squeeze()[index].cpu().numpy()
            Xs.append(val)
            Ys.append(key+"@"+str(sentence_index+1))
    return Xs,Ys
#dump vectors both on 2d and 3d
def dump_on_surface(Xs,Ys,dump_3d=True,dump_2d=False):
    pass
    '''
    colors =["#ff0000","#dd0000","#00ffdd","#0000ff","#ffaaaa"] 
    color_mapping = [ colors[ min(len(colors)-1, int(y.split("@")[1])-1) ] for y in Ys]
    #print(color_mapping)
    if dump_3d:    
        pca = PCA(n_components=3)
        X_transformed = pca.fit_transform(np.array(Xs))
        map_on_3d(np.array(X_transformed),Ys,color_mapping=color_mapping,
            use_word_colors=False,fig_size=[10,10])
    if dump_2d:    
        pca = PCA(n_components=2)
        X_transformed = pca.fit_transform(np.array(Xs))
        map_on_2d(np.array(X_transformed),Ys,color_mapping=color_mapping,
            use_word_colors=False,fig_size=[10,10])
    '''
#For 5 sentences, focus on 2 words on dump their projection and see 
#how model generates different vectors for them     
test_sentences = ["i can eat apple","i can eat bread","i can eat book","i can eat newspaper","i can eat apple book"]
indexes = [2,3]
keywords = ["can","eat"]
Xs,Ys = collect_sentence_data(test_sentences,"encoder@Q",indexes,keywords,dump_result=True)
dump_on_surface( Xs,Ys )
Xs,Ys = collect_sentence_data(test_sentences,"encoder@K",indexes,keywords)
dump_on_surface( Xs,Ys )
Xs,Ys = collect_sentence_data(test_sentences,"encoder@V",indexes,keywords)
dump_on_surface( Xs,Ys )
Xs,Ys = collect_sentence_data(test_sentences,"encoder@x1",indexes,keywords)
dump_on_surface( Xs,Ys )
Xs,Ys = collect_sentence_data(test_sentences,"encoder@x4",indexes,keywords)
dump_on_surface( Xs,Ys )
Xs,Ys = collect_sentence_data(test_sentences,"Encoder@src_final",indexes,keywords)
dump_on_surface( Xs,Ys )
#Do the same for decoder side , this time no need to give keywords
test_sentences = ["i can eat apple","i can eat bread","i can eat book","i can eat newspaper","i can eat apple book"]
indexes = [1,3]
keywords = None
Xs,Ys = collect_sentence_data(test_sentences,"decoder_encoder_attention@Q",indexes,keywords,
    use_last_vector=True,dump_result=True)
dump_on_surface( Xs,Ys )
Xs,Ys = collect_sentence_data(test_sentences,"decoder_encoder_attention@K",
    indexes,keywords,use_last_vector=True)
dump_on_surface( Xs,Ys )
Xs,Ys = collect_sentence_data(test_sentences,"decoder_encoder_attention@V",
    indexes,keywords,use_last_vector=True)
dump_on_surface( Xs,Ys )
Xs,Ys = collect_sentence_data(test_sentences,"decoder_encoder_attention@x4",
    indexes,keywords,use_last_vector=True)
dump_on_surface( Xs,Ys )
Xs,Ys = collect_sentence_data(test_sentences,"Decoder@output",indexes,keywords,use_last_vector=True)
dump_on_surface( Xs,Ys )
#how does encoding of EAT changes according to time. #lets do over over simplifaction and map is to 1d
search_sentence = "we can eat apple"
steps_for_sample = get_by_word(search_sentence,["eat","apple"],steps_encoder_byword)
X_s1,Y_s1 = map_word_by_sentence(steps_for_sample,includes="QVKX",ranks=[1,15,29])
X1d,Y1d = filter_by_key(X_s1,Y_s1,["eat"])
pca = PCA(n_components=1)
X_transformed = pca.fit_transform(np.array(X1d))
X_transformed = [x[0] for x in X_transformed]
df = pd.DataFrame()
mapped_vals = get_as_map(X_transformed,Y1d,["eat@Q@1","eat@K@1","eat@V@1","eat@X1@1"])
mapped_vals["qvk"] = mapped_vals["eat@Q@1"] * mapped_vals["eat@K@1"] * mapped_vals["eat@V@1"] 
df["apple_0"] = mapped_vals.keys()
df["apple_0 values"] = mapped_vals.values()
mapped_vals = get_as_map(X_transformed,Y1d,["eat@Q@29","eat@K@29","eat@V@29","eat@X1@29"])
mapped_vals["qvk"] = mapped_vals["eat@Q@29"] * mapped_vals["eat@K@29"] * mapped_vals["eat@V@29"] 
df["apple_29"] = mapped_vals.keys()
df["apple_29 values"] = mapped_vals.values()
#----------------------------
search_sentence = "we can eat bread"
steps_for_sample = get_by_word(search_sentence,["eat","bread"],steps_encoder_byword)
X_s1,Y_s1 = map_word_by_sentence(steps_for_sample,includes="QVKX",ranks=[1,15,29])
X1d,Y1d = filter_by_key(X_s1,Y_s1,["eat"])
pca = PCA(n_components=1)
X_transformed = pca.fit_transform(np.array(X1d))
X_transformed = [x[0] for x in X_transformed]
mapped_vals = get_as_map(X_transformed,Y1d,["eat@Q@1","eat@K@1","eat@V@1","eat@X1@1"])
#print("---mapped_vals",mapped_vals.keys())
mapped_vals["qvk"] = mapped_vals["eat@Q@1"] * mapped_vals["eat@K@1"] * mapped_vals["eat@V@1"] 
df["bread_0"] = mapped_vals.keys()
df["bread_0 values"] = mapped_vals.values()
mapped_vals = get_as_map(X_transformed,Y1d,["eat@Q@29","eat@K@29","eat@V@29","eat@X1@29"])
mapped_vals["qvk"] = mapped_vals["eat@Q@29"] * mapped_vals["eat@K@29"] * mapped_vals["eat@V@29"] 
df["bread_29"] = mapped_vals.keys()
df["bread_29 values"] = mapped_vals.values()
df.head(5)
#at 0 values are -195 and 81 = distance 276 
#at 29           -218 and 43 = distance 261
test_sentences = ["i can eat apple","i can eat bread"]
indexes = [0,1,2,3,4]
keywords = None
Xs,Ys = collect_sentence_data(test_sentences,"Decoder@trg_embedding",
    indexes,keywords,use_last_vector=True,dump_result=True)
dump_on_surface( Xs,Ys )
test_sentences = ["i can eat apple","i can eat bread"]
indexes = [2,3]
keywords = ["can","eat"]
Xs,Ys = collect_sentence_data(test_sentences,"Encoder@src_embedding",
    indexes,keywords,use_last_vector=False,dump_result=True)
dump_on_surface( Xs,Ys )


#check distribution of all source language vectors
encoder_embedding = model.encoder.tok_embedding
Xs_src = []
Ys_src = []
for key_index,key in enumerate(SRC.vocab.stoi.keys()):
    if key_index < 18:
        input = torch.LongTensor([key_index])
        #print("input.device=",input.device) #input.device= cpu
        #print("type(encoder_embedding)=",type(encoder_embedding))
        #print("encoder_embedding(input).device=",encoder_embedding.cpu()(input).device)
        #Xs_src.append( encoder_embedding(input).detach().cpu().numpy().flatten() )
        Xs_src.append( encoder_embedding.cpu()(input).detach().numpy().flatten() )
        Ys_src.append(key+"@"+str(key_index))
dump_on_surface( Xs_src,Ys_src )   
#check distribution of all target language vectors
decoder_embedding = model.decoder.tok_embedding
Xs_trg = []
Ys_trg = []
for key_index,key in enumerate(TRG.vocab.stoi.keys()):
    input = torch.LongTensor([key_index])
    #Xs_trg.append( decoder_embedding(input).detach().cpu().numpy().flatten() )
    Xs_trg.append( decoder_embedding.cpu()(input).detach().numpy().flatten() )
    Ys_trg.append(key+"@"+str(key_index))
dump_on_surface( Xs_trg,Ys_trg )   


#check similarity or source and tagrget vectors
from sklearn.metrics.pairwise import cosine_similarity
datas = []
for trg_index, y_trg in enumerate(Ys_trg[6:]):
    row_data = []
    for src_index,y_src in enumerate(Ys_src[6:]):
        cell_value = cosine_similarity( Xs_trg[6:][trg_index].reshape(1,64), Xs_src[6:][src_index].reshape(1,64) )[0][0]
        row_data.append(cell_value)
    datas.append( row_data )    
draw_df(datas,columns=Ys_src[6:],indexes=Ys_trg[6:])   
#check similarity  target vectors with each other, as expected diagonal is 1
datas = []
for trg_index, y_trg in enumerate(Ys_trg[4:]):
    row_data = []
    for src_index,y_src in enumerate(Ys_trg[4:]):
        cell_value = cosine_similarity( Xs_trg[4:][trg_index].reshape(1,64), Xs_trg[4:][src_index].reshape(1,64) )[0][0]
        row_data.append(cell_value)
    datas.append( row_data )    
draw_df(datas,columns=Ys_trg[4:],indexes=Ys_trg[4:]) 
np.array(datas).shape
#check similarity  source vectors with each other, as expected diagonal is 1
datas = []
for trg_index, y_trg in enumerate(Ys_src[4:]):
    row_data = []
    for src_index,y_src in enumerate(Ys_src[4:]):
        cell_value = cosine_similarity( Xs_src[4:][trg_index].reshape(1,64), Xs_src[4:][src_index].reshape(1,64) )[0][0]
        if trg_index == src_index:
            cell_value = cell_value #0
        row_data.append(cell_value )
    datas.append( row_data )    
draw_df(datas,columns=Ys_src[4:],indexes=Ys_src[4:]) 
np.array(datas).shape
#check similarity of vectors generated by 2 sentences
def dump_vector_relation(sentences,vector_name,is_q=False,dump_vector=False):
    print("sentence  :",sentences[0])
    print("sentence  :",sentences[1])
    print("Vector :",vector_name)
    test_logger1,_,t1 = translate_info(sentences[0],attention_type=1,draw_charts=False,dump_result=False)
    val1 = test_logger1.get_summary([vector_name],show_data=False,summary_count=10)
    test_logger2,_,t2 = translate_info(sentences[1],attention_type=1,draw_charts=False,dump_result=False)
    val2 = test_logger2.get_summary([vector_name],show_data=False,summary_count=10)
    data1 = val1[len(val1)-1][0][0]
    data2 = val2[len(val1)-1][0][0]
    if vector_name.find("x4") > 0:
        data1 = val1[len(val1)-1][0]
        data2 = val2[len(val2)-1][0]
    if dump_vector:
        print("data1")
        print(data1)
        print("data2")
        print(data2)
    data_matrix = []
    #print("type(data1)=",type(data1))
    #print("data1=",data1)
    for i in range(data1.shape[0] ):
        sub_data = []
        for j in range(data2.shape[0] ):
            shape_new = data1[i].shape[0]
            #sub_data.append( cosine_similarity(data1[i].reshape(1,shape_new), data2[j].reshape(1,shape_new))[0][0] )
            sub_data.append( cosine_similarity(data1[i].reshape(1,shape_new).cpu(), data2[j].reshape(1,shape_new).cpu())[0][0] )
        data_matrix.append( sub_data )
    columns = ["sos"] + sentences[1].split(" ") + ["eos"]
    print("columns1=",columns)
    indexes = ["sos"] + sentences[0].split(" ") + ["eos"]
    print("indexes1=",indexes)
    
    if is_q :
        print("t2[:-1]=",t2[:-1])
        columns = ["sos"] + t2[:-1] + ["eos"]  #deleted by ly
        print("columns2=",columns)
        indexes = ["sos"] + t1[:-1] + ["eos"]  #deleted by ly
        columns = [ columns[i]+"->"+columns[i+1] for i in range(len(columns)-1) ]
        indexes = [ indexes[i]+"->"+indexes[i+1] for i in range(len(indexes)-1) ]
    print("data_matrix=",data_matrix)
    print("columns3=",columns)
    #draw_df(data_matrix,columns=columns,indexes=indexes)  #deleted by ly
    draw_df(data_matrix,columns=columns,indexes=indexes)  #deleted by ly
    return 0
sentences = ["we can eat apple","we can read bread"]
dump_vector_relation(sentences,"decoder_encoder_attention@x4",True)
dump_vector_relation(sentences,"decoder_encoder_attention@Q",True)
dump_vector_relation(sentences,"decoder_encoder_attention@K")
dump_vector_relation(sentences,"decoder_encoder_attention@V")
dump_vector_relation(sentences,"decoder_encoder_attention@attention",True)
#Q depends on output itself, it is query so always same values
#K  features learned, how output aligns with input
#V  QxK , multiplication of query x features,what was learned
test_logger1,_,t1 = translate_info(sentences[0],attention_type=1,draw_charts=False,dump_result=False)
val1 = test_logger1.get_summary(["decoder_encoder_attention@attention"],show_data=False,summary_count=10)
test_logger2,_,t2 = translate_info(sentences[1],attention_type=1,draw_charts=False,dump_result=False)
val2 = test_logger2.get_summary(["decoder_encoder_attention@attention"],show_data=False,summary_count=10)
#same sentence all diagonals must be same
sentences = ["we can eat apple","we can eat apple"]
dump_vector_relation(sentences,"decoder_encoder_attention@Q",True)
dump_vector_relation(sentences,"decoder_encoder_attention@K")
dump_vector_relation(sentences,"decoder_encoder_attention@V")
dump_vector_relation(sentences,"decoder_encoder_attention@x4",True)
#order and  object is different
sentences = ["we can eat apple","eat bread can we"]
dump_vector_relation(sentences,"decoder_encoder_attention@Q",True)
dump_vector_relation(sentences,"decoder_encoder_attention@K")
dump_vector_relation(sentences,"decoder_encoder_attention@V")
dump_vector_relation(sentences,"decoder_encoder_attention@x4",True)
dump_vector_relation(sentences,"decoder_encoder_attention@attention",True,dump_vector=False)
#same words in different order
sentences = ["we can eat apple","apple eat can we"]
dump_vector_relation(sentences,"decoder_encoder_attention@Q",True)
dump_vector_relation(sentences,"decoder_encoder_attention@K")
dump_vector_relation(sentences,"decoder_encoder_attention@V")
dump_vector_relation(sentences,"decoder_encoder_attention@x4",True)
dump_vector_relation(sentences,"decoder_encoder_attention@attention",True,dump_vector=True)
#do the vector relation logic for encoder layers
def dump_vector_relation_encoder(sentences,vector_name,is_q=False):
    print("Vector :",vector_name)
    test_logger1,_,t1 = translate_info(sentences[0],attention_type=1,draw_charts=False,dump_result=False)
    val1 = test_logger1.get_summary([vector_name],show_data=False,summary_count=10)
    test_logger2,_,t2 = translate_info(sentences[1],attention_type=1,draw_charts=False,dump_result=False)
    val2 = test_logger2.get_summary([vector_name],show_data=False,summary_count=10)
    data1 = val1[len(val1)-1][0][0]
    data2 = val2[len(val1)-1][0][0]
    data_matrix = []
    for i in range(data1.shape[0] ):
        sub_data = []
        for j in range(data2.shape[0] ):
            #sub_data.append( cosine_similarity(data1[i].reshape(1,64), data2[j].reshape(1,64))[0][0] )
            sub_data.append( cosine_similarity(data1[i].reshape(1,64).cpu(), data2[j].reshape(1,64).cpu())[0][0] )
        data_matrix.append( sub_data )
    print("data1",data1.shape)
    columns = ["sos"] + sentences[1].split(" ") + ["eos"]
    indexes = ["sos"] + sentences[0].split(" ") + ["eos"]
    if is_q :
        columns =  ["sos"]+sentences[1].split(" ") + ["eos"]
        indexes =  ["sos"]+sentences[0].split(" ") + ["eos"]
    draw_df(data_matrix,columns=columns,indexes=indexes) 
#sentences = ["we can eat apple","i can read book"]
sentences = ["we can eat apple","we can eat bread"]
dump_vector_relation_encoder(sentences,"encoder@Q",True)
dump_vector_relation_encoder(sentences,"encoder@K")
dump_vector_relation_encoder(sentences,"encoder@V")
#test_logger.get_default_summary(show_data=True)
for index in range(20,22):
    test_logger,all_outputs,_ = translate_info(tabular_set[index].src,draw_charts=False,attention_type=1)
    encoded_attention = test_logger.get_summary(["EncoderLayer@sattention"],show_data=False)
    src_masks = test_logger.get_summary(["EncoderLayer@src_mask"],show_data=False,summary_count=10)
    #print("EncoderLayersrc_masks",src_masks)
    decodersrc_masks = test_logger.get_summary(["DecoderLayer@src_mask"],
        show_data=False,summary_count=10)
    #print("decodersrc_masks",decodersrc_masks)
    decoder_attention = test_logger.get_summary(["Decoder@attention"],show_data=False,summary_count=10)
    #print("decoder_attention",decoder_attention[0].shape)
    #print("decoder_attention",decoder_attention[0])
    decodertrg_masks = test_logger.get_summary(["DecoderLayer@trg_mask"],show_data=False,summary_count=10)
    #for index2,dm in enumerate(decodertrg_masks):
    #    print("decodertrg_masks",index2)
    #    print(dm)
    output_words = ["sos"]+tabular_set[index].src  +["eos"]
    for head_index in range(encoded_attention[0].shape[1]):
        all_outputs0 = encoded_attention[0][:,head_index,:,:]
        draw_df(all_outputs0.squeeze().cpu().numpy(),output_words,output_words)
    ### OPEN THIS COMMENT FOR MULTI HEAD all_outputs1 = encoded_attention[0][:,1,:,:]
    ###draw_df(all_outputs1.squeeze().cpu().numpy(),output_words,output_words)
#i want to read newspaper  diagonal
#i can eat bread  diagonal
#we can eat bread diagonal
#we can read book , very diagonal
#we can read newspaper , very diagonal
#i eat bread , very diagonal
#when want to comes into ,patterns change
