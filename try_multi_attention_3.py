import warnings
warnings.filterwarnings('ignore')

import argparse
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation
from torch import nn
import torch.nn.functional as F #jz

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import pandas as pd
import numpy as np
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from tqdm import tqdm
# from tqdm.notebook import tqdm_notebook as tqdm

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from transformers import AdamW, get_linear_schedule_with_warmup, Adafactor

from table_bert import TableBertModel
from table_bert import Table, Column

from torch.optim.lr_scheduler import ReduceLROnPlateau

flag = 2 #jz
learningrate=1e-5
method = 'relu'
max_len = 15 
num_epoch=1000

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WikiDataset(Dataset):
    def __init__(self, path, model):
        self.path = path
        self.model = model

        self.data = pd.read_json(self.path)

        self.data['title'] = self.data['title'].fillna('unknown')
        
        self.data['header_label'] = self.data['header'].apply(lambda x: [0]*len(x)) #jz
        self.data['select_column'] = self.data['sql'].apply(lambda x: x['sel'])  #jz
        self.data['length'] = self.data['header'].apply(lambda x: len(x))  #jz
    
        for i in range(len(self.data)):  
            self.data['header_label'][i][self.data['select_column'][i]] = 1  #jz
        
        #jz: drop length > max_len
        self.data = self.data[self.data['length'] <= max_len]    
        self.data = self.data.reset_index(drop=True)
        
        #jz: padding the header_label with 2, and padding the table with empty string, padding the table header with 'zzz'
        self.data['header_label'] = self.data['header_label'].apply(lambda x: x+[2]*(max_len-len(x)))
        
        for i in range(len(self.data)):
            rows = self.data['rows'][i]
            length = self.data['length'][i]
    
            self.data['header'][i] = self.data['header'][i]+[['padding', 'text', 'zzz']]*(max_len-length)
            for j in range(len(rows)):
                rows[j] = rows[j] + ['']*(max_len-length)
            
            self.data['rows'][i] = rows
        
    

        self.tabs = []
        self.context = []
        #self.answers = []
        self.label = []  #jz

        self._build()

    def __len__(self):
        return len(self.context)


    def _build(self):
        for idx in tqdm(range(len(self.data))):
            qs = self.data.loc[idx, 'context']
            #ans = self.data.loc[idx, 'answer']
            heads = self.data.loc[idx, 'header']
            tit = self.data.loc[idx, 'title']
            rs = self.data.loc[idx, 'rows']
            label = self.data.loc[idx, 'select_column'] #jz: if use multi-class classification, here is a number in [0,14].
            # print(label)

            col = [Column(z[0], z[1], sample_value=z[2]) for z in heads]

            table = Table(
                id=tit,
                header=col,
                data=rs
            ).tokenize(self.model.tokenizer)

            self.tabs.append(table)
            self.context.append(self.model.tokenizer.tokenize(qs))
            #self.answers.append(self.model.tokenizer.convert_tokens_to_ids(self.model.tokenizer.tokenize(str(ans[0])))[0])
           
            self.label.append(label)  #jz
    
    def __getitem__(self, index):
        tabi = self.tabs[index]
        conti = self.context[index]
        #ansi = self.answers[index]
        la = self.label[index]  #jz
        #return {"table": tabi, "context": conti, "answer": ansi, "label": la}  
        return {"table": tabi, "context": conti, "label": la}  #jz


def get_dataset(path, model):
    return WikiDataset(path=path, model=model)

def collate_fn(batch):
    
    batch_0 = [batch[i]['table'] for i in range(len(batch))]
    batch_1 = [batch[i]['context'] for i in range(len(batch))]
    
    #batch_3 = torch.tensor([batch[i]['label'] for i in range(len(batch))])
    batch_new = [batch[i]['label'] for i in range(len(batch))]
    return batch_0, batch_1, batch_new


class TaBERTTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(TaBERTTuner, self).__init__()
        self.hparams = hparams

        self.model = TableBertModel.from_pretrained('tabert_base_k1/model.bin') #jz
        
        #for multi-classification, col-encoding is (bs, 15, 768)->(bs, 768)->(bs, 500)->(bs, 15), label is (bs, 1)
        #first layer #jz
        self.l1 = nn.Linear(768, 500)
        self.l1_cat = nn.Linear(1536, 500)

        #second layer
        self.l2 = nn.Linear(500, max_len)  #jz
      
        #softmax
        self.sm = nn.Softmax(dim=1)
        
        #loss
        self.l = nn.CrossEntropyLoss() 
        
        #attention: apply to column_encoding and context_encoding seperately. 
        #weighted sum of column_encoding and weighted sum of context_encoding. 
        
        #col_weight: (bs, 15, 768)->(bs, 15, 768)->(bs, 15, 1)->(bs, 15), weight*column_encoding: (bs,1,15)*(bs,15,768)->(bs,1,768)
        self.lin_bias = nn.Linear(768, 768)
        self.att_weight = nn.Parameter(torch.rand(768,1))
        self.sm_att = nn.Softmax(dim=1)
        
        #ctx_weight: (bs, m, 768)->(bs, m, 768)->(bs, m, 1)->(bs, m), weight*context_encoding: (bs,1,m)*(bs,m,768)->(bs,1,768)
        self.lin_bias_ctx = nn.Linear(768, 768)
        self.att_weight_ctx = nn.Parameter(torch.rand(768,1))
        self.sm_att_ctx = nn.Softmax(dim=1)

    def forward(self, context_list, table_list):
        context_encoding, column_encoding, info_dict = self.model.encode(contexts=context_list, tables=table_list)
        
        #print('\nContext size: {}, Column size: {}\n'.format(context_encoding.size(), column_encoding.size()))
        
        bs = context_encoding.size()[0]
        
        #apply attention to context_encoding
        ctx_out = F.tanh(self.lin_bias_ctx(context_encoding)) #(bs, m, 768)

        attn_out_ctx = torch.bmm(ctx_out,self.att_weight_ctx.repeat(bs,1,1)).squeeze(dim=2) #(bs, m)
        
        sm_out2_ctx = self.sm_att_ctx(attn_out_ctx) #(bs, m)
        
        ctx_enc_sum = torch.bmm(sm_out2_ctx.unsqueeze(dim=1),context_encoding).squeeze(dim=1) #(bs, 768)
        
        #apply attention to column_encoding
        col_out = F.tanh(self.lin_bias(column_encoding)) #(bs, 15, 768)

        attn_out = torch.bmm(col_out,self.att_weight.repeat(bs,1,1)).squeeze(dim=2) #(bs, 15)
        
        sm_out2 = self.sm_att(attn_out) #(bs, 15)
        
        col_enc_sum = torch.bmm(sm_out2.unsqueeze(dim=1),column_encoding).squeeze(dim=1) #(bs, 768)
      
        #print('\nContext size: {}, Column size: {}\n'.format(ctx_enc_sum.size(), col_enc_sum.size()))
        
        if flag == 0: #ignore question embedding
            out = self.l1(col_enc_sum)  #(bs, 500)
            if method == 'relu':
                out = F.relu(out)
            if method == 'tanh':
                out = F.tanh(out)
            out = self.l2(out) #(bs, 15)
            out = self.sm(out) #(bs, 15)
        
        if flag == 1: #add question embedding and column embedding
            ctx_col_sum = ctx_enc_sum + col_enc_sum  #(bs, 768)
            out = self.l1(ctx_col_sum) 
            if method == 'relu':
                out = F.relu(out)
            if method == 'tanh':
                out = F.tanh(out)
            out = self.l2(out)
            out = self.sm(out)
        
        if flag == 2: #concating question embedding and column embedding
            concat = torch.cat([ctx_enc_sum, col_enc_sum], dim=1) #(bs, 1536)
            out = self.l1_cat(concat)
            if method == 'relu':
                out = F.relu(out)
            if method == 'tanh':
                out = F.tanh(out)
            out = self.l2(out)
            out = self.sm(out)

        return out

    def _step(self, batch):
        #if torch.cuda.is_available():
        tbl, ctx , label = batch[0], batch[1],torch.tensor(batch[2]).to('cuda')
        #print('label',label)
        
        outputs = self(ctx, tbl)
        #print('output', outputs)
        
        #print('\nOutput size: {}, Label size: {}\n'.format(outputs.size(), label.size()))
        
        loss = self.l(outputs, label)
        
        #print('\nLoss size: {}\n'.format(loss.size()))

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, "progress_bar": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        #print(outputs)
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs, "progress_bar": tensorboard_logs}

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },

            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,

            }, ]

        #optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        optimizer = optim.SGD(optimizer_grouped_parameters, lr=self.hparams.learning_rate, nesterov=True, momentum=self.hparams.momentum)
        self.opt = optimizer

        scheduler = [
            {'scheduler': ReduceLROnPlateau(optimizer, mode="min", min_lr=7.5e-5, patience=5, verbose=True),
             # might need to change here
             'monitor': "val_loss",  # Default: val_loss
             'interval': 'epoch',
             'frequency': 1
             }
        ]

        self.lr_scheduler = scheduler

        return [optimizer], scheduler

        #return [optimizer]

    #def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None, on_tpu=False,
                       #using_native_amp=False, using_lbfgs=False):
        #optimizer.step()
        #optimizer.zero_grad()
        #self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(path=self.hparams.train_data, model=self.model)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True,
                                num_workers=4, collate_fn=collate_fn)
        #t_total = (
                #(len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
                #// self.hparams.gradient_accumulation_steps * float(self.hparams.num_train_epochs)

        #)
        #scheduler = get_linear_schedule_with_warmup(
            #self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        #)
        #self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset=get_dataset(path=self.hparams.dev_data, model = self.model)
        return  DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4,collate_fn=collate_fn)

    def test_dataloader(self):
        test_dataset=get_dataset(path=self.hparams.test_data, model = self.model)
        return  DataLoader(test_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4,collate_fn=collate_fn)

if __name__=='__main__':
    set_seed(42)
    args_dict = dict(
        train_data="train_tabert.json",
        dev_data="dev_tabert.json",
        test_data="test_tabert.json",
        output_dir="./",
        # leanring rate
        learning_rate=learningrate,
        momentum = 0.99, 
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        train_batch_size=16,
        eval_batch_size=16,
        # change epoch here
        num_train_epochs=num_epoch,
        gradient_accumulation_steps=16,
        n_gpu=1,
        #early_stop_callback=False,
        fp_16=False,
        opt_level='O1',
        max_grad_norm=1.0,
        seed=42,
    )
    args = argparse.Namespace(**args_dict)
    print(args_dict)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=args.output_dir, prefix="attention3_relu_checkpoint", monitor="val_loss", mode="min", save_top_k=1)
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=3,
        verbose=True,
        mode='min'
    )
    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=1,
        #pus=0,
        max_epochs=args.num_train_epochs,
        # early_stop_callback=False,
        precision=32,
        amp_level=args.opt_level,
        gradient_clip_val=args.max_grad_norm,
        checkpoint_callback=checkpoint_callback,
        callbacks=[early_stop_callback]
    )
    model = TaBERTTuner(args)
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)

