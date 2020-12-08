import argparse
import random
import os
import pickle
import glob
import json
import time
import logging
import re
from itertools import chain
from string import punctuation

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pandas as pd
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook as tqdm

from transformers import AdamW, get_linear_schedule_with_warmup
from table_bert import TableBertModel
from table_bert import Table, Column

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class WikiDataset():
    def __init__(self, path, voc_location, model, minimum_count, max_num):
        # the initalization will end up with four parts: tabs, context, answers and target_voc
        self.path = path
        self.voc_location = voc_location
        self.model = model
        self.minimum_count=minimum_count
        self.max_num=max_num

        self.data = pd.read_json(self.path)
        self.data['title'] = self.data['title'].fillna('unknown')
        lens = self.data['answer'].apply(lambda x: len(model.tokenizer.tokenize(str(x[0]))))
        self.data = self.data.reset_index(drop=True)
        # print(self.data.shape)

        self.tabs = []
        self.context = []
        self.answers = []

        self._build()
        print('length of table, context, ans', len(self.tabs), len(self.context), len(self.answers))

    def _build(self):
        for idx in tqdm(range(len(self.data))):
            qs = self.data.loc[idx, 'context']
            ans = self.data.loc[idx, 'answer']
            heads = self.data.loc[idx, 'header']
            tit = self.data.loc[idx, 'title']
            rs = self.data.loc[idx, 'rows']

            col = [Column(z[0], z[1], sample_value=z[2]) for z in heads]
            table = Table(
                id=tit,
                header=col,
                data=rs
            ).tokenize(self.model.tokenizer)
            self.tabs.append(table)

            self.context.append(self.model.tokenizer.tokenize(qs))
#             print('ans:',ans)
            ans_tokenized = self.model.tokenizer.tokenize(str(ans[0]))
            encoded_dict = tokenizer.encode_plus(
                        ans_tokenized,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        truncation=True,
                        max_length = 35,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
#             print('answers',ans[0])
#             print('answers',ans_tokenized)
#             print('tockenized answers:',encoded_dict['input_ids'])
            self.answers.append(encoded_dict['input_ids'])

    def __len__(self):
        return len(self.context)
    def _get_voc(self):
        return self.target_voc
    def __getitem__(self, index):
        if len(self.tabs)==len(self.context)==len(self.answers):
            try:
                tabi = self.tabs[index]
                conti = self.context[index]
                ansi = self.answers[index]
                return {"table": tabi, "context": conti, "answer": ansi}
            except IndexError:
                print('Check the length and index')
                print('length of table, context, ans', len(self.tabs), len(self.context),len(self.answers))
                print('The current index is:', index)
        else:
            print('The lengths are not matching')
            print('length of table, context, ans', len(self.tabs), len(self.context), len(self.answers))
            print('The current index is:', index)

def get_dataset(path, voc_location, model, minimum_count=1,max_num=35):
    return WikiDataset(path=path, voc_location=voc_location,model=model,\
                       minimum_count=minimum_count,max_num=max_num)

def collate_fn(batch):
    return [batch[i]['table'] for i in range(len(batch))], [batch[i]['context'] for i in range(len(batch))], torch.stack([batch[i]['answer'] for i in range(len(batch))])

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim]
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        # print('attention shape:------------------------------------------------------------------------')
        # print('batch_size,src_len:',batch_size,src_len)

        # repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1) # hidden = [batch size, src len, dec hid dim]
        encoder_outputs = encoder_outputs.permute(1, 0, 2) # encoder_outputs = [batch size, src len, enc hid dim]
        # print('attention hidden,encoder_outputs shape:',hidden.shape,encoder_outputs.shape)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy = [batch size, src len, dec hid dim]
        # print('energy shape',energy.shape)
        attention = self.v(energy).squeeze(2) # attention= [batch size, src len]
        # print('attention shape', attention.shape)
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx = PAD_IDX)
        self.rnn = nn.GRU(enc_hid_dim + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear(enc_hid_dim + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):  # input is the answers to the table
        # input = [batch size]
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim]
        # print('decoder shape:------------------------------------------------------------------------')
        input = input.unsqueeze(0) #input = [1, batch size] [1,12]
        # print('input shape:',input.shape)
        # print('input:', input)
        # print('hidden shape:', hidden.shape)
        # print('encoder_outputs shape:', encoder_outputs.shape)
        embedded = self.dropout(self.embedding(input))  # embedded = [1, batch size, emb dim]
        # print('embedded shape:',embedded.shape)
        a = self.attention(hidden, encoder_outputs)  # a = [batch size, src len]
        a = a.unsqueeze(1)  # a = [batch size, 1, src len]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # encoder_outputs = [batch size,src len,enc hid dim]

        weighted = torch.bmm(a, encoder_outputs)  # weighted = [batch size, 1, enc hid dim]
        weighted = weighted.permute(1, 0, 2)  # weighted = [1, batch size, enc hid dim]
        # print('weighted shape:',weighted.shape)
        rnn_input = torch.cat((embedded, weighted), dim=2)  # rnn_input = [1, batch size, (enc hid dim) + emb dim] [1,12,768+256]
        # print('rnn input shape:',rnn_input.shape)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        # print('rnn output shape',output.shape,hidden.shape)
        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        assert (output == hidden).all()  # this also means that output == hidden

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        # prediction = [batch size, output dim]
        return prediction, hidden.squeeze(0)

class TaBERTTuner(pl.LightningModule):
    def __init__(self, hparams, decoder,enc_hid_dim, dec_hid_dim):
        super(TaBERTTuner, self).__init__()
        self.hparams = hparams
        self.encoder = TableBertModel.from_pretrained('bert-base-uncased')
        self.hidden = nn.Linear(enc_hid_dim, dec_hid_dim)
        self.decoder = decoder

    def forward(self, context_list, table_list, answer_list,teacher_forcing_ratio = 0.5):
        context_encoding, column_encoding, info_dict = self.encoder.encode(contexts=context_list, tables=table_list)
        # print('TaBERTTuner forward:--------------------------------------------------------------------------------------------------')
        # print('Encoder:--------------------------------------------------------------------------------------------------')
        # print('context_encoding, column_encoding, answer_list shape:',context_encoding.shape,column_encoding.shape, answer_list.shape)
        # context_encoding: [batch_size, token_size, 768] [12,17,768]
        # column_encoding: [batch_size, number of columns, 768] [12,6,768]
        # answer_list: [batch_size, 36] max_len is 35 and adds one for eos

        # ctx_enc_sum = torch.sum(context_encoding, axis=1) #[batch_size,768]
        # col_enc_sum = torch.sum(column_encoding, axis=1) #[batch_size,768]
        # print('ctx_enc_sum, col_enc_sum shape:',ctx_enc_sum.shape, col_enc_sum.shape)

        src = torch.cat([context_encoding, column_encoding], dim=1)  # [batch_size,scr_len,768*2] [12,123,768]
        encoder_outputs = src.permute(1,0,2)  # [src_len,batch_size,enc_hid_dim] [123,12,768]
        batch_size = encoder_outputs.shape[1]
        # print('scr shape:', src.shape)
        # print('encoder_outputs shape:', encoder_outputs.shape)
        # print('batch size:',batch_size)

        hidden = torch.tanh(self.hidden(torch.sum(encoder_outputs, axis=0)))  # [batch_size=12,dec_hid_dim=512]
        # print('encoder hidden output shape', hidden.shape)

        trg_vocab_size = self.decoder.output_dim
        # print(answer_list.shape)
        # answer list torch.Size([12, 1, 64]) => [12,64]
        trg = torch.squeeze(answer_list,dim=1).permute(1,0)  # [64,12] [trg_len, batch_size]
        trg_len = trg.shape[0]
        # print('trg_len:',trg_len)
        # print('trg shape:', trg.shape)

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size)
        # print('storage output shape:', outputs.shape)

        input = trg[0, :]
        # print('input shape:',input.shape)
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            # print('decoder_output,decoder_hidden',output.shape,hidden.shape)
            # place predictions in a tensor holding predictions for each token
            outputs[t] = output
            top1 = output.argmax(1)
            teacher_force = random.random() < teacher_forcing_ratio
            input = trg[t] if teacher_force else top1
        return outputs

    def _step(self, batch,teacher_forcing_ratio):
        tbl, ctx, ans = batch[0], batch[1], torch.tensor(batch[2])
        # print('step ----------------------------------------------------------------------------------------------------')
        outputs = self(ctx, tbl, ans,teacher_forcing_ratio =teacher_forcing_ratio) # output = [trg len, batch size, output dim]
        trg = ans # trg = [trg len, batch size]
        # print('outputs from step:', outputs.shape)
        # print('trg from step:', trg.shape)

        output_dim = outputs.shape[-1]
        outputs = outputs[0:].view(-1, output_dim) # output = [(trg len-0 ) * batch size, output dim]
        trg = trg[0:].view(-1)  # trg = [(trg len - 0) * batch size]
        # print('outputs from step:', outputs.shape)
        # print('trg from step:', trg.shape)

        criterion = nn.CrossEntropyLoss(ignore_index=0)
        outputs = outputs.to(device=device)
        trg = trg.to(device=device)
        loss = criterion(outputs, trg)
        return loss

    def training_step(self, batch, batch_idx,optimizer_idx):
        loss = self._step(batch,teacher_forcing_ratio=0.5)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
#         print('train epoch end--------------------------------------------------------')
#         print(outputs)
#         print(outputs[0])
#         print(outputs[1])
        avg_train_loss_encoder = torch.stack([x['loss'] for x in outputs[0]]).mean()
        avg_train_loss_decoder = torch.stack([x['loss'] for x in outputs[1]]).mean()
        avg_train_loss = torch.stack((avg_train_loss_encoder,avg_train_loss_decoder)).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, "progress_bar": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch,teacher_forcing_ratio=0) #turn off teacher forcing ratio for the validation steps
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_val_loss}
        return {"avg_val_loss": avg_val_loss, "log": tensorboard_logs, "progress_bar": tensorboard_logs}

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.encoder.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.encoder.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }, ]
        encoder_optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.lr,
                                  eps=self.hparams.adam_epsilon)
        decoder_optimizer = torch.optim.SGD(self.decoder.parameters(), lr=self.hparams.lr, nesterov=True,
                                 momentum = self.hparams.momentum)
        optimizers = [encoder_optimizer,decoder_optimizer]
        schedulers = [
            {'scheduler': ReduceLROnPlateau(encoder_optimizer, mode="min", min_lr= self.hparams.minlr, patience=self.hparams.patience, verbose=True),
                # might need to change here
             'monitor': "avg_train_loss",  # Default: val_loss
             'interval': 'epoch',
             'frequency': 1
                },
            {'scheduler': ReduceLROnPlateau(decoder_optimizer, mode="min", min_lr= self.hparams.minlr, patience=self.hparams.patience, verbose=True),
                # might need to change here
             'monitor': "avg_train_loss" ,  # Default: val_loss
             'interval': 'epoch',
             'frequency': 1
                }
        ]
        return optimizers, schedulers

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss)}
        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(path=self.hparams.train_data, voc_location=self.hparams.voc_location,
                                    model=self.encoder,minimum_count=self.hparams.minimum_count,
                                    max_num=self.hparams.max_num)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size,
                                drop_last=True, shuffle=True,num_workers=4, collate_fn=collate_fn)
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(path=self.hparams.dev_data, voc_location=self.hparams.voc_location,
                                  model=self.encoder)
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4,
                          collate_fn=collate_fn)


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='self attention model')
    parser.add_argument('-td')
    parser.add_argument('-vd')
    parser.add_argument('-lr')
    parser.add_argument('-minlr')
    parser.add_argument('-p')
    parser.add_argument('-gpu')

    args = parser.parse_args()
    train_data=args.td
    val_data = args.vd
    lr = float(args.lr)
    minlr = float(args.minlr)
    patience = int(args.p)
    gpu = int(args.gpu)

    # train_data = "./data/train_tabert_0.01.json"
    # val_data = "./data/dev_tabert_0.01.json"
    # lr = float(2.5e-4)
    # minlr = float(1e-5)
    # patience = int(0)
    # gpu = int(0)

    set_seed(42)

    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    PAD_IDX = 0
    UNK_IDX = 1
    SOS_IDX = 2
    EOS_IDX = 3
    SEP_IDX = 4
    PAD_TOKEN = '<pad>'
    UNK_TOKEN = '<unk>'
    SOS_TOKEN = '<sos>'
    EOS_TOKEN = '<eos>'
    SEP_TOKEN = '<sep>'  # separates utterances in the dialogue history

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device is:', device)

    args_dict = dict(
        train_data=train_data,
        #"./data/train_tabert.json",
        dev_data=val_data,
        #"./data/dev_tabert.json",
        voc_location='./voc',
        output_dir="./check_point",
        minimum_count=1,
        max_num=35,
        lr=lr,
        minlr=minlr,
        patience=patience,
        #7.5e-4,
        dampening=0.9,  # increase 0.9-0.99. learning rate decrease by factor of 10
        momentum=0.99,
        weight_decay=1e-2,
        adam_epsilon=1e-8,
        gradient_clip_val=0.3,
        warmup_steps=100,
        gradient_accumulation_steps=16,
        train_batch_size=16,
        eval_batch_size=12,
        num_train_epochs=2000,
        n_gpu=gpu,
        fp_16=False,  # fp_16 true will end up shorter trainning time. 32 is default
        opt_level='O1',  # pure or mixed precision
        seed=42
    )
    args = argparse.Namespace(**args_dict)
    print(args_dict)

    checkpoint_callback = ModelCheckpoint(
        filepath=args.output_dir,
        prefix= str(lr) +'_checkpoint_attn-{epoch:02d}',
        monitor="val_loss", mode="min", save_top_k=5)

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=3,
        verbose=True,
        mode='min'
    )
    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        max_epochs=args.num_train_epochs,
        amp_level=args.opt_level,
        gradient_clip_val=args.gradient_clip_val,
        auto_lr_find=True,
        precision=32,
        checkpoint_callback=checkpoint_callback,
        check_val_every_n_epoch=2,
        callbacks=[early_stop_callback],
    )

    OUTPUT_DIM = 32000

    ENC_HID_DIM = 768

    DEC_EMB_DIM = 516
    DEC_HID_DIM = 768
    DEC_DROPOUT = 0.5

    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
    model = TaBERTTuner(args, enc_hid_dim=ENC_HID_DIM, dec_hid_dim=DEC_HID_DIM, decoder=dec)

    trainer = pl.Trainer(**train_params)
    torch.cuda.empty_cache()
    trainer.fit(model)