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

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import spacy
from torchtext.data import Field, BucketIterator

import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
# from tqdm.notebook import tqdm_notebook as tqdm

from transformers import AdamW, get_linear_schedule_with_warmup

from table_bert import TableBertModel
from table_bert import Table, Column

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class Lang:
    def __init__(self, minimum_count=1):
        self.word2index = {}
        self.word2count = {}

        self.index2word = [None] * 4
        self.index2word[SOS_IDX] = SOS_TOKEN
        self.index2word[EOS_IDX] = EOS_TOKEN
        self.index2word[UNK_IDX] = UNK_TOKEN
        self.index2word[PAD_IDX] = PAD_TOKEN

        self.word2count[SOS_TOKEN] = 100;
        self.word2count[EOS_TOKEN] = 100;
        self.word2count[UNK_TOKEN] = 100;
        self.word2count[PAD_TOKEN] = 100;

        self.word2index[SOS_TOKEN] = SOS_IDX;
        self.word2index[EOS_TOKEN] = EOS_IDX;
        self.word2index[UNK_TOKEN] = UNK_IDX;
        self.word2index[PAD_TOKEN] = PAD_IDX;
        self.n_words = 4  # Count SOS and EOS

        self.minimum_count = minimum_count;

    def add_ans(self, ans):
        for word in ans:
            self.addWord(word.lower())

    def addWord(self, word):
        if word not in self.word2count.keys():
            self.word2count[word] = 1
        else:
            self.word2count[word] += 1
        if self.word2count[word] >= self.minimum_count:
            if word not in self.index2word:
                word = str(word);
                self.word2index[word] = self.n_words
                self.index2word.append(word)
                self.n_words += 1

    def vec2txt(self, list_idx):
        word_list = []
        if type(list_idx) == list:
            for i in list_idx:
                if i not in [EOS_IDX, SOS_IDX, PAD_IDX]:
                    word_list.append(self.index2word[i])
        else:
            for i in list_idx:
                if i.item() not in [EOS_IDX, SOS_IDX, PAD_IDX]:
                    word_list.append(self.index2word[i.item()])
        return (' ').join(word_list)

    def txt2vec(self, ans):
        token_list = ans;
        index_list = [self.word2index[token] if token in self.word2index else UNK_IDX for token in token_list]
        return torch.from_numpy(np.array(index_list)).to(device)

class voc(Dataset):
    def __init__(self, df, minimum_count = 1, max_num = 35):
        self.minimum_count = minimum_count;
        self.max_num = max_num;
        self.main_df,self.voc = self.load_voc(df,minimum_count = 1);
    def __len__(self):
        return len(self.main_df) if self.max_num is None else self.max_num
    def __getitem__(self, idx):
        return_list = [self.main_df.iloc[idx]['target_indized'], self.main_df.iloc[idx]['target_len'] ]
        return return_list
    def load_voc(self,df,minimum_count = 1):
        target_voc_obj = Lang(minimum_count = 1);
        for ans in df:
            target_voc_obj.add_ans(ans)
        indices_data = []
        for ans in df:
            index_list = [target_voc_obj.word2index[token] if token in target_voc_obj.word2index else UNK_IDX for token in ans]
            if len(index_list)<=self.max_num:
                index_list = index_list + [PAD_IDX]*(self.max_num-len(index_list))
            else:
                index_list = index_list[:self.max_num]
            index_list.append(EOS_IDX)
            indices_data.append(index_list)
        main_df = pd.DataFrame();
        main_df['target_tokenized'] = df;
        main_df['target_indized'] = indices_data;
        main_df['target_len'] = main_df['target_tokenized'].apply(lambda x: len(x)+1) #+1 for EOS
        main_df =  main_df[main_df['target_len'] >=2]
        return main_df,target_voc_obj

class WikiDataset(Dataset):
    def __init__(self, path, model):
        self.path = path
        self.model = model

        self.data = pd.read_json(self.path)
        self.data['title'] = self.data['title'].fillna('unknown')
        lens = self.data['answer'].apply(lambda x: len(model.tokenizer.tokenize(str(x[0]))))
        self.data = self.data.reset_index(drop=True)
        # print(self.data.shape)

        self.tabs = []
        self.context = []
        self.answers = []

        self._build()
        self.voc = self._answers_idx()

    def __len__(self):
        return len(self.context)

    def __getitem__(self, index):
        tabi = self.tabs[index]
        conti = self.context[index]
        ansi = self.answers[index]
        return {"table": tabi, "context": conti, "answer": ansi}

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
            self.answers.append(self.model.tokenizer.tokenize(str(ans[0])))

    def _answers_idx(self):
        target_voc = voc(self.answers, minimum_count=1,max_num=35)
        self.answers = target_voc.main_df.target_indized.tolist()
        return target_voc

def get_dataset(path, model):
    return WikiDataset(path=path, model=model)

def collate_fn(batch):
    return [batch[i]['table'] for i in range(len(batch))], [batch[i]['context'] for i in range(len(batch))], torch.tensor([batch[i]['answer'] for i in range(len(batch))])

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
        self.embedding = nn.Embedding(output_dim, emb_dim)
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
        self.model = TableBertModel.from_pretrained('bert-base-uncased')
        self.hidden = nn.Linear(enc_hid_dim, dec_hid_dim)
        self.decoder = decoder

    def forward(self, context_list, table_list, answer_list,teacher_forcing_ratio = 0.5):
        context_encoding, column_encoding, info_dict = self.model.encode(contexts=context_list, tables=table_list)
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
        print('encoder hidden output shape', hidden.shape)

        trg_vocab_size = self.decoder.output_dim
        trg = answer_list.permute(1,0)  # [trg_len,batch_size] [36,12]
        trg_len = trg.shape[0]
        print('trg_len:',trg_len)
        print('trg shape:', trg.shape)

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size)
        print('storage output shape:', outputs.shape)

        input = trg[0, :]
        print('input shape:',input.shape)
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            print('decoder_output,decoder_hidden',output.shape,hidden.shape)
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
        loss = criterion(outputs, trg)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch,teacher_forcing_ratio=0.5)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, "progress_bar": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch,teacher_forcing_ratio=0) #turn off teacher forcing ratio for the validation steps
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        # print(outputs)
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

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None, on_tpu=False,
                       using_native_amp=False, using_lbfgs=False):
        optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(path=self.hparams.train_data, model=self.model)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True,
                                num_workers=4, collate_fn=collate_fn)
        t_total = (
                (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
                // self.hparams.gradient_accumulation_steps * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(path=self.hparams.dev_data, model=self.model)
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4, collate_fn=collate_fn)

if __name__=='__main__':
    set_seed(42)

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

    args_dict = dict(
            train_data="./data/train_tabert.json",
            dev_data="./data/dev_tabert.json",
            output_dir="./",
            learning_rate=3e-4,
            weight_decay=0.0,
            adam_epsilon=1e-8,
            warmup_steps=0,
            train_batch_size=16,
            eval_batch_size=12,
            num_train_epochs=5,
            gradient_accumulation_steps=16,
            n_gpu=1,
            # early_stop_callback=False,
            fp_16=False,
            opt_level='O1',
            max_grad_norm=1.0,
            seed=42
        )
    args = argparse.Namespace(**args_dict)
    print(args_dict)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filepath=args.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=5)
    train_params = dict(
            accumulate_grad_batches=args.gradient_accumulation_steps,
            gpus=1,
            max_epochs=args.num_train_epochs,
            # early_stop_callback=False,
            precision=32,
            amp_level=args.opt_level,
            gradient_clip_val=args.max_grad_norm,
            checkpoint_callback=checkpoint_callback
        )
    OUTPUT_DIM = 11575
    DEC_EMB_DIM = 256
    ENC_HID_DIM = 768
    DEC_HID_DIM = 512
    DEC_DROPOUT = 0.5

    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
    model = TaBERTTuner(args,enc_hid_dim=ENC_HID_DIM,dec_hid_dim=DEC_HID_DIM,decoder=dec)
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)