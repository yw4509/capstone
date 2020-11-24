import argparse
import random
import os
import pickle
import glob
import json
import time
import math
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

class voc():
    def __init__(self, df, voc_location, minimum_count, max_num):
        #df here is answers, list of list of tokens
        self.df=df
        self.minimum_count = minimum_count;
        self.max_num = max_num;
        self.voc_location = voc_location;
        self.main_df, self.target_voc = self.load_or_create_voc()
        #main df includes target_tokenized, target_indized, target_len
        #target_voc is the Lang class with full vocab and can perform idx to token, token to idx, token to count opertations
    def __len__(self):
        return len(self.main_df) if self.max_num is None else self.max_num
    def __getitem__(self, idx):
        return_list = [self.main_df.iloc[idx]['target_indized'], self.main_df.iloc[idx]['target_len'] ]
        return return_list
    def load_or_create_voc(self):
        if not os.path.exists(self.voc_location):
            os.makedirs(self.voc_location)
        full_file_path = os.path.join(self.voc_location, 'mincnt_maxnum' +
                                      str(self.minimum_count) + '_' + \
                                      str(self.max_num)+'.p')
        #if the address exits, we will load the dictionary from the full path,
        #ow, we will create a new voc dictionary and pickle dump to full path
        if os.path.isfile(full_file_path):
            print('Load Pre-existing Voc Dictionary')
            target_voc = pickle.load(open(full_file_path,'rb'))
        else:
            print('Create New Voc Dictionary')
            target_voc = Lang(minimum_count = self.minimum_count);
            for ans in self.df: # load ans into voc
                target_voc.add_ans(ans)
            pickle.dump(target_voc,open(full_file_path,'wb'))
        # change token to idx based on dictionary
        indices_data = []
        for ans in self.df: # ans tokens to idx
            index_list = [target_voc.word2index[token] if token in target_voc.word2index else UNK_IDX for token in ans]
            if len(index_list)<=self.max_num:
                index_list = index_list + [PAD_IDX]*(self.max_num-len(index_list))
            else:
                index_list = index_list[:self.max_num]
            index_list.append(EOS_IDX) # add EOS token to the answer
            # print('index_list',index_list)
            # print('ans',ans)
            indices_data.append(index_list)
        main_df = pd.DataFrame();
        main_df['target_tokenized'] = self.df;
        main_df['target_indized'] = indices_data;
        main_df['target_len'] = main_df['target_tokenized'].apply(lambda x: len(x)+1) #+1 for EOS
        main_df =  main_df[main_df['target_len'] >=2] #filter out ans that are empty
        main_df['ind'] = main_df['target_len'] >=2
        return main_df,target_voc

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
        self.voc_obj = voc(self.answers, self.voc_location, minimum_count=minimum_count, max_num=max_num)

        self.answers = self.voc_obj.main_df.target_indized.tolist()
        ind = self.voc_obj.main_df.ind.tolist()
        self.tabs = np.array(self.tabs)[ind] #remove the ones with ans len <2
        self.context = np.array(self.context)[ind]
        print('len check', len(self.answers)==len(self.tabs)==len(self.context))

        self.target_voc=self.voc_obj.target_voc

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

    def __len__(self):
        return len(self.context)
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
    return [batch[i]['table'] for i in range(len(batch))], [batch[i]['context'] for i in range(len(batch))], torch.tensor([batch[i]['answer'] for i in range(len(batch))])


class Attention_Module(pl.LightningModule):
    def __init__(self, hidden_dim, output_dim):
        super(Attention_Module, self).__init__()
        self.l1 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.l2 = nn.Linear(hidden_dim + output_dim, output_dim, bias=False)

    def forward(self, hidden, encoder_outs):
        '''
        hiddden: bsz x hidden_dim
        encoder_outs: bsz x sq_len x encoder dim (output_dim)
        src_lens: bsz

        x: bsz x output_dim
        attn_score: bsz x sq_le
        '''
        x = self.l1(hidden)
        # [src_len,batch_size,enc_hid_dim]
        encoder_outs = encoder_outs.permute(1, 0, 2)  # [ batch_size,src_len,dim]
        att_score = torch.bmm(encoder_outs, x.unsqueeze(-1));  # this is bsz x seq x 1
        att_score = att_score.squeeze(-1);  # this is bsz x seq
        att_score = att_score.transpose(0, 1);

        batch_size = encoder_outs.shape[0]
        src_len = encoder_outs.shape[1]
        src_lens = torch.ones(batch_size) * src_len

        seq_mask = self.sequence_mask(src_lens,
                                      max_len=max(src_lens).item()).transpose(0, 1)
        masked_att = seq_mask * att_score  # [seq_len, batch_size], [seq_len, batch_size] element multiplication
        masked_att[masked_att == 0] = -1e10
        attn_scores = F.softmax(masked_att, dim=0)  # [seq_len, batch_size]
        # encoder_outs.transpose(0, 1) [seq_len, batch_size, dim] #[seq_len, batch_size,1]
        x = (attn_scores.unsqueeze(2) * encoder_outs.transpose(0, 1)).sum(dim=0)  # [batch_size, output_dim]
        x = torch.tanh(self.l2(torch.cat((x, hidden), dim=1)))  # [batch_size, output_dim]
        return x, attn_scores

    def sequence_mask(self, sequence_length, max_len=None, device=torch.device('cuda')):
        #         print('max len', max_len) 16
        #         print('sequence_length',sequence_length) [16,14,12,11,11,16...] shape [batch_size]
        if max_len is None:
            max_len = sequence_length.max().item()
        batch_size = sequence_length.size(0)
        #         print('batch_size',batch_size)
        seq_range = torch.arange(0, max_len).long()
        seq_range_expand = seq_range.unsqueeze(0).repeat([batch_size, 1])
        seq_range_expand = seq_range_expand  # [batch_size, max_len]
        #         print('seq_range_expand shape',seq_range_expand.shape)

        seq_length_expand = (sequence_length.unsqueeze(1)  # [batch_size,1]
                             .expand_as(seq_range_expand))  # [batch_size,max_len] seq_length repeated for max_len times
        # it returns a matrix of batch_size, max_len with diagonal above = True
        return (seq_range_expand < seq_length_expand).float().to(device)


class Decoder_SelfAttn(pl.LightningModule):
    """Generates a sequence of tokens in response to context with self attention.
       Note that this is the same as previous decoder if self_attention=False"""

    def __init__(self, output_size, hidden_size, idropout=0.5, self_attention=False, encoder_attention=False):
        super(Decoder_SelfAttn, self).__init__()

        self.output_size = output_size;

        self.self_attention = self_attention;
        self.encoder_attention = encoder_attention;

        self.hidden_size = hidden_size;
        self.embedding = nn.Embedding(output_size, hidden_size);

        self.memory_rnn = nn.GRUCell(hidden_size + int(self.encoder_attention == True) * self.hidden_size,
                                     hidden_size, bias=True);

        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

        if self.self_attention:
            self.projector_summ = nn.Sequential(nn.Dropout(idropout),
                                                nn.Linear(hidden_size * 2, hidden_size),
                                                nn.Dropout(idropout))
        if self.encoder_attention:
            self.encoder_attention_module = Attention_Module(self.hidden_size, self.hidden_size);

    def forward(self, input, memory, encoder_output=None, xs_len=None, context_vec=None):
        # memory [12, 768]
        memory = memory.unsqueeze(0)
        #         print('memory',memory.shape)
        memory = memory.transpose(0, 1);  # memory is decoder_hidden which is encoder_hidden here [64, 1, 512]
        #         print('memory',memory.shape)
        emb = self.embedding(input)  # input [64, 36] emb [64, 36, 512]
        emb = F.relu(emb)

        emb = emb.transpose(0, 1);  # emb [16,64,512]
        return_scores = torch.empty(emb.size(0), emb.size(1), self.output_size)  # [16, 64, 20111]
        if context_vec is None and self.encoder_attention:
            context_vec = torch.zeros([emb.size(1), self.hidden_size], device=self.device);  # [64, 512]
        if self.encoder_attention:
            attn_wts_list = [];
        else:
            attn_wts_list = None;
        for t in range(emb.size(0)):  # range trg_len
            current_vec = emb[t];  # [64, 512]
            selected_memory = memory[:, 0, :];  # [64, 512]
            if self.self_attention:
                #                 print('memory',memory.shape)
                selected_memory, attention0 = self.calculate_self_attention(current_vec, memory)
                # [64,521] [64, 1, 1]
            if self.encoder_attention:
                current_vec = torch.cat([current_vec, context_vec], dim=1);  # [64, 1024]
            if (not (self.self_attention or self.encoder_attention)):
                selected_memory, attention0 = memory[:, 0, :], None;  ##[64, 512]
            # recurrent
            mem_out = self.memory_rnn(current_vec, selected_memory);  # [64, 512]
            #             print('mem_out',mem_out.shape)
            # GRUCell input x and hidden, output is the next hidden
            if self.encoder_attention:
                context_vec, attention0 = self.encoder_attention_module(mem_out, encoder_output);
                scores = self.out(context_vec);
                attn_wts_list.append(attention0)
            else:
                scores = self.out(mem_out)  # linear hidden to output
            #                 print('scores',scores.shape)

            scores = self.softmax(scores);
            return_scores[t] = scores

            if self.self_attention:
                # update memory
                # [64, 1, 512], [64,0,512] => [64,1,512]
                memory = torch.cat([mem_out[:, None, :], memory[:, :-1, :]], dim=1);
            else:
                memory = mem_out[:, None, :];
        return return_scores.transpose(0, 1).contiguous(), memory.transpose(0, 1), attn_wts_list, context_vec

    def calculate_self_attention(self, input, memory):
        #         print('input', input.shape) which is current_vector
        #         print('memory', memory.shape)
        # input torch.Size([12, 768]) , memory torch.Size([1, 12, 768])
        concat_vec = torch.cat([input, memory[:, 0, :]], dim=1);  # [64,1024]
        projected_vec = self.projector_summ(concat_vec);  # [64,512]
        dot_product_values = torch.bmm(memory, projected_vec.unsqueeze(-1)).squeeze(-1) / math.sqrt(
            self.hidden_size);  # [64,1]
        weights = F.softmax(dot_product_values, dim=1).unsqueeze(-1);  # [64, 1, 1]
        selected_memory = torch.sum(memory * weights, dim=1)  # [64, 512]
        return selected_memory, weights


class TaBERTTuner(pl.LightningModule):
    def __init__(self, hparams, decoder, enc_hid_dim, dec_hid_dim):
        super(TaBERTTuner, self).__init__()
        self.hparams = hparams
        self.encoder = TableBertModel.from_pretrained('bert-base-uncased')
        self.hidden = nn.Linear(enc_hid_dim, dec_hid_dim)
        self.decoder = decoder
        self.START = torch.LongTensor([SOS_IDX]).to(device)

    def forward(self, context_list, table_list, answer_list):
        context_encoding, column_encoding, info_dict = self.encoder.encode(contexts=context_list, tables=table_list)
        # print('context_encoding, column_encoding, answer_list shape:',context_encoding.shape,column_encoding.shape, answer_list.shape)
        # context_encoding: [batch_size, token_size, 768] [12,17,768]
        # column_encoding: [batch_size, number of columns, 768] [12,6,768]
        # answer_list: [batch_size, 36] max_len is 35 and adds one for eos

        src = torch.cat([context_encoding, column_encoding], dim=1)  # [batch_size,scr_len,768*2] [12,123,768]
        encoder_output = src.permute(1, 0, 2)  # [src_len,batch_size,enc_hid_dim] [123,12,768]
        batch_size = encoder_output.shape[1]
        starts = self.START.expand(batch_size, 1)  # expand to batch size; start is [SOS]
        # print('scr shape:', src.shape)
        # print('encoder_outputs shape:', encoder_outputs.shape)
        # print('batch size:',batch_size)

        encoder_hidden = torch.tanh(self.hidden(torch.sum(encoder_output, axis=0)))  # [batch_size=12,dec_hid_dim=512]
        # print('encoder hidden output shape', hidden.shape)

        trg_vocab_size = self.decoder.output_size
        trg = answer_list  # [batch_size,trg_len] [64,36]
        trg_len = trg.shape[1]
        # print('trg_len:',trg_len)
        # print('trg shape:', trg.shape)

        # cut off eos from the input
        y_in = trg.narrow(1, 0, trg.size(1) - 1)  # narrow(dim, start, length)
        # add in sos in front of the sentence
        decoder_input = torch.cat([starts, y_in], 1)
        decoder_output, decoder_hidden, _, _ = self.decoder(decoder_input,
                                                            encoder_hidden,
                                                            encoder_output)

        return decoder_output

    def _step(self, batch):
        tbl, ctx, ans = batch[0], batch[1], torch.tensor(batch[2])
        # print('step ----------------------------------------------------------------------------------------------------')
        decoder_output = self(ctx, tbl, ans)  # output = [trg len, batch size, output dim]
        _max_score, predictions = decoder_output.max(2)
        # ans = [trg len, batch size]   [w,w,w,w,eos]
        # print('outputs from step:', outputs.shape)
        # print('trg from step:', trg.shape)
        output_dim = decoder_output.shape[-1]

        outputs = decoder_output.view(-1, output_dim)  # output = [(trg len-0 ) * batch size, output dim]
        trg = ans.contiguous().view(-1)  # trg = [(trg len - 0) * batch size] [w,w,w,w,eos]
        # print('outputs from step:', outputs.shape)
        # print('trg from step:', trg.shape)
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        outputs = outputs.to(device=device)
        trg = trg.to(device=device)
        loss = criterion(outputs, trg)
        return loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        loss = self._step(batch)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        #         print('train epoch end--------------------------------------------------------')
        #         print(outputs)
        #         print(outputs[0])
        #         print(outputs[1])
        avg_train_loss_encoder = torch.stack([x['loss'] for x in outputs[0]]).mean()
        avg_train_loss_decoder = torch.stack([x['loss'] for x in outputs[1]]).mean()
        avg_train_loss = torch.stack((avg_train_loss_encoder, avg_train_loss_decoder)).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, "progress_bar": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_val_loss}
        return {"avg_val_loss": avg_val_loss, "log": tensorboard_logs, "progress_bar": tensorboard_logs}

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters_encoder = [
            {
                "params": [p for n, p in self.encoder.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.encoder.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }, ]
        #         optimizer_grouped_parameters_decoder = [
        #             {
        #                 "params": [p for n, p in self.decoder.named_parameters() if not any(nd in n for nd in no_decay)],
        #                 "weight_decay": self.hparams.weight_decay,
        #             },
        #             {
        #                 "params": [p for n, p in self.decoder.named_parameters() if any(nd in n for nd in no_decay)],
        #                 "weight_decay": 0.0,
        #             }, ]

        encoder_optimizer = AdamW(optimizer_grouped_parameters_encoder, lr=self.hparams.lr,
                                  eps=self.hparams.adam_epsilon)
        decoder_optimizer = torch.optim.SGD(self.decoder.parameters(), lr=self.hparams.lr, nesterov=True,
                                            momentum=self.hparams.momentum)
        #         decoder_optimizer = torch.optim.SGD(optimizer_grouped_parameters_decoder, lr=self.hparams.lr, nesterov=True,
        #                                  momentum = self.hparams.momentum)
        optimizers = [encoder_optimizer, decoder_optimizer]

        schedulers = [
            {'scheduler': ReduceLROnPlateau(encoder_optimizer, mode="min", min_lr=7.5e-5, patience=5, verbose=True),
             # might need to change here
             'monitor': "val_loss",  # Default: val_loss
             'interval': 'epoch',
             'frequency': 1
             },
            {'scheduler': ReduceLROnPlateau(decoder_optimizer, mode="min", min_lr=7.5e-5, patience=5, verbose=True),
             # might need to change here
             'monitor': "val_loss",  # Default: val_loss
             'interval': 'epoch',
             'frequency': 1
             }
        ]
        return optimizers, schedulers

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(path=self.hparams.train_data, voc_location=self.hparams.voc_location,
                                    model=self.encoder, minimum_count=self.hparams.minimum_count,
                                    max_num=self.hparams.max_num)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size,
                                drop_last=True, shuffle=True, num_workers=4, collate_fn=collate_fn)
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

    args = parser.parse_args()
    train_data=args.td
    val_data = args.vd
    lr = float(args.lr)

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
        n_gpu=1,
        fp_16=False,  # fp_16 true will end up shorter trainning time. 32 is default
        opt_level='O1',  # pure or mixed precision
        seed=42
        # early_stop_callback=False,
    )
    args = argparse.Namespace(**args_dict)
    print(args_dict)

    checkpoint_callback = ModelCheckpoint(
        filepath=args.output_dir,
        prefix='checkpoint-{epoch:02d}',
        monitor="val_loss", mode="min", save_top_k=5)

    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        max_epochs=args.num_train_epochs,
        amp_level=args.opt_level,
        gradient_clip_val=args.gradient_clip_val,
        auto_lr_find=True,
        precision=32,
        checkpoint_callback=checkpoint_callback
        # early_stop_callback=False,
    )

    OUTPUT_DIM = 11575

    ENC_HID_DIM = 768

    DEC_EMB_DIM = 516
    DEC_HID_DIM = 768
    DEC_DROPOUT = 0.5

    dec = Decoder_SelfAttn(output_size=OUTPUT_DIM, hidden_size=DEC_HID_DIM, idropout=0.5,
                           self_attention=True, encoder_attention=True)
    model = TaBERTTuner(args, enc_hid_dim=ENC_HID_DIM, dec_hid_dim=DEC_HID_DIM, decoder=dec)

    trainer = pl.Trainer(**train_params)
    torch.cuda.empty_cache()
    trainer.fit(model)