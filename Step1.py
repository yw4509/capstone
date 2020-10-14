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

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import pandas as pd
import numpy as np
import torch
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


class WikiDataset(Dataset):
    def __init__(self, path, model):
        self.path = path
        self.model = model

        self.data = pd.read_json(self.path)
        self.data['title'] = self.data['title'].fillna('unknown')
        lens = self.data['answer'].apply(lambda x: len(model.tokenizer.tokenize(str(x[0]))))
        self.data = self.data[lens == 1]
        self.data = self.data.reset_index(drop=True)
        print(self.data.shape)

        self.tabs = []
        self.context = []
        self.answers = []

        self._build()

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
            self.answers.append(self.model.tokenizer.convert_tokens_to_ids(self.model.tokenizer.tokenize(str(ans[0])))[0])
            # self.answers.append(self.model.tokenizer.convert_tokens_to_ids(str(ans[0]))[0])

            # self.answers.append(self.model.tokenizer.encode(str(ans[0]), add_sepcial_tokens=False)[0])
            # ans_enc_t = self.model.tokenizer.encode(str(ans[0]),max_length=7,truncattion=True)
            # ans_enc = ans_enc_t + [0] * (7 - len(ans_enc_t))
            # self.answers.append(ans_enc)

def get_dataset(path, model):
    return WikiDataset(path=path, model=model)

def collate_fn(batch):
    # print('table','-' * 100)
    # print([batch[i]['table'] for i in range(len(batch))])
    # print('context','-'*100)
    # print([batch[i]['context'] for i in range(len(batch))])
    # print('answer','-' * 100)
    # print(torch.tensor([batch[i]['answer'] for i in range(len(batch))]))
    return [batch[i]['table'] for i in range(len(batch))], [batch[i]['context'] for i in range(len(batch))], torch.tensor([batch[i]['answer'] for i in range(len(batch))])


class TaBERTTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(TaBERTTuner, self).__init__()
        self.hparams = hparams

        self.model = TableBertModel.from_pretrained('bert-base-uncased')
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.l1 = nn.Linear(1536, 500)
        self.l2 = nn.Linear(500, len(self.model.tokenizer.vocab))
        # self.l2 = nn.Linear(500, self.model.tokenizer.vocab_size)

    def forward(self, context_list, table_list):
        context_encoding, column_encoding, info_dict = self.model.encode(contexts=context_list, tables=table_list)

        ctx_enc_sum = torch.sum(context_encoding, axis=1)
        col_enc_sum = torch.sum(column_encoding, axis=1)

        catenated = torch.cat([ctx_enc_sum, col_enc_sum], dim=1)
        out = self.l1(catenated)
        out = self.l2(out)

        return out

    def _step(self, batch):
        if torch.cuda.is_available():
            tbl, ctx, ans = batch[0], batch[1], torch.tensor(batch[2]).to('cuda')
        else:
            tbl, ctx, ans = batch[0], batch[1], torch.tensor(batch[2])
        # print(ans)
        outputs = self(ctx, tbl)
        l = nn.CrossEntropyLoss(ignore_index=0)
        loss = l(outputs, ans)

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
        print(outputs)
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
      val_dataset=get_dataset(path=self.hparams.dev_data, model = self.model)
      return  DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4,collate_fn=collate_fn)

    # def collate_fn(batch):
    #   return [batch[i]['table'] for i in range(len(batch))], [batch[i]['context'] for i in range(len(batch))], torch.tensor([batch[i]['answer'] for i in range(len(batch))])

if __name__=='__main__':
    # df = pd.read_json('./data/train_tabert.json')
    # print()
    set_seed(42)
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
        early_stop_callback=False,
        fp_16=False,
        opt_level='O1',
        max_grad_norm=1.0,
        seed=42,
    )
    args = argparse.Namespace(**args_dict)
    print(args_dict)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=args.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=5)
    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=1,
        max_epochs=args.num_train_epochs,
        early_stop_callback=False,
        precision=32,
        amp_level=args.opt_level,
        gradient_clip_val=args.max_grad_norm,
        checkpoint_callback=checkpoint_callback
    )
    model = TaBERTTuner(args)
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)