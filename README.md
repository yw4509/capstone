# capstone

## Overall Model Structure
In this project, the model structure is split into encoder part and decoder part. The encoder is based on TaBERT published by Facebook and the decoder are created by ourselves.
We have came up with two types for decoder, one is classification and the other one is Seq2Seq Model.
### Data
Data_Processing.ipynb is used to combine tables with files and the output (train_tabert.json, test_tabert.json, dev_tabert.json )could be used as the direct input.

### Classification
#### Binary
try_binary.py
#### Multi
try_multi_attention_1.py
try_multi_attention_2.py
try_multi_attention_3.py

### Seq2Seq
#### attention
attention.py
#### self attention
self_attention.py
### argument parsed:
1. training directory: td
2. validation directory: vd
3. learning rate: lr
4. minimum learning rate: minlr
5. patience: p
6. number of pgu: gpu

example:
python self_attention.py -td "./data/train_tabert.json" -vd "./data/dev_tabert.json" -lr 5e-5 -minlr 2.5e-8 -p 4 -gpu 1

