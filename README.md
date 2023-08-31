<div align="center">
    <a href="">
        <img alt="open-source-image"
		src="https://badges.frapsoft.com/os/v1/open-source.svg?v=103">
    </a>
    <a href="https://arxiv.org/abs/tba">
        <img alt="arxiv-image"
		src="https://img.shields.io/badge/arXiv-tba-b31b1b.svg">
    </a>
    <a href="https://github.com/Naereen/StrapDown.js/blob/master/LICENSE">
        <img alt="license-image"
		src="https://badgen.net/github/license/Naereen/Strapdown.js">
    </a>
</div>
<div align="center">
    <a href="https://www.youtube.com/channel/UChYsr-DIK7fSnTIhir0s3OQ">
        <img alt="youtube-image"
        src="https://img.shields.io/badge/YouTube-uygarkurtai-gray?style=flat&logo=YouTube&labelColor=%23FF0000">
    </a>
    <a href="https://www.linkedin.com/in/uygarr/">
        <img alt="linkedin-image"
        src="https://img.shields.io/badge/LinkedIn-uygarr-gray?logo=linkedin&labelColor=%230072b1">
    </a>
</div>

# Transformer Based Punctuation Restoration Models for Turkish
This repository contains the implementation of the paper Transformer Based Punctuation Restoration for Turkish. We present three pre-trained transformer models to predict **period**, **comma** and **question marks** for the Turkish language.

## Data
Dataset is provided in `data/` directory as train, validation and test splits.

Dataset can be summarized as below:

<!--
|    Split    |  Total  | Period (.) | Comma (,) | Question (?) |
|:-----------:|:-------:|:----------:|:---------:|:------------:|
|    Train    | 1471806 |   124817   |   98194   |     9816     |
| Validation  |  180326 |    15306   |   11980   |     1199     |
|   Test      |  182487 |    15524   |   12242   |     1255     |
-->

## Available Models
We experimented with BERT, ELECTRA and ConvBERT. Pre-trained models can be accessed via Huggingface.

BERT: https://huggingface.co/uygarkurt/bert-restore-punctuation-turkish \
ELECTRA: https://huggingface.co/uygarkurt/electra-restore-punctuation-turkish \
ConvBERT: https://huggingface.co/uygarkurt/convbert-restore-punctuation-turkish

## Training
For training you will need `transformers`, `datasets`. You can install the versions we used with the following commands:
`pip3 install transformers==4.25.1`
`pip3 install datasets==2.8.0`

To start the training run `python main.py`. By defualt BERT will be used for training. Trained models will be saved under `model_save/` in the main directory.

## Training With Different Models
BERT is used by default. To train with a different base model, change the tokenizer and model loaders which are `tokenizer = BertTokenizerFast.from_pretrained("dbmdz/bert-base-turkish-cased")` and `model = BertForTokenClassification.from_pretrained("dbmdz/bert-base-turkish-cased", num_labels = len(label_list), id2label = id2label, label2id = label2id)`  at `main.py` with the line numbers `46` and `47`.

## Usage
Recommended usage is via Huggingface. You can run an inference using the BERT model with the following code:
``` 
from transformers import pipeline

pipe = pipeline(task="token-classification", model="uygarkurt/bert-restore-punctuation-turkish")

sample_text = "Türkiye toprakları üzerindeki ilk yerleşmeler Yontma Taş Devri'nde başlar Doğu Trakya'da Traklar olmak üzere Hititler Frigler Lidyalılar ve Dor istilası sonucu Yunanistan'dan kaçan Akalar tarafından kurulan İyon medeniyeti gibi çeşitli eski Anadolu medeniyetlerinin ardından Makedonya kralı Büyük İskender'in egemenliğiyle ve fetihleriyle birlikte Helenistik Dönem başladı"

out = pipe(sample_text)
```

To use a different pre-trained model you can just replace the `model` argument with one of the other per-trained models we provided.

## Results
`Precision` and `Recall` and `F1` scores for each model and punctuation mark are summarized below.

<!--
|   Model  |          |  PERIOD  |          |          |  COMMA   |          |          | QUESTION |          |          | OVERALL  |          |
|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
|Score Type|     P    |     R    |    F1    |     P    |     R    |    F1    |     P    |     R    |    F1    |     P    |     R    |    F1    |
|   BERT   | 0.972602 | 0.947504 | 0.959952 | 0.576145 |  0.70001 | 0.632066 | 0.927642 | 0.911342 |  0.91942 | 0.825506 | 0.852952 | 0.837146 |
|  ELECTRA | 0.972602 | 0.948689 | 0.960497 |  0.5768  | 0.710208 |  0.63659 | 0.920325 | 0.921074 | 0.920699 | 0.823242 | 0.859990 | 0.839262 |
| ConvBERT | 0.972731 | 0.946791 | 0.959585 | 0.576964 | 0.708124 | 0.635851 | 0.922764 | 0.913849 | 0.918285 | 0.824153 | 0.856254 | 0.837907 |
-->
