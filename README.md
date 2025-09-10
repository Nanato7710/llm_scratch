# Build a LLM from scratch

## About

llmを事前学習から作成し，最終的に雑談，ロールプレイ，数学などの汎用的な用途で使える日本語と英語のバイリンガルChatBotにする．

モデルのアーキテクチャはGemma3-270mで、事前学習で日本語にはfineweb2のjpn_Jpan、英語にはfinewebのdefaultを使用する．
tokenizerにはbyte-level bpeを事前学習に用いたデータセットで作成し，語彙数は80kトークンとする．

## LLM Architecture

Gemma3-270m

## Tokenizer

Byte-level BPE

## Pre-Training

### Datasets

|Target|Dataset|Subset|Link|
|:---|:---|:---|:---|
|Japanese|HuggingFaceFW/fineweb-2|jpn_Jpan|[Hugging Face](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)|
|English|HuggingFaceFW/fineweb|default|[Hugging Face](https://huggingface.co/datasets/HuggingFaceFW/fineweb)|

### How to Train

基本的な学習手法は**Next Token Prediction**とし，**カリキュラム学習**を行う．

#### Curriculum Learning

1. FineWebシリーズでNTPを行う．
2. コードや数学，アニメなどの特定分野に特化したデータセットでNTPを行う．

## Post-Training

### Datasets
