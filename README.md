# Improving Bilingual Lexicon Induction with Cross-Encoder Reranking

This repository is the official PyTorch implementation of the following paper: 

Yaoyiran Li, Fangyu Liu, Ivan Vulić, and Anna Korhonen. 2022. *Improving Bilingual Lexicon Induction with Cross-Encoder Reranking*. In Findings of EMNLP 2022. 

## Dependencies:

- PyTorch >= 1.10.1
- Transformers >= 4.15.0
- Python >= 3.9.7
- Sentence-Transformers >= 2.1.0

## Get Data and Set Input/Output Directories:
Following our previous work [ContrastiveBLI](https://github.com/cambridgeltl/ContrastiveBLI/), our data are obtained from the [XLING repo](https://github.com/codogogo/xling-eval), please refer to [ContrastiveBLI](https://github.com/cambridgeltl/ContrastiveBLI/) for data preprocessing details.

We recommend to use [ContrastiveBLI](https://github.com/cambridgeltl/ContrastiveBLI/) (either C1 or C2) as the Cross-lingual Word Embedding (CLWE) backbone. 
