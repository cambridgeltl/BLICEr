# Improving Bilingual Lexicon Induction with Cross-Encoder Reranking

This repository is the official PyTorch implementation of the following paper: 

Yaoyiran Li, Fangyu Liu, Ivan Vulić, and Anna Korhonen. 2022. *Improving Bilingual Lexicon Induction with Cross-Encoder Reranking*. In Findings of the Association for Computational Linguistics: EMNLP 2022. [[arXiv]](https://arxiv.org/abs/2210.16953)

<p align="center">
  <img width="500" src="model.png">
</p>

**BLICEr** is a post-hoc reranking method that works in the synergy with any given Cross-lingual Word Embedding (CLWE) space to improve Bilingual Lexicon Induction (BLI) / Word Translation. **BLICEr** is applicable to any existing CLWE induction method such as [ContrastiveBLI](https://github.com/cambridgeltl/ContrastiveBLI/), [RCSLS](https://github.com/facebookresearch/fastText/tree/main/alignment), and [VecMap](https://github.com/artetxem/vecmap). Our method first **1)** creates a cross-lingual word similarity dataset, comprising positive word pairs (i.e., true translations) and hard negative pairs induced from the original CLWE space, and then **2)** fine-tunes an mPLM (e.g., mBERT or XLM-R) in a [Cross Encoder](https://www.sbert.net/examples/applications/cross-encoder/README.html) manner to predict the similarity scores. At inference, we **3)** combine the similarity score from the original CLWE space with the score from the BLI-tuned cross-encoder. 

As reported in our paper, **BLICEr** is tested in four different BLI setups: 

- **Supervised**, 5k seed translation pairs

- **Semi-supervised**, 1k seed translation pairs

- **Unsupervised**, 0 seed translation pairs

- **Zero-shot**, 0 translation pairs directly between source and target languages but assume seed pairs between them and a third language respectively (no overlapping)

## Dependencies:

- PyTorch >= 1.10.1
- Transformers >= 4.15.0
- Python >= 3.9.7
- Sentence-Transformers >= 2.1.0

## Get Data and Set Input/Output Directories:
Following [ContrastiveBLI](https://github.com/cambridgeltl/ContrastiveBLI/), our data are obtained from the [XLING](https://github.com/codogogo/xling-eval) (8 languages, 56 BLI directions in total) and [PanLex-BLI](https://github.com/cambridgeltl/panlex-bli) (15 lower-resource languages, 210 BLI directions in total); please refer to [ContrastiveBLI](https://github.com/cambridgeltl/ContrastiveBLI/) for data preprocessing details.

Our BLICEr is compatible with any CLWE backbones. For brevity, our demo here is based on the state-of-the-art [ContrastiveBLI](https://github.com/cambridgeltl/ContrastiveBLI/) 300-dim C1 CLWEs, which is derived with purely static fastText embeddings (ContrastiveBLI also provides even stronger 768-dim C2 CLWEs which are trained with both fastText and mBERT). Please modify the input/output directories accordingly when using different CLWEs.  

## Run the Code:
```bash
python run_all.py
```
**Output**: source->target and target->source P@1 scores for each of &lambda; values in [0, 0.01, 0.02, ...
, 0.99, 1.0]. 
## Citation:
Please cite our paper if you find **BLICEr** useful. If you like our work, please ⭐ this repo.
```bibtex
@inproceedings{li-etal-2022-improving-bilingual,
    title     = {Improving Bilingual Lexicon Induction with Cross-Encoder Reranking},
    author    = {Li, Yaoyiran and Liu, Fangyu and Vuli{\'c}, Ivan and Korhonen, Anna},
    booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2022},
    year      = {2022}
}
```
