import os
import sys
import torch
import numpy as np
import time
from utils import load_embs

lang_pairs = [('en', 'de'),
 ('en', 'fi'),
 ('en', 'fr'),
 ('en', 'hr'),
 ('en', 'it'),
 ('en', 'ru'),
 ('en', 'tr')]
 
#Hyper-parameters to reproduce our results.
gpuid = '0'
train_size = '1k' # "5k" or "1k" or "0k"
DIR_NEW = "./BLI/"
os.system("mkdir -p {}".format(DIR_NEW))
output_root = "Directory for Saving CE Models"
random_seed = 33
origin_model_dir = "xlm-roberta-large"#"bert-base-multilingual-uncased" #"bert-base-multilingual-uncased"
max_length = 20
num_neg = 28
use_template = 1
my_template = "'{} ({})!'"
 
batch_size_cross_encoder = 256 
task = "custom" 
delta = 0.1 
alpha = 0.7
num_epochs_cross_encoder = 3
num_dup = 8
if train_size == '1k':
    delta = 0.2
    alpha = 1.0
    num_epochs_cross_encoder = 5
    num_dup = 4

# To reproduce our reported results for unsupervised and zero-shot BLI setups, please just simply follow the instructions and settings described in our paper.


for (lang1, lang2) in lang_pairs:
    print(lang1, lang2)
    sys.stdout.flush()


    ROOT_FT = "Directory of Pre-calculated CLWEs".format(train_size)
    # ContrastiveBLI stage C1 will save mapped CLWEs into the following four files, including the vocabularies (in .npy files) and C1-induced embeddings (in .pt files) for each the source and target languages respectively. Please refer to https://github.com/cambridgeltl/ContrastiveBLI.
    l1_voc = ROOT_FT + "/{}2{}_{}_voc.npy".format(lang1,lang2,lang1)
    l1_emb = ROOT_FT + "/{}2{}_{}_emb.pt".format(lang1,lang2,lang1)
    l2_voc = ROOT_FT + "/{}2{}_{}_voc.npy".format(lang1,lang2,lang2)
    l2_emb = ROOT_FT + "/{}2{}_{}_emb.pt".format(lang1,lang2,lang2)

    # The VecMap and RCSLS codes save CLWEs in txt files (e.g., with suffix .tsv, .vec), we also transform them into .npy and .pt files. For ContrastiveBLI, the additional transformation is not needed.
    if (l1_emb[-3:] == "vec") or (l1_emb[-3:] == "tsv"): 
        voc_l1, embs_l1 = load_embs(l1_emb)
        voc_l2, embs_l2 = load_embs(l2_emb)
        NEW_ROOT = "./TMP"
        os.system("mkdir -p {}".format(NEW_ROOT))
        os.system("rm -f -r {}/*".format(NEW_ROOT))
        l1_voc = NEW_ROOT + "/{}2{}_{}_voc.npy".format(lang1,lang2,lang1)
        l1_emb = NEW_ROOT + "/{}2{}_{}_emb.pt".format(lang1,lang2,lang1)
        l2_voc = NEW_ROOT + "/{}2{}_{}_voc.npy".format(lang1,lang2,lang2)
        l2_emb = NEW_ROOT + "/{}2{}_{}_emb.pt".format(lang1,lang2,lang2)

        train_data_l1 = torch.from_numpy(embs_l1)
        train_data_l2 = torch.from_numpy(embs_l2)

        np.save(l1_voc, voc_l1)
        np.save(l2_voc, voc_l2)
        torch.save(train_data_l1, l1_emb) #aligned l1 WEs
        torch.save(train_data_l2, l2_emb) #aligned l2 WEs

        del train_data_l1,train_data_l2,embs_l1,embs_l2,voc_l1,voc_l2

    DIR_TEST_DICT = "Directory of The Test Set/xling-eval/bli_datasets/{}-{}/yacle.test.freq.2k.{}-{}.tsv".format(lang1,lang2,lang1,lang2)
    DIR_TRAIN_DICT = "Directory of The Training Set/xling-eval/bli_datasets/{}-{}/yacle.train.freq.{}.{}-{}.tsv".format(lang1,lang2,train_size,lang1,lang2)

    model_name_or_path = origin_model_dir #"bert-base-multilingual-uncased" #C2_root + "mbert_{}2{}_{}".format(lang1,lang2,train_size)
    custom_corpus_path = DIR_NEW + "{}2{}_".format(lang1,lang2)
    start = time.time()
    print("Generate Pos-Neg Pairs")
    sys.stdout.flush()
    os.system('CUDA_VISIBLE_DEVICES={} python gen_neg_samples.py --l1 {} --l2 {} --train_size {} --root {} --num_neg {} --l1_voc {} --l1_emb {} --l2_voc {} --l2_emb {} --train_dict_dir {} --test_dict_dir {} --delta {}'.format(gpuid,lang1, lang2, train_size, DIR_NEW, num_neg, l1_voc, l1_emb, l2_voc, l2_emb, DIR_TRAIN_DICT, DIR_TEST_DICT, delta))
    end = time.time() 
    print("Runtime for Step 1 (Generate Pos-Neg Pairs): ", end-start," seconds")
    sys.stdout.flush()

    current = os.getcwd().split('/')[-1]

    output_dir = output_root + "{}_{}2{}_{}".format(origin_model_dir, lang1, lang2, train_size)

    start = time.time()
    print("Cross-Encoder Training")
    sys.stdout.flush()
    os.system("CUDA_VISIBLE_DEVICES={} python src/train.py --model_name_or_path {} --batch_size_cross_encoder {} --num_epochs_cross_encoder {} --init_with_new_models --task {} --custom_corpus_path {} --random_seed {} --output_dir {} --origin_model_dir {} --use_template {} --max_length {} --my_template {} --num_dup {} --alpha {}".format(gpuid, model_name_or_path, batch_size_cross_encoder, num_epochs_cross_encoder, task, custom_corpus_path, random_seed, output_dir, origin_model_dir, use_template, max_length, my_template, num_dup, alpha))        
    end = time.time()
    print("Runtime for Step 2 (Cross-Encoder Training): ", end-start," seconds")
    sys.stdout.flush()
    start = time.time()
    print("EVALUATION")
    sys.stdout.flush()
    os.system('CUDA_VISIBLE_DEVICES={} python evaluate_ce.py --l1 {} --l2 {} --num_neg {}  --l1_voc {} --l1_emb {} --l2_voc {} --l2_emb {} --train_dict_dir {} --test_dict_dir {} --custom_corpus_path {} --model_name {}  --use_template {} --max_length {} --my_template {}'.format(gpuid,lang1, lang2, num_neg, l1_voc, l1_emb, l2_voc, l2_emb, DIR_TRAIN_DICT, DIR_TEST_DICT, custom_corpus_path, output_dir, use_template, max_length, my_template))
    #os.system("rm -r {}".format(output_dir))
    end = time.time()
    print("Runtime for Step 3 (EVALUATION): ", end-start," seconds")
    sys.stdout.flush()
