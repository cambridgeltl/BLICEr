import sys
import io
import subprocess as commands
import codecs
import copy
import argparse
import math
import pickle as pkl
import os
import numpy as np
import torch
import torch.nn as nn
import collections
import torch.nn.functional as F
from sentence_transformers import models, losses, util, SentenceTransformer, LoggingHandler
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sentence_transformers.readers import InputExample
import itertools

def getknn_test(src_, tgt_, src_ids, tgt_ids, lexicon_size, k=10, bsz=1024):
    #k: num of neg samples
    #k_csls: usually 10
    src_ = src_.cuda()
    tgt_ = tgt_.cuda()
    src = src_ / (torch.norm(src_, dim=1, keepdim=True) + 1e-9)
    tgt = tgt_ / (torch.norm(tgt_, dim=1, keepdim=True) + 1e-9)
    num_imgs = len(src)
    confuse_output_indices = []
    scores = torch.zeros(num_imgs,k)

    for batch_idx in range( int( math.ceil( float(num_imgs) / bsz ) ) ):
        start_idx = batch_idx * bsz
        end_idx = min( num_imgs, (batch_idx + 1) * bsz )
        length = end_idx - start_idx
        prod_batch = torch.matmul(src[start_idx:end_idx, :], tgt.T)
        dotprod = torch.topk(prod_batch,k=k,dim=1,sorted=True,largest=True).indices
        confuse_output_indices += dotprod.cpu().tolist()

    assert len(confuse_output_indices) == num_imgs
    for i in range(num_imgs):
        src_embs = src[[i]]
        tgt_embs = tgt[confuse_output_indices[i]]
        scores[i] = src_embs @ tgt_embs.T

    accuracy = 0
    for i in range(num_imgs):
        if confuse_output_indices[i][0] == tgt_ids[i]:
            accuracy += 1
    accuracy = accuracy / float(lexicon_size)
    return confuse_output_indices, scores, accuracy

def getknn_csls_test(src_, tgt_, src_ids, tgt_ids, src_hubness_, tgt_hubness_, lexicon_size, k=10, bsz=1024, t=1.0):
    #k: num of neg samples
    #k_csls: usually 10
    src_ = src_.cuda()
    tgt_ = tgt_.cuda()
    src = src_ / (torch.norm(src_, dim=1, keepdim=True) + 1e-9)
    tgt = tgt_ / (torch.norm(tgt_, dim=1, keepdim=True) + 1e-9)
    num_imgs = len(src)
    confuse_output_indices = []
    scores = torch.zeros(num_imgs,k)

    src_hubness, tgt_hubness = src_hubness_.cuda(), tgt_hubness_.cuda()

    src_hubness_sup = src_hubness[src_ids]
    tgt_hubness_sup = tgt_hubness[tgt_ids]

    for batch_idx in range( int( math.ceil( float(num_imgs) / bsz ) ) ):
        start_idx = batch_idx * bsz
        end_idx = min( num_imgs, (batch_idx + 1) * bsz )
        length = end_idx - start_idx
        prod_batch = (1.0 + t) * torch.matmul(src[start_idx:end_idx, :], tgt.T) - t * src_hubness_sup[start_idx:end_idx].unsqueeze(1) - t * tgt_hubness.unsqueeze(0)
        dotprod = torch.topk(prod_batch,k=k,dim=1,sorted=True,largest=True).indices
        confuse_output_indices += dotprod.cpu().tolist()

    assert len(confuse_output_indices) == num_imgs

    for i in range(num_imgs):
        src_embs = src[[i]]
        tgt_embs = tgt[confuse_output_indices[i]]
        scores[i] = (1.0 + t) * src_embs @ tgt_embs.T - t * src_hubness_sup[i:i+1].unsqueeze(1) - t * tgt_hubness[confuse_output_indices[i]].unsqueeze(0)

    accuracy = 0
    for i in range(num_imgs):
        if confuse_output_indices[i][0] == tgt_ids[i]:
            accuracy += 1
    accuracy = accuracy / float(lexicon_size)
    return confuse_output_indices, scores, accuracy


def lexicon_dict2list(lexicon_dict):
    res_src = []
    res_tgt = []
    for key in lexicon_dict.keys():
        for value in lexicon_dict[key]:
            res_src.append(key)
            res_tgt.append(value)
    return res_src, res_tgt

def eval_BLI(train_data_l1, train_data_l2, src2tgt, lexicon_size_s2t, tgt2src, lexicon_size_t2s, src_hubness, tgt_hubness, best_t, k=28):

    train_data_l1_translation = train_data_l1.cuda()
    train_data_l2_translation = train_data_l2.cuda()

    s2t_s, s2t_t = lexicon_dict2list(src2tgt)
    t2s_t, t2s_s = lexicon_dict2list(tgt2src)

    assert set(s2t_s) == set(t2s_s)
    assert set(t2s_t) == set(s2t_t)
    assert len(s2t_s) == len(t2s_s)
    assert len(s2t_t) == len(s2t_t)
    assert lexicon_size_s2t == len(src2tgt.keys())
    assert lexicon_size_t2s == len(tgt2src.keys())


    nn_predict_s2t, nn_score_s2t, acc_s2t = getknn_test(train_data_l1_translation[s2t_s], train_data_l2_translation, s2t_s, s2t_t, lexicon_size_s2t, k=k, bsz=1024)
    nn_predict_t2s, nn_score_t2s, acc_t2s = getknn_test(train_data_l2_translation[t2s_t], train_data_l1_translation, t2s_t, t2s_s, lexicon_size_t2s, k=k, bsz=1024)

    csls_predict_s2t, csls_score_s2t, cslsacc_s2t = getknn_csls_test(train_data_l1_translation[s2t_s], train_data_l2_translation, s2t_s, s2t_t, src_hubness, tgt_hubness, lexicon_size_s2t, k=k, bsz=1024, t=best_t)
    csls_predict_t2s, csls_score_t2s, cslsacc_t2s = getknn_csls_test(train_data_l2_translation[t2s_t], train_data_l1_translation, t2s_t, t2s_s, tgt_hubness, src_hubness, lexicon_size_t2s, k=k, bsz=1024, t=best_t)

    BLI_accuracy_l12l2 = (acc_s2t, cslsacc_s2t)
    BLI_accuracy_l22l1 = (acc_t2s, cslsacc_t2s)
    return (BLI_accuracy_l12l2, BLI_accuracy_l22l1), nn_predict_s2t, nn_score_s2t, nn_predict_t2s, nn_score_t2s, csls_predict_s2t, csls_score_s2t, csls_predict_t2s, csls_score_t2s, s2t_s, s2t_t, t2s_t, t2s_s



def csls_values(src_, tgt_, k=10, bsz=256):
    src_ = src_.cuda()
    tgt_ = tgt_.cuda()
    src = src_ / (torch.norm(src_, dim=1, keepdim=True) + 1e-9)
    tgt = tgt_ / (torch.norm(tgt_, dim=1, keepdim=True) + 1e-9)

    src_hubness = torch.zeros(src.size(0))
    tgt_hubness = torch.zeros(tgt.size(0))

    for i in range(0, tgt.size(0), bsz):
        j = min(i + bsz, tgt.size(0))
        sc_batch = torch.matmul(tgt[i:j,:], src.T)
        dotprod = torch.topk(sc_batch,k=k,dim=1,sorted=False).values
        tgt_hubness[i:j] = torch.mean(dotprod, dim=1)

    for i in range(0, src.size(0), bsz):
        j = min(i + bsz, src.size(0))
        sc_batch = torch.matmul(src[i:j,:], tgt.T)
        dotprod = torch.topk(sc_batch,k=k,dim=1,sorted=False).values
        src_hubness[i:j] = torch.mean(dotprod, dim=1)

    return src_hubness, tgt_hubness



def idx(words):
    w2i = {}
    for i, w in enumerate(words):
        if w not in w2i:
            w2i[w] = i
    return w2i

def load_lexicon_s2t(filename, words_src, words_tgt, verbose=True):
    f = io.open(filename, 'r', encoding='utf-8')
    lexicon = collections.defaultdict(set)
    idx_src , idx_tgt = idx(words_src), idx(words_tgt)
    vocab = set()
    for line in f:
        word_src, word_tgt = line.split()
        word_src, word_tgt = word_src.lower(), word_tgt.lower()
        if word_src in idx_src and word_tgt in idx_tgt:
            lexicon[idx_src[word_src]].add(idx_tgt[word_tgt])
        vocab.add(word_src)
    if verbose:
        coverage = len(lexicon) / float(len(vocab))
        print("Coverage of source vocab: %.4f" % (coverage))
    return lexicon, float(len(vocab))

def load_lexicon_t2s(filename, words_src, words_tgt, verbose=True):
    f = io.open(filename, 'r', encoding='utf-8')
    lexicon = collections.defaultdict(set)
    idx_src , idx_tgt = idx(words_src), idx(words_tgt)
    vocab = set()
    for line in f:
        word_tgt, word_src = line.split()
        word_tgt, word_src = word_tgt.lower(), word_src.lower()
        if word_src in idx_src and word_tgt in idx_tgt:
            lexicon[idx_src[word_src]].add(idx_tgt[word_tgt])
        vocab.add(word_src)
    if verbose:
        coverage = len(lexicon) / float(len(vocab))
        print("Coverage of source vocab: %.4f" % (coverage))
    return lexicon, float(len(vocab))




def rerank(cross_encoder, coefs, predict_s2t, scores_, lexicon_size, id2w_src, id2w_tgt, s_ids, t_ids, use_template, l1, l2, k=28, temp_="{} ({})"):

    scores = coefs[0] * scores_ + coefs[1]
    silvers = 0.0 * scores
    for i in range(len(s_ids)):
        s_word_id = s_ids[i]
        predict_word_id = predict_s2t[i]
        s_word_txt = id2w_src[s_word_id]
        predict_word_txt = [id2w_tgt[id] for id in predict_word_id]

        input_pairs = list(itertools.product([s_word_txt],predict_word_txt))
        input_pairs_inv = list(itertools.product(predict_word_txt,[s_word_txt]))
        bg_string = "български"
        ca_string = "català"
        he_string = "עברית"
        et_string = "eesti"
        hu_string = "magyar"
        ka_string = "ქართული"
        str2lang = {"hr":"hrvatski", "en":"english","fi":"suomi","fr":"français","de":"deutsch","it":"italiano","ru":"русский","tr":"türkçe","bg":bg_string,"ca":ca_string,"he":he_string,"et":et_string,"hu":hu_string,"ka":ka_string}
        my_template = temp_ 
 
        if my_template.count("{}") == 1:
            for k in str2lang.keys():
                str2lang[k] = ""        
            my_template += " {}"
        if 'q1' in my_template:
            my_template = my_template.replace("q1","`")
        if 'q2' in my_template:
            my_template = my_template.replace("q2","'")

        input_pairs_template = [(my_template.format(p[0],str2lang[l1]).strip(), my_template.format(p[1],str2lang[l2]).strip()) for p in input_pairs]
        input_pairs_inv_template = [(my_template.format(p[0],str2lang[l1]).strip(), my_template.format(p[1],str2lang[l2]).strip()) for p in input_pairs_inv]

        with torch.no_grad():
            if use_template:
                silver_scores = cross_encoder.predict(input_pairs_template)
                silver_scores_inv = cross_encoder.predict(input_pairs_inv_template)
            else:
                silver_scores = cross_encoder.predict(input_pairs)
                silver_scores_inv = cross_encoder.predict(input_pairs_inv)
        assert len(silver_scores) == len(silver_scores_inv)
        silver_scores = [ 0.5 * (silver_scores[i] + silver_scores_inv[i]) for i in range(len(silver_scores))]
        silvers[i] = torch.tensor(silver_scores)
    
    for lambda_ in [i/100 for i in range(101)]:   
        
        new_scores = scores * (1 - lambda_) + silvers * lambda_
        new_predicts = torch.argmax(new_scores,dim=1).cpu().tolist()
        new_predicts =[predict_s2t[i][idx] for i,idx in enumerate(new_predicts)]
        accuracy = 0
        for i in range(len(s_ids)):
            if new_predicts[i] == t_ids[i]:
                accuracy += 1
        accuracy = accuracy / float(lexicon_size)
        print(lambda_, accuracy)
    return accuracy



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BLICEr Eval')

    parser.add_argument("--l1", type=str, default=" ",
                    help="l1")
    parser.add_argument("--l2", type=str, default=" ",
                    help="l2")
    parser.add_argument("--num_neg", type=int, default=10,
                    help="num_neg")
    parser.add_argument('--l1_voc', type=str, required=True,
                        help='Directory of L1 Vocabulary')
    parser.add_argument('--l1_emb', type=str, required=True,
                        help='Directory of Aligned Static Embeddings for L1')
    parser.add_argument('--l2_voc', type=str, required=True,
                        help='Directory of L2 Vocabulary')
    parser.add_argument('--l2_emb', type=str, required=True,
                        help='Directory of Aligned Static Embeddings for L2')
    parser.add_argument("--train_dict_dir", type=str, default="./",
                    help="train dict directory")
    parser.add_argument("--test_dict_dir", type=str, default="./",
                    help="test dict directory")
    parser.add_argument("--custom_corpus_path", type=str, default="./",
                    help="custom_corpus_path")
    parser.add_argument("--model_name", type=str, default="./",
                    help="model name")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--use_template", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=0)
    parser.add_argument("--my_template", type=str, default='{} ({})')
    args, remaining_args = parser.parse_known_args()
    args.use_template = bool(args.use_template)
    assert remaining_args == []
    print("Generate Neg Samples")
    sys.stdout.flush()
    l1_voc = args.l1_voc
    l1_emb = args.l1_emb
    l2_voc = args.l2_voc
    l2_emb = args.l2_emb
    DIR_TEST_DICT = args.test_dict_dir

    l1_voc = np.load(l1_voc, allow_pickle=True).item()
    l2_voc = np.load(l2_voc, allow_pickle=True).item()
    l1_emb = torch.load(l1_emb)
    l2_emb = torch.load(l2_emb)

    l1_emb = l1_emb / (torch.norm(l1_emb, dim=1, keepdim=True) + 1e-9 )
    l2_emb = l2_emb / (torch.norm(l2_emb, dim=1, keepdim=True) + 1e-9 )

    words_src = list(l1_voc.keys())
    words_tgt = list(l2_voc.keys())
    src_hubness, tgt_hubness = csls_values(l1_emb, l2_emb, k=10, bsz=256)

    # Test Set
    src2tgt, lexicon_size_s2t = load_lexicon_s2t(DIR_TEST_DICT, words_src, words_tgt)
    tgt2src, lexicon_size_t2s = load_lexicon_t2s(DIR_TEST_DICT, words_tgt, words_src)
    print("lexicon_size_s2t, lexicon_size_t2s", lexicon_size_s2t, lexicon_size_t2s)

    if l1_emb.size(1) < 99999: 
        accuracy_BLI, nn_predict_s2t, nn_score_s2t, nn_predict_t2s, nn_score_t2s, csls_predict_s2t, csls_score_s2t, csls_predict_t2s, csls_score_t2s, s2t_s, s2t_t, t2s_t, t2s_s = eval_BLI(l1_emb, l2_emb, src2tgt, lexicon_size_s2t, tgt2src, lexicon_size_t2s, src_hubness, tgt_hubness, 1.0, k=args.num_neg)
        print("CLWEs: ", "BLI Accuracy L1 to L2: ", accuracy_BLI[0], "BLI Accuracy L2 to L1: ", accuracy_BLI[1])
        sys.stdout.flush()

    use_template = args.use_template

    if use_template:
        max_len_cross = args.max_length #32
    else:
        max_len_cross = args.max_length #18

    coefs = torch.load(args.custom_corpus_path + "coefs.pt")
    cross_encoder = CrossEncoder(args.model_name, max_length=max_len_cross, device=args.device)

    voc_l1_id2word = {v:k for k,v in l1_voc.items()}
    voc_l2_id2word = {v:k for k,v in l2_voc.items()}

    
    print("s2t test")
    csls_s2t = rerank(cross_encoder, coefs, csls_predict_s2t, csls_score_s2t, lexicon_size_s2t, voc_l1_id2word, voc_l2_id2word, s2t_s, s2t_t, use_template, args.l1, args.l2, k=args.num_neg,temp_=args.my_template)
    print("t2s test")
    csls_t2s = rerank(cross_encoder, coefs, csls_predict_t2s, csls_score_t2s, lexicon_size_t2s, voc_l2_id2word, voc_l1_id2word, t2s_t, t2s_s, use_template, args.l2, args.l1, k=args.num_neg,temp_=args.my_template)


