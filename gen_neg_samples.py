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

def eval_BLI(train_data_l1, train_data_l2, src2tgt, lexicon_size_s2t, tgt2src, lexicon_size_t2s, src_hubness, tgt_hubness, best_t):

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


    _, _, acc_s2t = getknn_test(train_data_l1_translation[s2t_s], train_data_l2_translation, s2t_s, s2t_t, lexicon_size_s2t, k=28, bsz=1024)
    _, _, acc_t2s = getknn_test(train_data_l2_translation[t2s_t], train_data_l1_translation, t2s_t, t2s_s, lexicon_size_t2s, k=28, bsz=1024)

    _, _, cslsacc_s2t = getknn_csls_test(train_data_l1_translation[s2t_s], train_data_l2_translation, s2t_s, s2t_t, src_hubness, tgt_hubness, lexicon_size_s2t, k=28, bsz=1024, t=best_t)
    _, _, cslsacc_t2s = getknn_csls_test(train_data_l2_translation[t2s_t], train_data_l1_translation, t2s_t, t2s_s, tgt_hubness, src_hubness, lexicon_size_t2s, k=28, bsz=1024, t=best_t)

    BLI_accuracy_l12l2 = (acc_s2t, cslsacc_s2t)
    BLI_accuracy_l22l1 = (acc_t2s, cslsacc_t2s)
    return (BLI_accuracy_l12l2, BLI_accuracy_l22l1)


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

def getknn_csls(src_, tgt_, src_ids, tgt_ids, src_hubness_, tgt_hubness_, lexicon_size, k=10, bsz=1024, t=1.0):
    #k: num of neg samples
    #k_csls: usually 10
    src_ = src_.cuda()
    tgt_ = tgt_.cuda()
    src = src_ / (torch.norm(src_, dim=1, keepdim=True) + 1e-9)
    tgt = tgt_ / (torch.norm(tgt_, dim=1, keepdim=True) + 1e-9)
    num_imgs = len(src)
    confuse_output_indices = []
    confuse_output_indices_long = []
    scores = torch.zeros(num_imgs,k+1)

    src_hubness, tgt_hubness = src_hubness_.cuda(), tgt_hubness_.cuda()

    src_hubness_sup = src_hubness[src_ids]
    tgt_hubness_sup = tgt_hubness[tgt_ids]
    for batch_idx in range( int( math.ceil( float(num_imgs) / bsz ) ) ):
        start_idx = batch_idx * bsz
        end_idx = min( num_imgs, (batch_idx + 1) * bsz )
        length = end_idx - start_idx
        prod_batch = (1.0 + t) * torch.matmul(src[start_idx:end_idx, :], tgt.T) - t * src_hubness_sup[start_idx:end_idx].unsqueeze(1) - t * tgt_hubness.unsqueeze(0)
        dotprod = torch.topk(prod_batch,k=k+1,dim=1,sorted=True,largest=True).indices
        confuse_output_indices_long += dotprod.cpu().tolist()

    for i in range(len(confuse_output_indices_long)):
        confuse_output_i = confuse_output_indices_long[i]
        if tgt_ids[i] in confuse_output_i:
            confuse_output_i_new = confuse_output_i.copy()
            confuse_output_i_new.remove(tgt_ids[i])
            confuse_output_indices.append(confuse_output_i_new)
        else:
            confuse_output_indices.append(confuse_output_i[:-1])

    assert len(confuse_output_indices) == num_imgs

    for i in range(num_imgs):
        src_embs = src[[i]]
        tgt_embs = tgt[[tgt_ids[i]]+confuse_output_indices[i]]
        scores[i] = (1.0 + t) * src_embs @ tgt_embs.T - t * src_hubness_sup[i:i+1].unsqueeze(1) - t * tgt_hubness[[tgt_ids[i]]+confuse_output_indices[i]].unsqueeze(0)

    accuracy = (torch.max(scores,dim=1).values == scores[:,0]).sum().item() / float(lexicon_size)
    return confuse_output_indices, scores, accuracy



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


def SAVE_DATA(args, train_data_l1, train_data_l2, l1_idx_sup, l2_idx_sup, voc_l1, voc_l2, src_hubness, tgt_hubness, t=1.0):
    num_imgs_l1 = len(train_data_l1)
    num_imgs_l2 = len(train_data_l2)
    train_data_l1_translation = train_data_l1 
    train_data_l2_translation = train_data_l2 

    sup_data_l1_translation = torch.index_select(train_data_l1_translation,0,torch.tensor(l1_idx_sup))
    sup_data_l2_translation = torch.index_select(train_data_l2_translation,0,torch.tensor(l2_idx_sup))

    voc_l1_id2word = {v:k for k,v in voc_l1.items()} 
    voc_l2_id2word = {v:k for k,v in voc_l2.items()} 

    neg_sample = args.num_neg

    src, tgt = l1_idx_sup, l2_idx_sup
    lexicon_size_s2t = len(l1_idx_sup)
    lexicon_size_t2s = len(l2_idx_sup)
 
    confuse_tgt, scores_s2t, cslsacc_s2t = getknn_csls(sup_data_l1_translation, train_data_l2_translation[:], src, tgt, src_hubness, tgt_hubness[:], lexicon_size_s2t, k=neg_sample, bsz=1024, t=t)
    confuse_src, scores_t2s, cslsacc_t2s = getknn_csls(sup_data_l2_translation, train_data_l1_translation[:], tgt, src, tgt_hubness, src_hubness[:], lexicon_size_t2s, k=neg_sample, bsz=1024, t=t)

    scores = torch.zeros(len(sup_data_l1_translation),neg_sample*2+1)
    with open(args.root + "{}2{}_train.txt".format(args.l1, args.l2)  ,"w") as f:
        for i in range(len(src)):
            l1_word = src[i]
            l2_word = tgt[i]
            l2_conf = confuse_tgt[i]
            l1_conf = confuse_src[i]
            l1_words = [l1_word] + l1_conf
            l2_words = [l2_word] + l2_conf
            l1_words = [voc_l1_id2word[idx] for idx in l1_words]
            l2_words = [voc_l2_id2word[idx] for idx in l2_words]
            l1_words = " ".join(l1_words)
            l2_words = " ".join(l2_words)
            line = str(i)+"|+|"+l1_words+"|+|"+l2_words
            f.write(line+"\n")
            scores[i, :1+neg_sample] = scores_s2t[i]
            scores[i, 1+neg_sample:] = scores_t2s[i,1:]

    torch.save(scores, args.root + "{}2{}_scores.pt".format(args.l1, args.l2))



    if True:
        max_ = scores.max()
        min_ = scores.min()
        max_aim = 1.0
        min_aim = 0.0
        print("max_aim, min_aim", max_aim, min_aim)
        a_coef = (max_aim - min_aim) / (max_ - min_)
        b_coef = max_aim - a_coef * max_
        scores = a_coef * scores + b_coef
        torch.save([a_coef,b_coef],args.root + "{}2{}_coefs.pt".format(args.l1, args.l2))
    else:
        a_coef, b_coef = 1.0, 0.0
        torch.save([a_coef,b_coef],args.root + "{}2{}_coefs.pt".format(args.l1, args.l2))

    pos_pairs = {}
    neg_pairs = {}
    neg_pairs_final = {}
    if True:
        for i in range(len(l1_idx_sup)):
            l1_word = l1_idx_sup[i]
            l2_word = l2_idx_sup[i]
            l2_conf = confuse_tgt[i]
            l1_conf = confuse_src[i]

            score = scores[i]

            delta = args.delta
            l2_conf_cut = (score[1:neg_sample+1] >= (score[0] - delta)).sum().item()
            l1_conf_cut = (score[neg_sample+1:] >= (score[0] - delta)).sum().item()
            l2_conf = l2_conf[:l2_conf_cut]
            l1_conf = l1_conf[:l1_conf_cut]

            l1_words = [l1_word] + l1_conf
            l2_words = [l2_word] + l2_conf
            l1_words = [voc_l1_id2word[idx] for idx in l1_words]
            l2_words = [voc_l2_id2word[idx] for idx in l2_words]

            pos_pairs[(l1_words[0], l2_words[0])] = score[0]
            for j,w in enumerate(l2_words[1:]):
                neg_pairs[(l1_words[0], w)] = score[j+1]

            for j,w in enumerate(l1_words[1:]):
                neg_pairs[(w, l2_words[0])] = score[1+neg_sample+j]
    for k in neg_pairs.keys():
        if k not in pos_pairs:
            neg_pairs_final[k] = neg_pairs[k]
    print(len(pos_pairs),len(neg_pairs), len(neg_pairs_final))
    torch.save(pos_pairs, args.root + "{}2{}_pos_pairs.pt".format(args.l1, args.l2))
    torch.save(neg_pairs_final, args.root + "{}2{}_neg_pairs.pt".format(args.l1, args.l2))


def high_conf_pairs(args, train_data_l1, train_data_l2, l1_idx_sup, l2_idx_sup):
    num_imgs_l1 = len(train_data_l1)
    num_imgs_l2 = len(train_data_l2)


    train_data_l1_translation = train_data_l1.cuda()
    train_data_l2_translation = train_data_l2.cuda()

    l1_idx_aug, l2_idx_aug = generate_new_dictionary_bidirectional(args, train_data_l1_translation, train_data_l2_translation, l1_idx_sup, l2_idx_sup)


    return l1_idx_aug, l2_idx_aug

def get_nn_avg_dist(emb, query, knn):
    bs = 1024
    all_distances = []
    for i in range(0, query.shape[0], bs):
        distances = query[i:i + bs].mm(emb.T)
        best_distances, _ = distances.topk(knn, dim=1, largest=True, sorted=True)
        all_distances.append(best_distances.mean(1).cpu())
    all_distances = torch.cat(all_distances)
    return all_distances

def generate_new_dictionary_bidirectional(args, emb1_, emb2_, l1_idx_sup, l2_idx_sup):

    emb1 = emb1_ / (torch.norm(emb1_, dim=1, keepdim=True) + 1e-9) #.cuda()
    emb2 = emb2_ / (torch.norm(emb2_, dim=1, keepdim=True) + 1e-9)#.cuda()
    bs = 128
    all_scores_S2T = []
    all_targets_S2T = []
    all_scores_T2S = []
    all_targets_T2S = []
    n_src = args.dico_max_rank
    knn = 10

    average_dist1 = get_nn_avg_dist(emb2, emb1, knn) 
    average_dist2 = get_nn_avg_dist(emb1, emb2, knn) 
    average_dist1 = average_dist1.type_as(emb1)
    average_dist2 = average_dist2.type_as(emb2)

    ## emb1 to emb2
    for i in range(0, n_src, bs):
        scores = emb2.mm(emb1[i:min(n_src, i + bs)].transpose(0, 1)).transpose(0, 1)
        scores.mul_(2)
        scores.sub_(average_dist2[None, :])
        best_scores, best_targets = scores.topk(1, dim=1, largest=True, sorted=True)

        all_scores_S2T.append(best_scores.cpu())
        all_targets_S2T.append(best_targets.cpu())

    all_scores_S2T = torch.cat(all_scores_S2T, 0).squeeze(1).tolist()
    all_targets_S2T = torch.cat(all_targets_S2T, 0).squeeze(1).tolist()

    pairs_S2T = [(i, all_targets_S2T[i], all_scores_S2T[i]) for i in range(len(all_scores_S2T))]

    # emb2 to emb1
    for i in range(0, n_src, bs):
        scores = emb1.mm(emb2[i:min(n_src, i + bs)].transpose(0, 1)).transpose(0, 1)
        scores.mul_(2)
        scores.sub_(average_dist1[None, :])
        best_scores, best_targets = scores.topk(1, dim=1, largest=True, sorted=True)

        all_scores_T2S.append(best_scores.cpu())
        all_targets_T2S.append(best_targets.cpu())

    all_scores_T2S = torch.cat(all_scores_T2S, 0).squeeze(1).tolist()
    all_targets_T2S = torch.cat(all_targets_T2S, 0).squeeze(1).tolist()

    pairs_T2S = [(all_targets_T2S[i], i, all_scores_T2S[i]) for i in range(len(all_scores_T2S))]

    pairs_S2T = sorted(pairs_S2T,key=lambda x:x[-1],reverse=True)[:args.num_aug]
    pairs_T2S = sorted(pairs_T2S,key=lambda x:x[-1],reverse=True)[:args.num_aug]

    final_pairs = set()

    S_set = set(l1_idx_sup)
    T_Set = set(l2_idx_sup)

    for i in range(len(pairs_S2T )):

        if (pairs_S2T[i][0] not in S_set) and (pairs_S2T[i][1] not in T_Set) and (len(final_pairs) < args.num_aug_total):
            final_pairs.add((pairs_S2T[i][0], pairs_S2T[i][1]))

        if (pairs_T2S[i][0] not in S_set) and (pairs_T2S[i][1] not in T_Set) and (len(final_pairs) < args.num_aug_total):
            final_pairs.add((pairs_T2S[i][0], pairs_T2S[i][1]))



    final_pairs = list(final_pairs)

    if len(final_pairs) > 0:
        final_s_aug = [a for (a,b) in final_pairs]
        final_t_aug = [b for (a,b) in final_pairs]
    else:
        final_s_aug, final_t_aug = [], []


    return final_s_aug, final_t_aug    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BLICEr GEN POS-NEG SAMPLES')

    parser.add_argument("--l1", type=str, default=" ",
                    help="l1")
    parser.add_argument("--l2", type=str, default=" ",
                    help="l2")
    parser.add_argument("--num_iter", type=int, default=1,
                    help="num of iterations")
    parser.add_argument("--train_size", type=str, default="5k",
                    help="train dict size")
    parser.add_argument("--root", type=str, default="./",
                    help="save root")
    parser.add_argument("--dico_max_rank", type=int, default=20000,
                    help="dico max rank")
    parser.add_argument("--num_aug", type=int, default=6000,
                    help="num_aug")
    parser.add_argument("--num_neg", type=int, default=10,
                    help="num_neg")
    parser.add_argument("--num_aug_total", type=int, default=4000,
                    help="num_aug_total")
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
    parser.add_argument("--delta", type=float, default=0.1,
                    help="delta")
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == []
    print("Generate Neg Samples")
    sys.stdout.flush()
    l1_voc = args.l1_voc
    l1_emb = args.l1_emb
    l2_voc = args.l2_voc
    l2_emb = args.l2_emb
    DIR_TEST_DICT = args.test_dict_dir
    DIR_TRAIN_DICT = args.train_dict_dir


    l1_voc = np.load(l1_voc, allow_pickle=True).item()
    l2_voc = np.load(l2_voc, allow_pickle=True).item()
    l1_emb = torch.load(l1_emb)
    l2_emb = torch.load(l2_emb)

    l1_emb = l1_emb / (torch.norm(l1_emb, dim=1, keepdim=True) + 1e-9 )
    l2_emb = l2_emb / (torch.norm(l2_emb, dim=1, keepdim=True) + 1e-9 )

    words_src = list(l1_voc.keys())
    words_tgt = list(l2_voc.keys())

    src2tgt, lexicon_size_s2t = load_lexicon_s2t(DIR_TEST_DICT, words_src, words_tgt)
    tgt2src, lexicon_size_t2s = load_lexicon_t2s(DIR_TEST_DICT, words_tgt, words_src)
    print("lexicon_size_s2t, lexicon_size_t2s", lexicon_size_s2t, lexicon_size_t2s)


    #Load Train

    file = open(DIR_TRAIN_DICT,'r')
    l1_dic = []
    l2_dic = []
    for line in file.readlines():
        pair = line[:-1].split('\t')
        l1_dic.append(pair[0].lower())
        l2_dic.append(pair[1].lower())
    file.close()
    l1_idx_sup = []
    l2_idx_sup = []
    for i in range(len(l1_dic)):
        l1_tok = l1_voc.get(l1_dic[i])
        l2_tok = l2_voc.get(l2_dic[i])
        if (l1_tok is not None) and (l2_tok is not None):
            l1_idx_sup.append(l1_tok)
            l2_idx_sup.append(l2_tok)
    
    print("Sup Set Size: ", len(l1_idx_sup), len(l2_idx_sup))

    #Find High Conf Pairs 



    src_hubness, tgt_hubness = csls_values(l1_emb, l2_emb, k=10, bsz=256)
    if l1_emb.size(1) < 9999999:
        accuracy_BLI = eval_BLI(l1_emb, l2_emb, src2tgt, lexicon_size_s2t, tgt2src, lexicon_size_t2s, src_hubness, tgt_hubness, 1.0)
        print("CLWEs: ", "BLI Accuracy L1 to L2: ", accuracy_BLI[0], "BLI Accuracy L2 to L1: ", accuracy_BLI[1])
        sys.stdout.flush()


    if args.train_size == "1k":
        with torch.no_grad():
            l1_idx_aug, l2_idx_aug = high_conf_pairs(args, l1_emb, l2_emb, l1_idx_sup, l2_idx_sup)
            print("augment ", len(l1_idx_aug), " training pairs")
            sys.stdout.flush()
    else:
        l1_idx_aug, l2_idx_aug = [], []

    SAVE_DATA(args, l1_emb, l2_emb, l1_idx_sup+l1_idx_aug, l2_idx_sup+l2_idx_aug, l1_voc, l2_voc, src_hubness, tgt_hubness, 1.0)
    print("positive-negative pairs for contrastive tuning saved")


