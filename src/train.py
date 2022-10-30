import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sentence_transformers import models, losses, util, SentenceTransformer, LoggingHandler
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sentence_transformers.readers import InputExample
from datetime import datetime
import argparse
import logging 
import sys
import random
import tqdm
import math
import os
import numpy as np

from data import load_data

def polarise_minus(x_,a=0.6):
    x = torch.minimum(torch.tensor(1.0),torch.maximum(torch.tensor(0.0),x_))
    y = a * x
    return y

def polarise_plus(x_,a=0.6):
    x = torch.minimum(torch.tensor(1.0),torch.maximum(torch.tensor(0.0),x_))
    y = 1.0 - a + a*x
    return y


parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, 
        default=None,
        help="Transformers' model name or path")
parser.add_argument("--task", type=str, default='sts')
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--num_epochs_cross_encoder", type=int, default=1)
parser.add_argument("--batch_size_cross_encoder", type=int, default=32)
parser.add_argument("--init_with_new_models", action="store_true")
parser.add_argument("--random_seed", type=int, default=2021)
parser.add_argument("--add_snli_data", type=int, default=0)
parser.add_argument("--custom_corpus_path", type=str, default=None)
parser.add_argument("--num_training_pairs", type=int, default=None)
parser.add_argument("--save_all_predictions", action="store_true")
parser.add_argument("--quick_test", action="store_true")
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--origin_model_dir", type=str, default=None)
parser.add_argument("--use_template", type=int, default=0)
parser.add_argument("--max_length", type=int, default=0)
parser.add_argument("--num_dup", type=int, default=8)
parser.add_argument("--alpha", type=float, default=0.7)
parser.add_argument("--my_template", type=str, default='{} ({})')
args = parser.parse_args()
args.use_template = bool(args.use_template)
print (args)

torch.manual_seed(args.random_seed)

logging.basicConfig(format="%(asctime)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


### read datasets
all_pairs, all_test, dev_samples, all_pairs_template, all_test_template, dev_samples_template, cosine_scores, len_pos_pairs = load_data(args.task, fpath=args.custom_corpus_path, num_dup=args.num_dup, temp_=args.my_template)
all_pairs_inv = [(p[1],p[0]) for p in all_pairs]
all_pairs_template_inv = [(p[1],p[0]) for p in all_pairs_template]
# load_pairs from other tasks

cosine_scores = torch.maximum(torch.minimum(cosine_scores, torch.tensor(1.0)),torch.tensor(0.0))

print ("|raw sentence pairs|:", len(all_pairs))
print ("|dev set|:", len(dev_samples))
for key in all_test:
    print ("|test set: %s|" % key, len(all_test[key]))
 
batch_size_cross_encoder = args.batch_size_cross_encoder
num_epochs_cross_encoder = args.num_epochs_cross_encoder 
device=args.device
l1, l2 = args.custom_corpus_path.split("/")[-1].split("_")[0].split("2")

model_name = args.origin_model_dir

logging.info(f"Loading Cross-Encoder Model: {model_name}")
start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

cross_encoder_dev_scores = []
all_predictions_cross_encoder = {}


use_template_cross = args.use_template

if use_template_cross:
    print("USING TEMPLATE")
    max_len_cross = args.max_length
else:
    max_len_cross = args.max_length

if True:
    logging.info ("Polarise Cross-Lingual Word Similarities Derived from CLWEs")

    sents1_template = [p[0] for p in all_pairs_template]
    sents2_template = [p[1] for p in all_pairs_template]        
    
    sents1 = [p[0] for p in all_pairs]
    sents2 = [p[1] for p in all_pairs]

    #Polarize
    GROUD_TRUTH_NUM = len_pos_pairs * (1 + args.num_dup)


    print("BEFORE POLARISATION: ")

    print("POS PAIRS:", cosine_scores[:GROUD_TRUTH_NUM].max().item(), cosine_scores[:GROUD_TRUTH_NUM].min().item(), cosine_scores[:GROUD_TRUTH_NUM].mean().item())
    print("NEG PAIRS:", cosine_scores[GROUD_TRUTH_NUM:].max().item(),cosine_scores[GROUD_TRUTH_NUM:].min().item(),cosine_scores[GROUD_TRUTH_NUM:].mean().item())
    print("ALL:", cosine_scores.max().item(), cosine_scores.min().item(), cosine_scores.mean().item())


    a_val = args.alpha
    print("a_val: ", a_val)
    cosine_scores[:GROUD_TRUTH_NUM] = polarise_plus(cosine_scores[:GROUD_TRUTH_NUM], a = a_val)
    cosine_scores[GROUD_TRUTH_NUM:] = polarise_minus(cosine_scores[GROUD_TRUTH_NUM:], a = a_val)


    print("After POLARISATION: ")

    print("POS PAIRS:", cosine_scores[:GROUD_TRUTH_NUM].max().item(), cosine_scores[:GROUD_TRUTH_NUM].min().item(), cosine_scores[:GROUD_TRUTH_NUM].mean().item())
    print("NEG PAIRS:", cosine_scores[GROUD_TRUTH_NUM:].max().item(),cosine_scores[GROUD_TRUTH_NUM:].min().item(),cosine_scores[GROUD_TRUTH_NUM:].mean().item())
    print("ALL:", cosine_scores.max().item(), cosine_scores.min().item(), cosine_scores.mean().item())



    train_samples = []

    for i in range(len(sents1)):
        if use_template_cross:
            train_samples.append(InputExample(texts=[sents1_template[i], sents2_template[i]], label=cosine_scores[i]))
            train_samples.append(InputExample(texts=[sents2_template[i], sents1_template[i]], label=cosine_scores[i]))
        else:
            train_samples.append(InputExample(texts=[sents1[i], sents2[i]], label=cosine_scores[i]))
            train_samples.append(InputExample(texts=[sents2[i], sents1[i]], label=cosine_scores[i]))
    del cosine_scores
    torch.cuda.empty_cache()

    ###### Cross-encoder learning ######
    if args.init_with_new_models:
        cross_encoder_path = args.origin_model_dir 
    logging.info(f"Loading Cross-Encoder Model: {cross_encoder_path}")
    cross_encoder = CrossEncoder(cross_encoder_path, num_labels=1, device=device, max_length=max_len_cross)

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size_cross_encoder)

    if use_template_cross:
        evaluator = CECorrelationEvaluator.from_input_examples(dev_samples_template, name='dev')
    else:
        evaluator = CECorrelationEvaluator.from_input_examples(dev_samples, name='dev')

    warmup_steps = math.ceil(len(train_dataloader) * num_epochs_cross_encoder * 0.1) 
    logging.info(f"Warmup-steps: {warmup_steps}")

    cross_encoder_path = args.output_dir 

    # Train the cross-encoder model
    cross_encoder.fit(
            train_dataloader=train_dataloader,
            evaluator=evaluator,
            evaluation_steps=-1,
            use_amp=True,
            epochs=num_epochs_cross_encoder,
            warmup_steps=warmup_steps,
            optimizer_params= {"lr": 1.2e-5},
            output_path=cross_encoder_path)

    cross_encoder = CrossEncoder(cross_encoder_path, max_length=max_len_cross, device=device)



    dev_score = evaluator(cross_encoder)
    cross_encoder_dev_scores.append(dev_score)
    logging.info (f"***** dev's spearman's rho: {dev_score:.4f} *****")

    logging.info ("Label sentence pairs with cross-encoder...")

 
    
logging.info ("\n")
print (args)
logging.info ("\n")
logging.info ("***** END *****")
