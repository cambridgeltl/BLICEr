import itertools
from sentence_transformers.readers import InputExample
import os
import random
import torch 
 

def load_custom(fpath,num_dup,temp_):
    l1, l2 = fpath.split("/")[-1].split("_")[0].split("2")
    bg_string = "български"
    ca_string = "català"
    he_string = "עברית"
    et_string = "eesti"
    hu_string = "magyar"
    ka_string = "ქართული"    
    str2lang = {"hr":"hrvatski", "en":"english","fi":"suomi","fr":"français","de":"deutsch","it":"italiano","ru":"русский","tr":"türkçe","bg":bg_string,"ca":ca_string,"he":he_string,"et":et_string,"hu":hu_string,"ka":ka_string}
    #str2lang = {"hr":"croatian", "en":"english","fi":"finnish","fr":"french","de":"german","it":"italian","ru":"russian","tr":"turkish"}
    my_template = temp_ #"{} ({})" #"the word '{}' in {}."

    if 'q1' in my_template:
        my_template = my_template.replace("q1","`")
    if 'q2' in my_template:
        my_template = my_template.replace("q2","'")
    if my_template.count("{}") == 1:
        for k in str2lang.keys():
            str2lang[k] = ""
        my_template += " {}"

    pos_pairs = torch.load(fpath + "pos_pairs.pt")
    neg_pairs = torch.load(fpath + "neg_pairs.pt")


    len_pos_pairs = len(pos_pairs)

    all_pairs = []
    cosine_scores = []
    
    for k, v in pos_pairs.items():
        all_pairs.append(k)
        cosine_scores.append(v.item())

    for k, v in neg_pairs.items():
        all_pairs.append(k)
        cosine_scores.append(v.item())    
    all_pairs_template = [(my_template.format(p[0],str2lang[l1]).strip(), my_template.format(p[1],str2lang[l2]).strip()) for p in all_pairs]
    print(all_pairs_template[0])
    random.seed(33)
    neg_pairs_sample = random.sample(neg_pairs.keys(), 7000)

    dev_samples = []
    dev_samples_template = []

    for pairs in pos_pairs.keys():
        dev_samples.append(InputExample(texts=[pairs[0],pairs[1]], label=1))
        dev_samples_template.append(InputExample(texts=[my_template.format(pairs[0],str2lang[l1]).strip(), my_template.format(pairs[1],str2lang[l2]).strip()], label=1))
        dev_samples.append(InputExample(texts=[pairs[1],pairs[0]], label=1))
        dev_samples_template.append(InputExample(texts=[my_template.format(pairs[1],str2lang[l2]).strip(), my_template.format(pairs[0],str2lang[l1]).strip()], label=1))

    for pairs in neg_pairs_sample:
        dev_samples.append(InputExample(texts=[pairs[0],pairs[1]], label=0))
        dev_samples_template.append(InputExample(texts=[my_template.format(pairs[0],str2lang[l1]).strip(), my_template.format(pairs[1],str2lang[l2]).strip()], label=0))
        dev_samples.append(InputExample(texts=[pairs[1],pairs[0]], label=0))
        dev_samples_template.append(InputExample(texts=[my_template.format(pairs[1],str2lang[l2]).strip(), my_template.format(pairs[0],str2lang[l1]).strip()], label=1))

    test_samples = {}
    test_samples["BLI"] = dev_samples

    test_samples_template = {}
    test_samples_template["BLI"] = dev_samples_template

    all_pairs = all_pairs[:len_pos_pairs] * num_dup + all_pairs
    all_pairs_template = all_pairs_template[:len_pos_pairs] * num_dup + all_pairs_template
    cosine_scores = cosine_scores[:len_pos_pairs] * num_dup + cosine_scores

    return all_pairs, test_samples, dev_samples, all_pairs_template, test_samples_template, dev_samples_template, torch.tensor(cosine_scores), len(pos_pairs)


task_loader_dict = {
    "custom": load_custom
}

def load_data(task, fpath=None, num_dup=0, temp_=None):
    if task not in task_loader_dict.keys():
        raise NotImplementedError()
    if task == "custom":
        return task_loader_dict[task](fpath,num_dup,temp_)
    else:
        return task_loader_dict[task]()


if __name__ == "__main__":
    for task in task_loader_dict:
        print (f"loading {task}...")
        load_data(task)
        print ("done.")
