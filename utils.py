import pprint
import codecs
import os
import sys
import time
import pickle as pkl
import numpy as np
from collections import OrderedDict
import io
import collections

def load_embs(path, topk = None, dimension = None):
  print(topk)
  print("Loading embeddings")
  vocab_dict = {}
  embeddings = []
  with codecs.open(path, encoding = 'utf8', errors = 'replace') as f:
      line = f.readline().strip().split()
      cntr = 1
      if len(line) == 2:
        vocab_size = int(line[0])
        if not dimension:
          dimension = int(line[1])
      else:
        if not dimension or (dimension and len(line[1:]) == dimension):
          vocab_dict[line[0].strip()] = len(vocab_dict)
          embeddings.append(np.array(line[1:], dtype=np.float32))
        if not dimension:
          dimension = len(line) - 1
      print("Vector dimensions: " + str(dimension))
      while line:
        line = f.readline().strip().split()
        if (not line):
          print("Loaded " + str(cntr) + " vectors.")
          break

        if line[0].strip() == "":
          continue

        cntr += 1
        if cntr % 20000 == 0:
          print(cntr)

        if len(line[1:]) == dimension:
          if (line[0].strip().lower() not in vocab_dict):
              vocab_dict[line[0].strip().lower()] = len(vocab_dict)
              embeddings.append(np.array(line[1:], dtype=np.float32))
        else:
          print("Error in the embeddings file, line " + str(cntr) +
                             ": unexpected vector length (expected " + str(dimension) +
                             " got " + str(len(np.array(line[1:]))) + " for word '" + line[0] + "'")

        if (topk and cntr >= topk):
          print("Loaded " + str(cntr) + " vectors.")
          break

  embeddings = np.array(embeddings, dtype=np.float32)
  print(len(vocab_dict), str(embeddings.shape))
  return vocab_dict, embeddings
