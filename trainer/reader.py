from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import numpy as np
import tensorflow as tf
import math
Py3 = sys.version_info[0] == 3

def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    if Py3:
      return f.read().replace("\n", " ").split()
    else:
      return f.read().decode("utf-8").replace("\n", " ").split()

def id_to_word(arr):
  filename='C:\\Users\\t-tazha\\CNN_PPT\\microsoftPPTX.txt'
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  print(len(words))
  return [[words[i - 1] for i in row if i > 0] for row in arr]


def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data = []
  color_id = 0
  color_dict = {}
  with tf.gfile.GFile(filename, "r") as f:
    sentences = f.read().split("\n")
    for sentence in sentences:
        words = sentence.split()
        if(len(words) == 0):
          continue
        color = words[len(words) - 1]
        if color not in color_dict:
          color_dict[color] = color_id
          color_id += 1
        words.pop()
        line = [word_to_id[word] + 1 for word in words if word in word_to_id]
        line.append(color_dict[color])
        data.append(line)
  mx = 0;
  for line in data:
      mx = max(len(line), mx)
  data = [line for line in data if len(line) > 0]
  print(mx)
  l = int(math.sqrt(mx))
  if l * l < mx:
    l = l + 1
  for line in data:
      num = line[len(line) - 1]
      line.pop(len(line) - 1)
      while len(line) < l * l:
          line.insert(0, 0)
      line.extend([num])
  print(color_dict)

  return data, color_dict


def read_raw_data():

  train_path = os.path.join("", "processed_ppt.dat")
  word_to_id = _build_vocab(train_path)
  train_data, color_dict = _file_to_word_ids(train_path, word_to_id)
  data = np.asarray(train_data)
  print(data.shape)
  lenx, leny = data.shape
  x = data[:, 0 : leny -1]
  y = data[:, leny - 1 : leny]
  colors = open("../color_dict.dat", "w", encoding = 'utf-8')
  for color in color_dict:
    colors.write(color + "\n")
  colors.close()
  return x, y, len(color_dict)

x, y, _ = read_raw_data()

print(x.shape)
