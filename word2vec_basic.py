# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Basic word2vec example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import datetime
import math
import os
import random
import sys
from tempfile import gettempdir
import zipfile

import word2vec_eval
import util

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector
from gensim.models.phrases import Phrases, Phraser
from spacy.lang.en.stop_words import STOP_WORDS
import re



data_index = 0

def process_inputs(X, y):

    set_inputs = []
    set_labels = []

    for case in range(len(X)):
        set_inputs.append(X[case][0])
        set_inputs[-1] = set_inputs[-1].replace(' ', '_') 
        set_inputs[-1] = re.sub('_\d', '', set_inputs[-1])

        set_labels.append(y[case])
        set_labels[-1] = set_labels[-1].replace(' ', '_') 
        set_labels[-1] = re.sub('_\d', '', set_labels[-1])
    return set_inputs, set_labels

def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
      data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

def read_data_nonzip(filename):
    with open(filename, 'r') as f:
      data = f.read().split()
    return data


def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
      dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
      index = dictionary.get(word, 0)
      if index == 0:  # dictionary['UNK']
        unk_count += 1
      data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

def get_pretrain_dictionaries(filename):

    n_words = 200000
    data = read_data_nonzip(filename)
    return build_dataset(data, n_words)


def normalize_embeddings(embeddings):
    sess = tf.Session()
    with sess.as_default():
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
        normalized_embeddings = embeddings / norm

        return normalized_embeddings.eval()

def bigram_embedding_lookup(embeddings, train_inputs, reverse_dictionary):
    embed1 = tf.nn.embedding_lookup(embeddings, train_inputs[:, 0])
    embed2 = tf.nn.embedding_lookup(embeddings, train_inputs[:, 1])
    return (embed1 + embed2) / 2


def split_bigrams(batch_inputs, batch_labels, unused_dictionary, reverse_dictionary):
    modded_batch_inputs = []
    modded_batch_labels = []
    for i in range(len(batch_inputs)):
        w = reverse_dictionary[batch_inputs[i]]
        if '_' in w:
            w1, w2 = w.split("_")
        else:
            w1, w2 = w, w
        try:
            modded_batch_inputs.append((unused_dictionary[w1], unused_dictionary[w2]))
        except KeyError:
            modded_batch_inputs.append((unused_dictionary[w], unused_dictionary[w]))
      
        label = reverse_dictionary[batch_labels[i][0]]

        if '_' in label:
            label1, label2 = label.split("_")

            try:
                modded_batch_labels.append(unused_dictionary[label1])
            except KeyError:
                modded_batch_labels.append(unused_dictionary[label])
                continue
            try:
                modded_batch_labels.append(unused_dictionary[label2])
            except KeyError:
                del modded_batch_labels[-1]
                modded_batch_labels.append(unused_dictionary[label])
                continue
            modded_batch_inputs.append(modded_batch_inputs[-1])

        else:
            modded_batch_labels.append(unused_dictionary[label])
         


    modded_batch_inputs = np.array(modded_batch_inputs)
    modded_batch_labels = np.array(modded_batch_labels)
    modded_batch_labels = np.reshape(modded_batch_labels, (len(modded_batch_labels), 1))

    return modded_batch_inputs, modded_batch_labels
    
    





# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window, data):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
  if data_index + span > len(data):
    data_index = 0
  buffer.extend(data[data_index:data_index + span])
  data_index += span
  for i in range(batch_size // num_skips):
    context_words = [w for w in range(span) if w != skip_window]
    words_to_use = random.sample(context_words, num_skips)
    for j, context_word in enumerate(words_to_use):
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[context_word]
    if data_index == len(data):
      buffer.extend(data[0:span])
      data_index = span
    else:
      buffer.append(data[data_index])
      data_index += 1
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels


#NOTE:: MUST ALTERNATE LOSS FUNCTION BASED ON WHAT RETRAINING FOR

def word2vec_turk(log_dir, load_dir, filename, retraining=False, X=None, y=None, dictionaries=None, get_embeddings=False, bigram_split=False, load=True, save=True, cosine=False, joint_training=False):
  vocabulary = read_data_nonzip(filename)
  vocabulary_size = 200000


  # Filling 4 global variables:
  # data - list of codes (integers from 0 to vocabulary_size-1).
  #   This is the original text but words are replaced by their codes
  # count - map of words(strings) to count of occurrences
  # dictionary - map of words(strings) to their codes(integers)
  # reverse_dictionary - maps codes(integers) to words(strings)
  data = []
  count = 0
  
  if not dictionaries:
      data, count, unused_dictionary, reverse_dictionary = build_dataset(
          vocabulary, vocabulary_size)
  else:
      data, count, unused_dictionary, reverse_dictionary = dictionaries


  del vocabulary  # Hint to reduce memory.
  print('Most common words (+UNK)', count[:5])
  print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

  batch_size=8500
  embedding_size = 128  # Dimension of the embedding vector.
  skip_window = 1  # How many words to consider left and right.
  num_skips = 2  # How many times to reuse an input to generate a label.
  num_sampled = 64  # Number of negative examples to sample.

  # We pick a random validation set to sample nearest neighbors. Here we limit
  # the validation samples to the words that have a low numeric ID, which by
  # construction are also the most frequent. These 3 variables are used only for
  # displaying model accuracy, they don't affect calculation.
  valid_size = 16  # Random set of words to evaluate similarity on.
  valid_window = 100  # Only pick dev samples in the head of the distribution.
  valid_examples = np.random.choice(valid_window, valid_size, replace=False)
  #print(valid_examples)

  graph = tf.Graph()
  with graph.as_default():

    # Input data.
    with tf.name_scope('inputs'):
      if bigram_split:
        train_inputs = tf.placeholder(tf.int32, shape=[None,2])
      else:
        train_inputs = tf.placeholder(tf.int32, shape=[None])
      train_labels = tf.placeholder(tf.int32, shape=[None, 1])
      valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    with tf.device('/gpu:0'):

      # Look up embeddings for inputs.
      with tf.name_scope('embeddings'):
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        #embed = tf.nn.
        if bigram_split:
            embed = bigram_embedding_lookup(embeddings, train_inputs, reverse_dictionary)
        else: embed = tf.nn.embedding_lookup(embeddings, train_inputs)


      # Construct the variables for the NCE loss
      with tf.name_scope('weights'):
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))

      with tf.name_scope('biases'):
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    # Explanation of the meaning of NCE loss:
    #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/

    with tf.name_scope('loss'):
      loss = tf.reduce_mean(
          tf.nn.nce_loss(
              weights=nce_weights,
              biases=nce_biases,
              labels=train_labels,
              inputs=embed,
              num_sampled=num_sampled,
              num_classes=vocabulary_size))

    # Add the loss value as a scalar to summary.
    tf.summary.scalar('loss', loss)

    # Construct the SGD optimizer using a learning rate of 1.0.
    with tf.name_scope('optimizer'):
      optimizer = tf.train.AdamOptimizer(5e-4).minimize(loss)
      #optimizer = tf.train.GradientDescentOptimizer(1).minimize(loss)
      #cosine_optimizer = tf.train.GradientDescentOptimizer(1).minimize(cosine_loss)

    # Compute the cosine similarity between minibatch examples and all
    
    # Merge all summaries.
    merged = tf.summary.merge_all()

    # Add variable initializer.
    init = tf.global_variables_initializer()

    # Create a saver.
    saver = tf.train.Saver()


  # Step 5: Begin training.
  if retraining: 
      num_steps = 20000
  else:
      num_steps = 100000


  with tf.Session(graph=graph) as session:
    # Open a writer to write summaries.
    writer = tf.summary.FileWriter(log_dir, session.graph)

    # We must initialize all variables before we use them.
    init.run()
    print('Initialized')

    if get_embeddings or load:
        saver.restore(session, os.path.join(load_dir, 'model.ckpt')) 
        print("MODEL RESTORED")
        if get_embeddings:
            return embeddings.eval(), nce_weights.eval()

    print("LEN EMBEDDINGS")
    print(tf.size(embeddings), tf.size(embeddings[0]))
    average_loss = 0
    #batch_inputs = np.zeros([batch_size])
    batch_inputs = np.array([], dtype=int)
    #batch_labels = np.zeros([batch_size, 1])
    batch_labels = np.array([], dtype=int)
    
    if not joint_training:
        
        if retraining:
          inputs, labels = process_inputs(X, y)
          for i in range(len(inputs)):
              batch_inputs = np.append(batch_inputs, unused_dictionary.get(inputs[i]))
              batch_labels = np.append(batch_labels, unused_dictionary.get(labels[i]))

          batch_labels = np.reshape(batch_labels, (len(batch_labels), 1))

          if cosine:
              if bigram_split:
                  itemplaceholder = tf.placeholder(tf.int32, [None,2])
                  nexttoplaceholder = tf.placeholder(tf.int32, [None])
                  x = bigram_embedding_lookup(embeddings, itemplaceholder, reverse_dictionary)
                  y = tf.nn.embedding_lookup(embeddings, nexttoplaceholder)
              else:
                  itemplaceholder = tf.placeholder(tf.int32, [None])
                  nexttoplaceholder = tf.placeholder(tf.int32, [None])
                  x = tf.nn.embedding_lookup(embeddings, itemplaceholder)
                  y = tf.nn.embedding_lookup(embeddings, nexttoplaceholder)
            

              new_loss = tf.losses.cosine_distance(tf.math.l2_normalize(x, axis=1), tf.math.l2_normalize(y, axis=1), axis=1)
              new_optimizer = tf.train.AdamOptimizer(5e-4).minimize(new_loss)

        for step in xrange(num_steps):
          if not retraining:
              batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window, data)

          # Deal with bigrams
          if bigram_split:
              modded_batch_inputs, modded_batch_labels = split_bigrams(batch_inputs, batch_labels, unused_dictionary, reverse_dictionary)
              feed_dict = {train_inputs: modded_batch_inputs, train_labels: modded_batch_labels}
              if cosine:
                  feed_dict = {itemplaceholder: modded_batch_inputs, nexttoplaceholder : np.squeeze(modded_batch_labels)}
                  #feed_dict = {itemplaceholder: modded_batch_inputs, nexttoplaceholder : modded_batch_labels}
              else:
                  feed_dict = {train_inputs: modded_batch_inputs, train_labels: modded_batch_labels}
          else:
              if cosine:
                  feed_dict = {itemplaceholder: batch_inputs, nexttoplaceholder: np.squeeze(batch_labels)}
                  #feed_dict = {itemplaceholder: batch_inputs, nexttoplaceholder: batch_labels}
                
              else:
                  feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

          if cosine:
                  _, loss_val = session.run([new_optimizer, new_loss],
                                                         feed_dict=feed_dict)
          else:
                  run_metadata = tf.RunMetadata()
                  _, summary, loss_val = session.run([optimizer, merged, loss],
                                                         feed_dict=feed_dict,
                                                         run_metadata=run_metadata)
          average_loss += loss_val

          if step % 100 == 0:
              if step > 0:
                  average_loss /= 100
              print('Average loss at step ', step, ': ', average_loss, " time: ", datetime.datetime.now())
              average_loss = 0

    if joint_training:
        # I know hardcoding the hyperparameters like this is not ideal, it's just easier than having a twentieth parameter to this word2vec_turk function
        a = 0.5
        b = 0.5
        #a = 1/5 
        #b = 1/128
        inputs, labels = process_inputs(X, y)
        turk_batch_inputs = np.array([], dtype=int)
        turk_batch_labels = np.array([], dtype=int)
        for i in range(len(inputs)):
            turk_batch_inputs = np.append(turk_batch_inputs, unused_dictionary.get(inputs[i]))
            turk_batch_labels = np.append(turk_batch_labels, unused_dictionary.get(labels[i]))

        turk_batch_labels = np.reshape(turk_batch_labels, (len(turk_batch_labels), 1))

        if bigram_split:
            itemplaceholder = tf.placeholder(tf.int32, [None,2])
            nexttoplaceholder = tf.placeholder(tf.int32, [None])
            x = bigram_embedding_lookup(embeddings, itemplaceholder, reverse_dictionary)
            y = tf.nn.embedding_lookup(embeddings, nexttoplaceholder)

        else:
            itemplaceholder = tf.placeholder(tf.int32, [None])
            nexttoplaceholder = tf.placeholder(tf.int32, [None])
            x = tf.nn.embedding_lookup(embeddings, itemplaceholder)
            y = tf.nn.embedding_lookup(embeddings, nexttoplaceholder)

        cosine_loss = tf.losses.cosine_distance(tf.math.l2_normalize(x, axis=1), tf.math.l2_normalize(y, axis=1), axis=1)
        joint_loss = tf.add(tf.math.multiply(loss, a), tf.math.multiply(cosine_loss, b)) 
        joint_optimizer = tf.train.AdamOptimizer(5e-4).minimize(joint_loss)

        if bigram_split:
            modded_turk_inputs, modded_turk_labels = split_bigrams(turk_batch_inputs, turk_batch_labels, unused_dictionary, reverse_dictionary)




        for step in xrange(num_steps):
            sg_batch_inputs, sg_batch_labels = generate_batch(batch_size, num_skips, skip_window, data)
            if bigram_split:
                modded_sg_inputs, modded_sg_labels = split_bigrams(sg_batch_inputs, sg_batch_labels, unused_dictionary, reverse_dictionary) 
                feed_dict = {train_inputs: modded_sg_inputs, train_labels: modded_sg_labels, itemplaceholder: modded_turk_inputs, nexttoplaceholder: np.squeeze(modded_turk_labels)} 
            else: feed_dict = {train_inputs: sg_batch_inputs, train_labels: sg_batch_labels, itemplaceholder: turk_batch_inputs, nexttoplaceholder: np.squeeze(turk_batch_labels) }
            _, loss_val = session.run([joint_optimizer, joint_loss],
                                                   feed_dict=feed_dict)
            average_loss += loss_val
            if step % 100 == 0:
                if step > 0:
                    average_loss /= 100
                print('Average loss at step ', step, ': ', average_loss, " time: ", datetime.datetime.now())
                average_loss = 0
        



    # Write corresponding labels for the embeddings.
    with open(log_dir + '/metadata.tsv', 'w') as f:
        for i in xrange(vocabulary_size):
            f.write(reverse_dictionary[i] + '\n')

      # Save the model for checkpoints.
            
      #Note: You ONLY WANT to save the model if you are training on the
      #data for the first time. If retraining for multiple test sets, don't save it.
    if save: saver.save(session, os.path.join(log_dir, 'model.ckpt'))
    return embeddings.eval(), nce_weights.eval()

