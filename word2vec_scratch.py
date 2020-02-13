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

#from tensorflow.contrib.tensorboard.plugins import projector
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
    sess = tf.compat.v1.Session()
    with sess.as_default():
        norm = tf.sqrt(tf.reduce_sum(input_tensor=tf.square(embeddings), axis=1, keepdims=True))
        normalized_embeddings = embeddings / norm

        return normalized_embeddings.eval()

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
        else:
            label1, label2 = label, label
        try:
            modded_batch_labels.append((unused_dictionary[label1], unused_dictionary[label2]))
        except KeyError:
            modded_batch_labels.append((unused_dictionary[label], unused_dictionary[label]))

    modded_batch_inputs = np.array(modded_batch_inputs)
    modded_batch_labels = np.array(modded_batch_labels)
    #modded_batch_labels = np.reshape(modded_batch_labels, (len(modded_batch_labels), 1))

    return modded_batch_inputs, modded_batch_labels
    

def gen_io_vector(input_pairs,label_pairs,vocabulary_size, batch_size):
    #one-hot used as input (two-hot if input is a bigram)
    #vec = np.zeros([1, vocabulary_size])

    #vec = tf.SparseTensor(indices=[bigram_pair, 1], values=[1], dense_shape=[vocabulary_size, 1])
    value_vec = [1 for x in range(len(input_pairs))]
    input_mat = tf.compat.v1.SparseTensorValue(indices=input_pairs, values=value_vec, dense_shape=[batch_size, vocabulary_size])

    label_mat = tf.compat.v1.SparseTensorValue(indices=label_pairs, values=value_vec, dense_shape=[vocabulary_size, batch_size])


    """
    if bigram_pair[1]:
        # Norm of the vector must be 1, so we use 1/sqrt(2) for two elements
        fill_val = 1/sqrt(2)
        vec[0][bigram_pair[0]] = fill_val
        vec[0][bigram_pair[1]] = fill_val
            
    else:
    """
    #vec[0][bigram_pair] = 1
    #return vec[0]
    return input_mat, label_mat
    





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
@tf.function
def w2v_loss(output_matrix, labels, skip_window=10, batch_size=60):
    #context_words = 2*skip_window+1
    
    output_unpacked = tf.unstack(output_matrix)
    #labels_unpacked = tf.unstack(labels, axis=1)
    labels_unpacked = labels.indices

    loss_scalars = []
    loss_scalars.append(tf.gather_nd(output_matrix, indices=labels_unpacked))

        
    loss_scalars = tf.add(1e-9, loss_scalars)
    loss = tf.reduce_sum(-tf.math.log(loss_scalars))
    return loss


        

    #loss = tf.reduce_sum(tf.math.abs(tf.math.subtract(labels, inputs)))
    return loss 


def get_embeddings(vec):
    embed = tf.matmul(vec, embeddings)
    return embed



def word2vec_turk(log_dir, load_dir, filename, retraining=False, X=None, y=None, dictionaries=None, get_embeddings=False, bigram_split=False, load=True, save=True, cosine=False, joint_training=False, load_early=True, a=0.5, b=0.5):

  if joint_training:
  	load_early = False

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

  batch_size=120
  embedding_size = 128  # Dimension of the embedding vector.
  skip_window = 10# How many words to consider left and right.
  num_skips = 20# How many times to reuse an input to generate a label.
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
    with tf.compat.v1.name_scope('inputs'):
      #train_inputs = tf.compat.v1.placeholder(tf.float32, shape=[batch_size, vocabulary_size])
      #train_labels = tf.compat.v1.placeholder(tf.float32, shape=[vocabulary_size, batch_size])
      train_inputs = tf.compat.v1.sparse_placeholder(tf.float32)
      train_labels = tf.compat.v1.sparse_placeholder(tf.float32)
    
      cosine_input = tf.compat.v1.placeholder(tf.float32, shape=[1, vocabulary_size])
      cosine_label = tf.compat.v1.placeholder(tf.float32, shape=[vocabulary_size, 1])
    


      valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    #with tf.device('/job:localhost/replica:0/task:0/device:XLA_CPU:0'):
    with tf.device('/job:localhost/replica:0/task:0/device:CPU:0'):

      # Look up embeddings for inputs.
      with tf.compat.v1.name_scope('embeddings'):
        embeddings = tf.Variable(
            tf.random.uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        #if bigram_split:
        #    embed = bigram_embedding_lookup(embeddings, train_inputs, reverse_dictionary)
        #embed = tf.nn.embedding_lookup(params=embeddings, ids=train_inputs)

        # represents first matrix multiplication: Result of this is the selected word embeddings
        # with all else as zero.
        #embed = tf.matmul(train_inputs, embeddings)
        embed = tf.sparse.sparse_dense_matmul(train_inputs, embeddings)




      # Construct the variables for the NCE loss
      with tf.compat.v1.name_scope('weights'):
        nce_weights = tf.Variable(
            tf.random.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))

      with tf.compat.v1.name_scope('biases'):
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
      
    output_layer = tf.reshape(tf.nn.softmax(tf.matmul(embed, tf.transpose(nce_weights))), [batch_size, vocabulary_size])

     # Compute the average NCE loss for the batch.
    with tf.compat.v1.name_scope('loss'):
      #output_layer = tf.matmul(embed, tf.transpose(nce_weights))
      #loss = tf.math.reduce_sum(w2v_loss(output_layer, train_labels)) 
      loss = w2v_loss(output_layer, train_labels)
      #loss = tf.math.reduce_mean(tf.squeeze(tf.nn.softmax_cross_entropy_with_logits(output_layer, train_labels[0])))
      """
      loss = tf.reduce_mean(
          input_tensor=tf.nn.nce_loss(
              weights=nce_weights,
              biases=nce_biases,
              labels=train_labels,
              inputs=embed,
              num_sampled=num_sampled,
              num_classes=vocabulary_size))
      """
    tf.compat.v1.summary.scalar('loss', loss)

    # Construct the SGD optimizer using a learning rate of 1.0.
    with tf.compat.v1.name_scope('optimizer'):
      #optimizer = tf.train.AdamOptimizer(learning_rate=5e-4,beta1=0.9,beta2=0.999).minimize(loss)
      optimizer = tf.compat.v1.train.GradientDescentOptimizer(1).minimize(loss)
      #cosine_optimizer = tf.train.GradientDescentOptimizer(1).minimize(cosine_loss)

    # Compute the cosine similarity between minibatch examples and all
    
    # Add variable initializer.
    init = tf.compat.v1.global_variables_initializer()

    # Create a saver.
    saver = tf.compat.v1.train.Saver()


  # Step 5: Begin training.
  num_steps = 30000
  #if retraining: 
  #else:
  #    num_steps = 20000


  with tf.compat.v1.Session(graph=graph) as session:
    # Open a writer to write summaries.

    # We must initialize all variables before we use them.
    init.run()
    print('Initialized')

    if (get_embeddings or load) and load_early:
        saver.restore(session, os.path.join(load_dir, 'model.ckpt')) 
        print("MODEL RESTORED")
        if get_embeddings:
            return embeddings.eval(), nce_weights.eval()

    print("LEN EMBEDDINGS")
    print(tf.size(input=embeddings), tf.size(input=embeddings[0]))
    average_loss = 0
    #batch_inputs = np.zeros([batch_size])
    batch_inputs = np.array([], dtype=int)
    #batch_labels = np.zeros([batch_size, 1])
    batch_labels = np.array([], dtype=int)

    
    # training regular wikipedia model
    for step in xrange(num_steps):
      if not retraining:
          batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window, data)
          #modded_batch_inputs, modded_batch_labels = split_bigrams(batch_inputs, batch_labels, unused_dictionary, reverse_dictionary)

          input_indices = []
          label_indices = []

          for ind in range(batch_size):
              input_indices.append([ind, batch_inputs[ind]])
              label_indices.append([ind, batch_labels[ind]])

          #generate input sparse matrices
          input_mat, label_mat = gen_io_vector(input_indices, label_indices, vocabulary_size, batch_size)

          feed_dict = {train_inputs: input_mat, train_labels: label_mat}
          merged = tf.compat.v1.summary.merge_all()
          run_metadata = tf.compat.v1.RunMetadata()
          _, _, loss_val = session.run([optimizer, merged, loss],
                                                 feed_dict=feed_dict,
                                                     run_metadata=run_metadata)
          average_loss += loss_val

          if step % 100 == 0:
            if step > 0:
                average_loss /= 100
                print('Average loss at step ', step, ': ', average_loss, " time: ", datetime.datetime.now())
                average_loss = 0
            if save: saver.save(session, os.path.join(log_dir, 'model.ckpt'))
            step += 1





      if retraining:

          #COSINE LOOP
          #==========================
          input_vectors = []
          output_vectors = []

          inputs, labels = process_inputs(X, y) 
          for ind in range(len(inputs)):
              #input_word = modded_batch_inputs[ind]
              #output_word = modded_batch_labels[ind]

              input_word = inputs[ind]
              output_word = labels[ind]

              input_vectors.append(gen_io_vector(input_word, vocabulary_size)) 

              output_vectors.append(gen_io_vector(output_word, vocabulary_size, output=True))
          


          for instance in range(len(input_vectors)):

              input_vector = get_embeddings(input_vectors[instance])

              label_vector = get_embeddings(output_vectors[instance])

              other_object_vectors = [get_embeddings(shelf_object_vectors[instance][x] for x in shelf_object_vectors[instance])]


              feed_dict = {train_inputs: input_vector, train_labels: label_vector}
              merged = tf.compat.v1.summary.merge_all()
              run_metadata = tf.compat.v1.RunMetadata()
              cosine_loss = tf.math.scalar_mul(1000, tf.compat.v1.losses.cosine_distance(tf.math.l2_normalize(input_vector, axis=1), tf.math.l2_normalize(label_vector, axis=1), axis=1))
              _, _, loss_val = session.run([optimizer, merged, cosine_loss],
                                                     feed_dict=feed_dict,
                                                         run_metadata=run_metadata)
              average_loss += loss_val
              if step % 100 == 0:
                if step > 0:
                    average_loss /= 100
                    print('Average loss at step ', step, ': ', average_loss, " time: ", datetime.datetime.now())
                    average_loss = 0
                step += 1
                      

          #WIKIPEDIA LOOP
          #==========================
          batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window, data)
          #modded_batch_inputs, modded_batch_labels = split_bigrams(batch_inputs, batch_labels, unused_dictionary, reverse_dictionary)

          input_indices = []
          label_indices = []

          for ind in range(batch_size):
              input_indices.append([ind, batch_inputs[ind]])
              label_indices.append([ind, batch_labels[ind]])

          #generate input sparse matrices
          input_mat, label_mat = gen_io_vector(input_indices, label_indices, vocabulary_size, batch_size)

          feed_dict = {train_inputs: input_mat, train_labels: label_mat}
          merged = tf.compat.v1.summary.merge_all()
          run_metadata = tf.compat.v1.RunMetadata()
          _, _, loss_val = session.run([optimizer, merged, loss],
                                                 feed_dict=feed_dict,
                                                     run_metadata=run_metadata)
          average_loss += loss_val

          if step % 100 == 0:
            if step > 0:
                average_loss /= 100
                print('Average loss at step ', step, ': ', average_loss, " time: ", datetime.datetime.now())
                average_loss = 0
            if save: saver.save(session, os.path.join(log_dir, 'model.ckpt'))
            step += 1








    
    """
    if not joint_training:
        
        if retraining:
          inputs, labels = process_inputs(X, y)
          for i in range(len(inputs)):
              batch_inputs = np.append(batch_inputs, unused_dictionary.get(inputs[i]))
              batch_labels = np.append(batch_labels, unused_dictionary.get(labels[i]))

          batch_labels = np.reshape(batch_labels, (len(batch_labels), 1))

          if cosine:
            itemplaceholder = tf.compat.v1.placeholder(tf.int32, [1,2])
            nexttoplaceholder = tf.compat.v1.placeholder(tf.int32, [1,2])
            x = bigram_embedding_lookup(embeddings, itemplaceholder, reverse_dictionary)
            y = tf.nn.embedding_lookup(params=embeddings, ids=nexttoplaceholder)
            

            new_loss = tf.compat.v1.losses.cosine_distance(tf.math.l2_normalize(x, axis=1), tf.math.l2_normalize(y, axis=1), axis=1)
            new_optimizer = tf.compat.v1.train.GradientDescentOptimizer(1).minimize(new_loss)
            session.run(tf.compat.v1.global_variables_initializer())

        if (get_embeddings or load) and not load_early:
            saver.restore(session, os.path.join(load_dir, 'model.ckpt')) 
            print("MODEL RESTORED")
            if get_embeddings:
                return embeddings.eval(), nce_weights.eval()

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
                  run_metadata = tf.compat.v1.RunMetadata()
                  _, _, loss_val = session.run([optimizer, merged, loss],
                                                         feed_dict=feed_dict,
                                                         run_metadata=run_metadata)
          average_loss += loss_val

          if step % 100 == 0:
              if step > 0:
                  average_loss /= 100
              print('Average loss at step ', step, ': ', average_loss, " time: ", datetime.datetime.now())
              average_loss = 0

    if joint_training:
        a = a
        b = b
        inputs, labels = process_inputs(X, y)
        turk_batch_inputs = np.array([], dtype=int)
        turk_batch_labels = np.array([], dtype=int)
        for i in range(len(inputs)):
            turk_batch_inputs = np.append(turk_batch_inputs, unused_dictionary.get(inputs[i]))
            turk_batch_labels = np.append(turk_batch_labels, unused_dictionary.get(labels[i]))

        turk_batch_labels = np.reshape(turk_batch_labels, (len(turk_batch_labels), 1))

        if bigram_split:
            itemplaceholder = tf.compat.v1.placeholder(tf.int32, [None,2])
            nexttoplaceholder = tf.compat.v1.placeholder(tf.int32, [None])
            x = bigram_embedding_lookup(embeddings, itemplaceholder, reverse_dictionary)
            y = tf.nn.embedding_lookup(params=embeddings, ids=nexttoplaceholder)

        else:
            itemplaceholder = tf.compat.v1.placeholder(tf.int32, [None])
            nexttoplaceholder = tf.compat.v1.placeholder(tf.int32, [None])
            x = tf.nn.embedding_lookup(params=embeddings, ids=itemplaceholder)
            y = tf.nn.embedding_lookup(params=embeddings, ids=nexttoplaceholder)

        joint_loss = tf.add(tf.math.scalar_mul(a, loss), tf.math.scalar_mul(b, cosine_loss)) 
        joint_optimizer = tf.compat.v1.train.GradientDescentOptimizer(5e-4).minimize(joint_loss)
        session.run(tf.compat.v1.global_variables_initializer())

        if bigram_split:
            modded_turk_inputs, modded_turk_labels = split_bigrams(turk_batch_inputs, turk_batch_labels, unused_dictionary, reverse_dictionary)

        if (get_embeddings or load) and not load_early:
            saver.restore(session, os.path.join(load_dir, 'model.ckpt')) 
            print("MODEL RESTORED")
            if get_embeddings:
                return embeddings.eval(), nce_weights.eval()

        for step in xrange(num_steps):
                sg_batch_inputs, sg_batch_labels = generate_batch(batch_size, num_skips, skip_window, data)
                if bigram_split:
                    modded_sg_inputs, modded_sg_labels = split_bigrams(sg_batch_inputs, sg_batch_labels, unused_dictionary, reverse_dictionary) 
                    feed_dict = {train_inputs: modded_sg_inputs, train_labels: modded_sg_labels, itemplaceholder: modded_turk_inputs, nexttoplaceholder: np.squeeze(modded_turk_labels)} 
                else: feed_dict = {train_inputs: sg_batch_inputs, train_labels: sg_batch_labels, itemplaceholder: turk_batch_inputs, nexttoplaceholder: np.squeeze(turk_batch_labels) }


                _, loss_val = session.run([joint_optimizer, joint_loss],
                                                   feed_dict=feed_dict)

                average_loss += loss_val
                if step % 10 == 0:
                    if step > 0:
                        average_loss /= 10
                    print('Average loss at step ', step, ': ', average_loss, " time: ", datetime.datetime.now())
                    average_loss = 0
        

      """


      # Save the model for checkpoints.
            
      #Note: You ONLY WANT to save the model if you are training on the
      #data for the first time. If retraining for multiple test sets, don't save it.
    if save: saver.save(session, os.path.join(log_dir, 'model.ckpt'))
    return embeddings.eval(), nce_weights.eval()

