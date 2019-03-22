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
import math
import os
import random
import sys
from tempfile import gettempdir
import zipfile

import word2vec_eval

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



def word2vec_basic(log_dir, filename, retraining=False, X=None, y=None, dictionaries=None, get_embeddings=False):
  """Example of building, training and visualizing a word2vec model."""

  #if get_embeddings, no training. Just returns final_embeddings

  # Create the directory for TensorBoard variables if there is not.
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)

  # Step 1: Download the data.
  url = 'http://mattmahoney.net/dc/'

  # pylint: disable=redefined-outer-name
  def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    local_filename = os.path.join(gettempdir(), filename)
    if not os.path.exists(local_filename):
      local_filename, _ = urllib.request.urlretrieve(url + filename,
                                                     local_filename)
    statinfo = os.stat(local_filename)
    if statinfo.st_size == expected_bytes:
      print('Found and verified', filename)
    else:
      print(statinfo.st_size)
      raise Exception('Failed to verify ' + local_filename +
                      '. Can you get to it with a browser?')
    return local_filename

  #filename = maybe_download('text8.zip', 31344016)
  #filename = 'modified_text'

  # Read the data into a list of strings.

  #vocabulary = read_data(filename)
  vocabulary = read_data_nonzip(filename)
  
  print('Data size', len(vocabulary))

  # Step 2: Build the dictionary and replace rare words with UNK token.
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

  # Step 3: Function to generate a training batch for the skip-gram model.
  def generate_batch(batch_size, num_skips, skip_window):
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


  """
  for elem in unused_dictionary:
      print(elem)
  """

  #batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)

  # Step 4: Build and train a skip-gram model.

  #batch_size = 128
  #batch_size = len(global_inputs[0])
  #batch_size = 278
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
      train_inputs = tf.placeholder(tf.int32, shape=[None])
      train_labels = tf.placeholder(tf.int32, shape=[None, 1])
      valid_dataset = tf.constant(valid_examples, dtype=tf.int32)


    # Ops and variables pinned to the CPU because of missing GPU implementation
    # Now using GPU
    with tf.device('/gpu:0'):
      # Look up embeddings for inputs.
      with tf.name_scope('embeddings'):
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)


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
              inputs=graph.get_tensor_by_name("embeddings/embedding_lookup:0"),
              num_sampled=num_sampled,
              num_classes=vocabulary_size))
    
    # Add the loss value as a scalar to summary.
    tf.summary.scalar('loss', loss)

    # Construct the SGD optimizer using a learning rate of 1.0.
    with tf.name_scope('optimizer'):
      #optimizer = tf.train.AdamOptimizer(5e-4).minimize(loss)
      optimizer = tf.train.GradientDescentOptimizer(1).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all
    
    #normalized_embeddings = normalize_embeddings(embeddings)
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                              valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)

    # Merge all summaries.
    merged = tf.summary.merge_all()

    # Add variable initializer.
    init = tf.global_variables_initializer()

    # Create a saver.
    saver = tf.train.Saver()

  # Step 5: Begin training.
  if retraining:
    num_steps = 10000
  else: num_steps = 100001

  with tf.Session(graph=graph) as session:
    # Open a writer to write summaries.
    writer = tf.summary.FileWriter(log_dir, session.graph)



    # We must initialize all variables before we use them.
    init.run()
    print('Initialized')

    if retraining or get_embeddings:
        saver.restore(session, os.path.join(log_dir, 'model.ckpt')) 
        print("MODEL RESTORED")
        if get_embeddings:
            #final_embeddings = normalized_embeddings.eval()
            return embeddings.eval(), nce_weights.eval()


    print("LEN EMBEDDINGS")
    print(tf.size(embeddings), tf.size(embeddings[0]))
    average_loss = 0
    print(valid_embeddings.eval())
    for step in xrange(num_steps):
      #batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
      #print(batch_inputs)
      #print(batch_labels)
      #inputs, labels = global_inputs[train_batch], global_labels[train_batch]

      #batch_labels = np.transpose(batch_labels)
      #print("TRANSPOSED")
      #print(len(batch), len(labels))
      batch_inputs = np.zeros([batch_size])
      batch_labels = np.zeros([batch_size, 1])
    
      if retraining:
          inputs, labels = process_inputs(X, y)
          for i in range(len(inputs)):
              #print(unused_dictionary.get(inputs[i]))
              batch_inputs[i] = unused_dictionary.get(inputs[i])
              batch_labels[i,0] = unused_dictionary.get(labels[i])
          #print(batch_labels)
      
      else: batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
    
      #for x in range(batch_size - len(batch_inputs)):
          #batch_inputs = np.append(batch_inputs, [999999])
          #batch_labels = np.append(batch_labels, [999999])

      #print(np.shape(batch_inputs))
      #np.reshape(batch_inputs, 278, 1)
      #np.reshape(batch_labels, 1, 278)


      feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

      # Define metadata variable.
      run_metadata = tf.RunMetadata()

      # We perform one update step by evaluating the optimizer op (including it
      # in the list of returned values for session.run()
      # Also, evaluate the merged op to get all summaries from the returned
      # "summary" variable. Feed metadata variable to session for visualizing
      # the graph in TensorBoard.
      _, summary, loss_val = session.run([optimizer, merged, loss],
                                         feed_dict=feed_dict,
                                         run_metadata=run_metadata)
      average_loss += loss_val

      # Add returned summaries to writer in each step.
      writer.add_summary(summary, step)
      # Add metadata to visualize the graph for the last run.
      if step == (num_steps - 1):
        writer.add_run_metadata(run_metadata, 'step%d' % step)

      if step % 2000 == 0:
        if step > 0:
          average_loss /= 2000
        # The average loss is an estimate of the loss over the last 2000
        # batches.
        print('Average loss at step ', step, ': ', average_loss)
        average_loss = 0
      
      # Note that this is expensive (~20% slowdown if computed every 500 steps)
      if step % 10000 == 0:
        sim = similarity.eval()
        for i in xrange(valid_size):
          valid_word = reverse_dictionary[valid_examples[i]]
          top_k = 8  # number of nearest neighbors
          nearest = (-sim[i, :]).argsort()[1:top_k + 1]
          log_str = 'Nearest to %s:' % valid_word
          for k in xrange(top_k):
            close_word = reverse_dictionary[nearest[k]]
            log_str = '%s %s,' % (log_str, close_word)
          print(log_str)
    final_embeddings = normalized_embeddings.eval()
    print(final_embeddings)

    # Write corresponding labels for the embeddings.
    with open(log_dir + '/metadata.tsv', 'w') as f:
      for i in xrange(vocabulary_size):
        f.write(reverse_dictionary[i] + '\n')

    # Save the model for checkpoints.
    
    #Note: You ONLY WANT to save the model if you are training on the
    #data for the first time. If retraining for multiple test sets, don't save it.
    if not retraining: saver.save(session, os.path.join(log_dir, 'model.ckpt'))
    #return final_embeddings, 
    return embeddings.eval(), nce_weights.eval()



# All functionality is run after tf.app.run() (b/122547914). This could be split
# up but the methods are laid sequentially with their usage for clarity.
"""
def main(unused_argv):
  # Give a folder path as an argument with '--log_dir' to save
  # TensorBoard summaries. Default is a log folder in current directory.
  current_path = os.path.dirname(os.path.realpath(sys.argv[0]))

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join(current_path, 'log'),
      help='The log directory for TensorBoard summaries.')
  flags, unused_flags = parser.parse_known_args()
  word2vec_basic(flags.log_dir)

if __name__ == '__main__':
  tf.app.run()
"""
