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
from pathlib import Path

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
    filename_dir = os.path.join(os.getcwd(), "..", filename)
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

def read_data_nonzip(filename):
    filename_dir = os.path.join(os.getcwd(), "..", filename)
    with open(filename_dir, 'r') as f:
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

def bigram_embedding_lookup(embeddings, train_inputs, reverse_dictionary):
    embed1 = tf.nn.embedding_lookup(params=embeddings, ids=train_inputs[:, 0])
    embed2 = tf.nn.embedding_lookup(params=embeddings, ids=train_inputs[:, 1])
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

def word2vec_turk(log_dir, load_dir, filename, retraining=False, X=None, y=None, dictionaries=None, get_embeddings=False, bigram_split=False, load=True, save=True, cosine=False, joint_training=False, load_early=True, a=0.5, b=0.5, data_index_dir="data_index"):

    log_dir = os.path.join(os.path.abspath(os.getcwd()), log_dir)
    load_dir = os.path.join(os.path.abspath(os.getcwd()), load_dir)

    global data_index

    # load the current data index from file if it exists
    index_file_path = Path("../"+data_index_dir)
    if index_file_path.is_file():
        di_f = open(os.path.join(os.getcwd(), "..", data_index_dir), 'r' )
        data_index = int(di_f.readline())
        di_f.close()
    elif retraining:
        di_f = open(os.path.join(os.getcwd(), "..", "data_index"), 'r')
        data_index = int(di_f.readline())
        di_f.close()
    
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
    #print('Most common words (+UNK)', count[:5])
    #print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

    batch_size=8500
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
            if bigram_split:
                train_inputs = tf.compat.v1.placeholder(tf.int32, shape=[None,2])
            else:
                train_inputs = tf.compat.v1.placeholder(tf.int32, shape=[None])
            train_labels = tf.compat.v1.placeholder(tf.int32, shape=[None, 1])
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        #with tf.device('/job:localhost/replica:0/task:0/device:XLA_CPU:0'):
        #with tf.device('/job:localhost/replica:0/task:0/device:XLA_GPU:0'):
        with tf.device('/job:localhost/replica:0/task:0/device:cpu:0'):

            # Look up embeddings for inputs.
            with tf.compat.v1.name_scope('embeddings'):
                embeddings = tf.Variable(
                    tf.random.uniform([vocabulary_size, embedding_size], -1.0, 1.0))
                #embed = tf.nn.
                if bigram_split:
                    embed = bigram_embedding_lookup(embeddings, train_inputs, reverse_dictionary)
                else: embed = tf.nn.embedding_lookup(params=embeddings, ids=train_inputs)


            # Construct the variables for the NCE loss
            with tf.compat.v1.name_scope('weights'):
                nce_weights = tf.Variable(
                    tf.random.truncated_normal([vocabulary_size, embedding_size],
                                        stddev=1.0 / math.sqrt(embedding_size)))

            with tf.compat.v1.name_scope('biases'):
                nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        # Explanation of the meaning of NCE loss:
        #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/

        with tf.compat.v1.name_scope('loss'):
            loss = tf.reduce_mean(
                input_tensor=tf.nn.nce_loss(
                    weights=nce_weights,
                    biases=nce_biases,
                    labels=train_labels,
                    inputs=embed,
                    num_sampled=num_sampled,
                    num_classes=vocabulary_size))

        # Add the loss value as a scalar to summary.
        tf.compat.v1.summary.scalar('loss', loss)

        # Construct the SGD optimizer using a learning rate of 1.0.
        with tf.compat.v1.name_scope('optimizer'):
            #optimizer = tf.train.AdamOptimizer(learning_rate=5e-4,beta1=0.9,beta2=0.999).minimize(loss)
            optimizer = tf.compat.v1.train.GradientDescentOptimizer(1).minimize(loss)
            #cosine_optimizer = tf.train.GradientDescentOptimizer(1).minimize(cosine_loss)

        # Compute the cosine similarity between minibatch examples and all

        # Merge all summaries.
        merged = tf.compat.v1.summary.merge_all()

        # Add variable initializer.
        init = tf.compat.v1.global_variables_initializer()

        # Create a saver.
        saver = tf.compat.v1.train.Saver()


    # Step 5: Begin training.
    num_wiki_steps = 100000
    num_cosine_steps = 5
    num_wiki_retrain = 500


    with tf.compat.v1.Session(graph=graph) as session:
        # Open a writer to write summaries.
        writer = tf.compat.v1.summary.FileWriter(log_dir, session.graph)

        # We must initialize all variables before we use them.
        init.run()
        print('Initialized')

        if (get_embeddings or load) and load_early:
            print("Loading from ", load_dir)
            saver.restore(session, os.path.join(load_dir, 'model.ckpt'))
            print("MODEL RESTORED")
            if get_embeddings:
                return embeddings.eval(), nce_weights.eval()

        #print("LEN EMBEDDINGS")
        #print(tf.size(input=embeddings), tf.size(input=embeddings[0]))
        average_loss = 0
        #batch_inputs = np.zeros([batch_size])
        batch_inputs = np.array([], dtype=int)
        #batch_labels = np.zeros([batch_size, 1])
        batch_labels = np.array([], dtype=int)


        ### TRAIN ORIGINAL WIKI
        if not retraining:
            for step in xrange(num_wiki_steps):
                batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window, data)
                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
                run_metadata = tf.compat.v1.RunMetadata()
                _, summary, loss_val = session.run([optimizer, merged, loss],
                                                       feed_dict=feed_dict,
                                                       run_metadata=run_metadata)
                average_loss += loss_val

                if step % 100 == 0:
                    if step > 0:
                        average_loss /= 100
                    print('Average loss at step ', step, ': ', average_loss, " time: ", datetime.datetime.now())
                    average_loss = 0
                    if save: 
                        saver.save(session, os.path.join(log_dir, 'model.ckpt'))
                        di_f = open(os.path.join(os.getcwd(), "..", data_index_dir), 'w')
                        di_f.write(str(data_index)) 
                        di_f.close()



        if retraining:
            ### COSINE LOOP
            inputs, labels = process_inputs(X, y)
            for i in range(len(inputs)):
                batch_inputs = np.append(batch_inputs, unused_dictionary.get(inputs[i]))
                batch_labels = np.append(batch_labels, unused_dictionary.get(labels[i]))

            batch_labels = np.reshape(batch_labels, (len(batch_labels), 1))

            itemplaceholder = tf.compat.v1.placeholder(tf.int32, [None])
            nexttoplaceholder = tf.compat.v1.placeholder(tf.int32, [None])
            x = tf.nn.embedding_lookup(params=embeddings, ids=itemplaceholder)
            y = tf.nn.embedding_lookup(params=embeddings, ids=nexttoplaceholder)


            new_loss = tf.compat.v1.losses.cosine_distance(tf.math.l2_normalize(x, axis=1), tf.math.l2_normalize(y, axis=1), axis=1)
            new_optimizer = tf.compat.v1.train.GradientDescentOptimizer(1).minimize(new_loss)
            session.run(tf.compat.v1.global_variables_initializer())


            if (get_embeddings or load) and not load_early:
                saver.restore(session, os.path.join(load_dir, 'model.ckpt'))
                #print("MODEL RESTORED")
                print("Loading from", load_dir)
                if get_embeddings:
                    return embeddings.eval(), nce_weights.eval()


            for step in xrange(num_cosine_steps):
                feed_dict = {itemplaceholder: batch_inputs, nexttoplaceholder: np.squeeze(batch_labels)}

                _, loss_val = session.run([new_optimizer, new_loss],
                                                           feed_dict=feed_dict)
                average_loss += loss_val

                if step % 100 == 0:
                    if step > 0:
                        average_loss /= 100
                    print('Average loss at step ', step, ': ', average_loss, " time: ", datetime.datetime.now())
                    average_loss = 0
                    if save: saver.save(session, os.path.join(log_dir, 'model.ckpt'))

            """

            ###===================================================================================
            ### WIKI LOOP
            for step in xrange(num_wiki_retrain):
                batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window, data)
                run_metadata = tf.compat.v1.RunMetadata()
                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
                _, summary, loss_val = session.run([optimizer, merged, loss],
                                                       feed_dict=feed_dict,
                                                       run_metadata=run_metadata)
                average_loss += loss_val

                if step % 100 == 0:
                    if step > 0:
                        average_loss /= 100
                    print('Average loss at step ', step, ': ', average_loss, " time: ", datetime.datetime.now())
                    average_loss = 0
                    if save: 
                        saver.save(session, os.path.join(log_dir, 'model.ckpt'))
                        di_f = open(data_index_dir, 'w')
                        di_f.write(str(data_index)) 
                        di_f.close()

            """





        # Write corresponding labels for the embeddings.
        with open(log_dir + '/metadata.tsv', 'w') as f:
            for i in xrange(vocabulary_size):
                f.write(reverse_dictionary[i] + '\n')

            # Save the model for checkpoints.

            #Note: You ONLY WANT to save the model if you are training on the
            #data for the first time. If retraining for multiple test sets, don't save it.
        if save: saver.save(session, os.path.join(log_dir, 'model.ckpt'))
        return embeddings.eval(), nce_weights.eval()
