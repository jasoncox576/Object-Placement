import csv
import spacy
from nltk.corpus import wordnet as wn
import numpy as np
import matrix_priors
from word2vec_eval import *
import random





def evaluate_wordnet(X, y, dictionary, rows_dict=None):

    # Run this as many times as you want, average the accuracy.

    total_correct_w2v = 0
    total_correct_wordnet = 0


    for case in range(len(X)):
        primary = X[case][0]

        primary_syn = matrix_priors.get_synset_and_strip(primary)[0]

        other_objects = X[case][1:]
        wordnet_sim_vector = []
        for obj in other_objects:
            new_syn = matrix_priors.get_synset_and_strip(obj)[0]
            wordnet_sim_vector.append(primary_syn.path_similarity(new_syn)

        wordnet_sim_vector.append(primary_syn.path_similarity(a_syn))
        wordnet_sim_vector.append(primary_syn.path_similarity(b_syn))
        wordnet_sim_vector.append(primary_syn.path_similarity(c_syn))

        if(np.argmax(wordnet_sim_vector) == X[case][1:].index(y[case])):
            total_correct_wordnet += 1
        """
        print(X[case])
        print("ACTUAL ANSWER: ", y[case])

        print("WORDNET MAX: ", str(X[case][np.argmax(wordnet_sim_vector)+1]))
        """

    return (total_correct_wordnet/len(X))


if __name__=="__main__":

    bigram_filename = '/home/justin/Data/modified_text'
    turk_data_filename = 'official_results.csv'

    bigram_dictionaries = get_pretrain_dictionaries(bigram_filename)
    bigram_unused_dictionary = bigram_dictionaries[2]

    X, y, = get_train_test(turk_data_filename)
    wordnet_acc = 0

    wordnet_acc = evaluate_wordnet(X, y, bigram_unused_dictionary) 
    




