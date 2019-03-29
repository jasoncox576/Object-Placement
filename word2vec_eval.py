import csv
import spacy
from nltk.corpus import wordnet as wn
from sklearn.model_selection import StratifiedKFold
from object_placement_turk import *
from word2vec_basic import *
from word2vec_train import *
import numpy as np
import matrix_priors
import random

def evaluate_empirical(matrix, X, y, test, rows_dict):
    total_correct = 0

    for instance in test:

        X_inst = X[instance]
        
        sim_vector = []

        primary = rows_dict[X_inst[0]]
        w_1 = rows_dict[X_inst[1]]
        w_2 = rows_dict[X_inst[2]]
        w_3 = rows_dict[X_inst[3]]

        sim_vector.append(matrix[primary][w_1])
        sim_vector.append(matrix[primary][w_2])
        sim_vector.append(matrix[primary][w_3])

        selected_index = roll_probs(sim_vector)

        if X_inst[selected_index+1] == y[instance]:
            total_correct += 1

    return total_correct / len(test)


def eval_random(X, y, test):

    total_correct = 0

    for instance in test:
        
        X_inst = X[instance]
        if random.choice(X_inst[1:]) == y[instance]:
            total_correct += 1
    
    return total_correct / len(test)

def get_word(in_word, dictionary, synset_dic, embeddings):
  index = dictionary.get(synset_dic.get_synset_and_strip(in_word)[1])
  embed = embeddings[index]
  n_embed = embed/ np.linalg.norm(embed)
  return index, embed, n_embed

def evaluate_word2vec(X, y, embeddings, weights, dictionary, outfile_name, rows_dict=None):
    out_file = open(outfile_name, 'w') 

    total_correct_w2v = 0
    total_correct_output = 0

    total_unigram = 0
    total_bigram = 0
    
    w2v_unigram_correct = 0
    output_unigram_correct = 0
    w2v_bigram_correct = 0
    output_bigram_correct = 0

    for case in range(len(X)):
        p = X[case][0]          #primary
        a = X[case][1]          #choice a
        b = X[case][2]          #choice b
        c = X[case][3]          #choice c
        z = y[case]             #answer

        p_index, p_embedding, p_nembedding = get_word(p, dictionary, matrix_priors, embeddings)
        a_index, a_embedding, a_nembedding = get_word(a, dictionary, matrix_priors, embeddings)
        b_index, b_embedding, b_nembedding = get_word(b, dictionary, matrix_priors, embeddings)
        c_index, c_embedding, c_nembedding = get_word(c, dictionary, matrix_priors, embeddings)
        z_index, z_embedding, z_nembedding = get_word(z, dictionary, matrix_priors, embeddings)
         
        embedding_sim_vector = []
        embedding_sim_vector.append(np.dot(p_nembedding, a_nembedding))
        embedding_sim_vector.append(np.dot(p_nembedding, b_nembedding))
        embedding_sim_vector.append(np.dot(p_nembedding, c_nembedding))

        indices = [a_index, b_index, c_index]
        answer_index = X[case][1:].index(y[case])
        machine_answer_index = np.argmax(embedding_sim_vector)

        out_file.write("%d, %d, %d, %d, %d, %d, %d, %d, %s, %s, %s, %s, %s \n" % (p_index, a_index, b_index, c_index, z_index, indices[machine_answer_index], answer_index, machine_answer_index, p, a, b, c, z))

        if(np.argmax(embedding_sim_vector) == X[case][1:].index(y[case])):
            total_correct_w2v += 1
            if '_' in p:
                w2v_bigram_correct += 1
            else:
                w2v_unigram_correct += 1

        output_vector = np.matmul([embeddings[p_index]], np.transpose(weights))
        output_vector = np.reshape(output_vector, (len(output_vector[0])))
        

        output_sim_vector = []

        output_sim_vector.append(output_vector[a_index])
        output_sim_vector.append(output_vector[b_index])
        output_sim_vector.append(output_vector[c_index])

        if(np.argmax(output_sim_vector) == X[case][1:].index(y[case])):
            total_correct_output += 1
            if '_' in p:
                output_bigram_correct += 1
            else:
                output_unigram_correct += 1
        else:
            if X[case][0] in X[case][1:]:
                total_correct_output += 1
                if '_' in p:
                    output_bigram_correct += 1
                else:
                    output_unigram_correct += 1 

        if '_' in p:
            total_bigram += 1
        else:
            total_unigram += 1
        
    out_file.close();

    return total_correct_w2v/len(X), total_correct_output/len(X)
    #, [w2v_unigram_correct/total_unigram, w2v_bigram_correct/total_bigram, output_unigram_correct/total_unigram, output_bigram_correct/total_bigram]


def evaluate_wordnet(X, y, test, dictionary, rows_dict=None):

    # test is a one-dimensional array.
    # Run this as many times as you want, average the accuracy.

    total_correct_w2v = 0
    total_correct_wordnet = 0

    
    for case in test:

        primary = X[case][0] 

        primary_syn = matrix_priors.get_synset_and_strip(primary)[0]
        
        a = X[case][1]
        a_syn = matrix_priors.get_synset_and_strip(a)[0]

        b = X[case][2]
        b_syn = matrix_priors.get_synset_and_strip(b)[0]

        c = X[case][3]
        c_syn = matrix_priors.get_synset_and_strip(c)[0]
         
        wordnet_sim_vector = []
        wordnet_sim_vector.append(primary_syn.path_similarity(a_syn))
        wordnet_sim_vector.append(primary_syn.path_similarity(b_syn))
        wordnet_sim_vector.append(primary_syn.path_similarity(c_syn))
    
        if(np.argmax(wordnet_sim_vector) == X[case][1:].index(y[case])):
            total_correct_wordnet += 1

    return (total_correct_wordnet/len(test))

