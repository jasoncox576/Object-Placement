import csv
#import spacy
#from nltk.corpus import wordnet as wn
from data_gen import *
from word2vec_basic import *
from word2vec_train import *
import numpy as np
import matrix_priors
import random
import os

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


def get_max_performance(X, y):
    # Returns the Theoretical Max performance on a given dataset.
    used_indices = {}

    total_agreement = 0

    agreement_counter = {4:0, 3:0, 2:0, 1:0}
    
    for x1 in range(len(X)):
        if used_indices.get(x1): continue
        answer_count = [0,0,0]
        answer_count[X[x1][1:].index(y[x1])] += 1
        current_indices = [x1]

        for x2 in range(len(X)):
            if (used_indices.get(x2)) or (x1 == x2): continue
            if(X[x1] == X[x2]):
                answer_count[X[x2][1:].index(y[x2])] += 1
                used_indices[x2] = 1
                current_indices.append(x2)
        used_indices[x1] = 1
        total_agreement += max(answer_count)
        agreement_counter[max(answer_count)] += 1

    #print(agreement_counter)

    return (total_agreement / len(X))
    






def get_word(in_word, dictionary, synset_dic, embeddings, bigram_split=False):
    '''
    if '_' in in_word:
        w1, w2 = in_word.split("_")
        index1 = dictionary.get(matrix_priors.get_synset_and_strip(w1)[1])
        index2 = dictionary.get(matrix_priors.get_synset_and_strip(w2)[1])

        embed1 = embeddings[index1]
        embed2 = embeddings[index2]

        n_embed1 = embed1 / np.linalg.norm(embed1)
        n_embed2 = embed2 / np.linalg.norm(embed2)

        embeds = (embed1, embed2)
        n_embeds = (n_embed1, n_embed2)

        indices = (index1, index2)

    else:
    '''
    index = dictionary.get(matrix_priors.get_synset_and_strip(in_word)[1])
    embed = embeddings[index]
    embeds = (embed, None)
    n_embed = embed/ np.linalg.norm(embed)
    n_embeds = (n_embed, None)
    indices = (index, None)
        
    #n_embed = embed / np.linalg.norm(embed)
    return indices, embeds, n_embeds
        
    

def get_word_primary(in_word, dictionary, synset_dic, embeddings, bigram_split=False):
    #index = dictionary.get(synset_dic.get_synset_and_strip(in_word)[1])
    '''
    if '_' in in_word:
        w1, w2 = in_word.split("_")
        index1 = dictionary.get(matrix_priors.get_synset_and_strip(w1)[1])
        index2 = dictionary.get(matrix_priors.get_synset_and_strip(w2)[1])
    
        embed = (embeddings[index1] + embeddings[index2])/2
        n_embed = embed/np.linalg.norm(embed) 

        indices = (index1,index2)
    else:
    '''
    index = dictionary.get(matrix_priors.get_synset_and_strip(in_word)[1])
    embed = embeddings[index]
    n_embed = embed/np.linalg.norm(embed)

    indices = (index, None)

    return indices, embed, n_embed





def evaluate_word2vec_cosine(X, y, embeddings, weights, dictionary, outfile_name, bigram_split=False, discard_instances=False):
    outfile_name = os.path.join(os.getcwd(), "..", outfile_name)
    out_file = open(outfile_name, 'w') 
    hist_file = open(os.path.join(os.getcwd(), "..", "choices_res.csv"), 'a')

    total_correct = 0
    first_choice = 0
    second_choice = 0
    third_choice = 0

    return_x = []
    return_y = []

    for case in range(len(X)):
        p = X[case][0]          #primary
        other_objects = X[case][1:] 
        z = y[case]             #answer

        indices = []
        nembeddings = []


        p_indices, p_embedding, p_nembedding = get_word_primary(p, dictionary, matrix_priors, embeddings, bigram_split)
        z_indices, z_embeddings, z_nembeddings = get_word(z, dictionary, matrix_priors, embeddings, bigram_split)
         
        for obj in other_objects:
            index_set, embed, nembeds = get_word(obj, dictionary, matrix_priors, embeddings, bigram_split) 
            indices.append(index_set)
            nembeddings.append(nembeds)



        embedding_sim_vector = []
        if bigram_split:
            for embed_index in range(len(nembeddings)):
                embeds = nembeddings[embed_index]
                current_indices = indices[embed_index]
                if current_indices[-1] == None:
                    sim = np.dot(p_nembedding, embeds[0])
                    embedding_sim_vector.append(sim)
                else:
                    sim1 = np.dot(p_nembedding, embeds[0])
                    sim2 = np.dot(p_nembedding, embeds[1])
                    embedding_sim_vector.append(np.mean([sim1,sim2]))
            
        else:
            for embed in nembeddings: 
                embedding_sim_vector.append(np.dot(p_nembedding, embed[0]))
        #FOR DEBUGGING PURPOSES
        #print("SIM SCORE: ", max(embedding_sim_vector), p, X[case][np.argmax(embedding_sim_vector)+1], embedding_sim_vector)

        answer_index = X[case][1:].index(y[case])
        machine_answer_index = np.argmax(embedding_sim_vector)

        first_choice_index = np.argmax(embedding_sim_vector)

        num_choices = len(embedding_sim_vector) 
        second_choice_index = -1
        third_choice_index = -1
            
        if (num_choices >= 3):
            #print("SIM VECTOR!")
            #print(len(embedding_sim_vector))
            #print(embedding_sim_vector)
            third_choice_index = embedding_sim_vector.index(np.partition(embedding_sim_vector, -3)[-3])
        if (num_choices >= 2):
            second_choice_index = embedding_sim_vector.index(np.partition(embedding_sim_vector, -2)[-2])

        if (first_choice_index == X[case][1:].index(y[case])):
            first_choice += 1
        elif (second_choice_index == X[case][1:].index(y[case])):
            second_choice += 1
        elif (third_choice_index == X[case][1:].index(y[case])):
            third_choice += 1

        out_file.write("%s, %s \n" % (p, z))

        if(np.argmax(embedding_sim_vector) == X[case][1:].index(y[case])):
            total_correct += 1
        else:
            if discard_instances:
                return_x.append(X[case])
                return_y.append(y[case])

    res = ("first choice: " + str(first_choice) + " second choice: " + str(second_choice) + " third choice: " + str(third_choice) + "\n")
    hist_file.write(res)  
    hist_file.close() 
    out_file.write(res) 

    if discard_instances:
        return return_x, return_y
    else:
        return total_correct/len(X)

