import csv
import spacy
from nltk.corpus import wordnet as wn
import numpy as np
import matrix_priors
import random

print("Loading SPACY..")
nlp = spacy.load('en_core_web_lg')
print("Loading complete")

#important columns:
#27-31 starting from 0

total_row_counter = 0
embedding_correct_counter = 0
wordnet_correct_counter = 0
matrix_correct_counter = 0
word2vec_matrix_correct_counter = 0

def get_synset_and_strip(word):
    
    synset_num = 0

    if word[-2] == '_':
        synset_num = int(word[-1]) 
        word = word[:-2]
    
    synset = wn.synsets(word.replace(" ", "_"))[synset_num]    
    return (synset, word)


def roll_probs(matrix_prob_vector):

        #This should be [0,1,2] by default but will generalize to n dimensions
        possible_indices = [*range(len(matrix_prob_vector))]

        while len(possible_indices) > 1:
            for prob in matrix_prob_vector:
                roll = (random.randint(0, 100))/100
                print(str(roll) + " " + str(prob))
                if roll > prob:
                    #that object is now out of the running,
                    #we won't use it for placement
                    del_index = matrix_prob_vector.index(prob)
                    del matrix_prob_vector[del_index]
                    del possible_indices[del_index]
                    break 

        selected_index = possible_indices[0]
        return selected_index



filename = "dummy_results.csv"
prior_matrix, rows_dict = matrix_priors.fill_matrix("dummy_priors.csv")
word2vec_matrix = matrix_priors.fill_matrix_word2vec(prior_matrix)



with open(filename) as csvfile:
    #reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    reader = csv.reader(csvfile)
    for row in reader:
        if reader.line_num == 1:
            continue

        row_result = row[27:32]
        
        #for i in range(len(row_result)-1):
            #row_result[i] = row_result[i][:-6] 

        primary = row_result[0]
        #print("PRIMARY: " + primary)

        primary_syn, primary_token = get_synset_and_strip(primary) 
        primary_token = nlp(primary_token)

        answer = row_result[4]

        synset1, token1 = get_synset_and_strip(row_result[1])
        token1 = nlp(token1)

        synset2, token2 = get_synset_and_strip(row_result[2])
        token2 = nlp(token2)

        synset3, token3 = get_synset_and_strip(row_result[3])
        token3 = nlp(token3)


        if answer == "Top":
            answer = 0
        elif answer == "Middle":
            answer = 1
        elif answer == "Bottom":
            answer = 2

        embedding_sim_vector = []
        embedding_sim_vector.append(primary_token.similarity(token1))
        embedding_sim_vector.append(primary_token.similarity(token2))
        embedding_sim_vector.append(primary_token.similarity(token3))

        wordnet_sim_vector = []
        wordnet_sim_vector.append(primary_syn.path_similarity(synset1))
        wordnet_sim_vector.append(primary_syn.path_similarity(synset2))
        wordnet_sim_vector.append(primary_syn.path_similarity(synset3))

       
        matrix_prob_vector = [] 
        matrix_prob_vector.append(prior_matrix[rows_dict.get(primary)][rows_dict.get(row_result[1])])
        matrix_prob_vector.append(prior_matrix[rows_dict.get(primary)][rows_dict.get(row_result[2])])
        matrix_prob_vector.append(prior_matrix[rows_dict.get(primary)][rows_dict.get(row_result[3])])

        word2vec_matrix_prob_vector = []
        word2vec_matrix_prob_vector.append(word2vec_matrix[rows_dict.get(primary)][rows_dict.get(row_result[1])])
        word2vec_matrix_prob_vector.append(word2vec_matrix[rows_dict.get(primary)][rows_dict.get(row_result[2])])
        word2vec_matrix_prob_vector.append(word2vec_matrix[rows_dict.get(primary)][rows_dict.get(row_result[3])])

        if np.argmax(embedding_sim_vector) == answer:
            embedding_correct_counter += 1
        if np.argmax(wordnet_sim_vector) == answer:
            wordnet_correct_counter += 1
        """
        if np.argmax(matrix_prob_vector) == answer:
            matrix_correct_counter += 1
        """

        # Computing the choice for the prob matrix isn't
        # deterministic, so it is slightly more complicated.
        # We must roll for probabilities.
        
        selected_index = roll_probs(matrix_prob_vector)
        if selected_index == answer:
            matrix_correct_counter += 1

        selected_index_w2v = roll_probs(word2vec_matrix_prob_vector)
        if selected_index_w2v == answer:
            word2vec_matrix_correct_counter += 1
         

        total_row_counter += 1

print("Total accuracy of word2vec is:")
print(str(embedding_correct_counter / total_row_counter))

print("\nTotal accuracy of wordnet is:")
print(str(wordnet_correct_counter / total_row_counter))

print("\nTotal accuracy of matrix is:")
print(str(matrix_correct_counter / total_row_counter))

print("\nTotal accuracy of word2vec matrix is:")
print(str(word2vec_matrix_correct_counter / total_row_counter))
