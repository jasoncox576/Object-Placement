import csv
import spacy
from nltk.corpus import wordnet as wn
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matrix_priors
import random

print("Loading SPACY in eval..")
#nlp = spacy.load('en_core_web_lg')
nlp = matrix_priors.nlp
print("Loading complete: eval beginning")

#important columns:
#27-31 starting from 0

total_row_counter = 0
embedding_correct_counter = 0
wordnet_correct_counter = 0
matrix_correct_counter = 0
word2vec_matrix_correct_counter = 0
empirical_matrix_correct_counter = 0

random_correct_counter = 0

def roll_probs(matrix_prob_vector):

        matrix_prob_vector /= np.sum(matrix_prob_vector) 
        
        selected_index = np.argmax(np.random.multinomial(1, matrix_prob_vector))
    
        return selected_index


filename = "dummy_results.csv"
print("trying to get prior matrix")
prior_matrix, rows_dict = matrix_priors.fill_matrix("dummy_priors.csv")
print(rows_dict)
empirical_matrix = matrix_priors.fill_empirical_matrix("dummy_results.csv", rows_dict)
print("GOT EMPIRICAL MATRIX:", empirical_matrix)
word2vec_matrix = matrix_priors.fill_matrix_word2vec(rows_dict)

print("WORD2VEC MATRIX:", word2vec_matrix)

def get_train_test():

        X = []
        y = []


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

                primary_syn, primary_token = matrix_priors.get_synset_and_strip(primary) 
                primary_token = nlp(primary_token)

                answer = row_result[4]

                synset1, token1 = matrix_priors.get_synset_and_strip(row_result[1])
                token1 = nlp(token1)

                synset2, token2 = matrix_priors.get_synset_and_strip(row_result[2])
                token2 = nlp(token2)

                synset3, token3 = matrix_priors.get_synset_and_strip(row_result[3])
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

                empirical_matrix_prob_vector = []
                empirical_matrix_prob_vector.append(empirical_matrix[rows_dict.get(primary)][rows_dict.get(row_result[1])])
                empirical_matrix_prob_vector.append(empirical_matrix[rows_dict.get(primary)][rows_dict.get(row_result[2])])
                empirical_matrix_prob_vector.append(empirical_matrix[rows_dict.get(primary)][rows_dict.get(row_result[3])])
            
                X.append([row_result[4], row_result[1], row_result[2], row_result[3]])
                #Fetch the answer to the most recent instance
                y.append(X[-1][answer+1])

                if np.argmax(embedding_sim_vector) == answer:
                    embedding_correct_counter += 1
                if np.argmax(wordnet_sim_vector) == answer:
                    wordnet_correct_counter += 1
                if np.argmax(matrix_prob_vector) == answer:
                    matrix_correct_counter += 1
                
                if np.argmax(empirical_matrix_prob_vector) == answer:
                    empirical_matrix_correct_counter += 1


                # Computing the choice for the prob matrix isn't
                # deterministic, so it is slightly more complicated.
                # We must roll for probabilities.
                
                """
                selected_index = roll_probs(matrix_prob_vector)
                if selected_index == answer:
                    matrix_correct_counter += 1
                """

                selected_index_w2v = roll_probs(word2vec_matrix_prob_vector)
                if selected_index_w2v == answer:
                    word2vec_matrix_correct_counter += 1

                """
                selected_index_empirical = roll_probs(empirical_matrix_prob_vector)
                if selected_index_empirical == answer:
                    empirical_matrix_correct_counter += 1 
                """

                total_row_counter += 1


        skf = StratifiedKFold(n_splits=5, shuffle=True)
        print(rows_dict)
        for index, word in enumerate(y):
            row = rows_dict[word]
            y[index] = row 


        print(len(X))
        print(y)
        for train, test in skf.split(X, y):
            print("TRAIN", train)
            print("TEST", test)
        
        return X, y, skf.split(X, y)


def evaluate_word2vec_wordnet(X, y, test, rows_dict):

    # test is a one-dimensional array.
    # Run this as many times as you want, average the accuracy.

    total_correct_w2v = 0
    total_corect_wordnet = 0
    
    for case in test:

        primary = X[case][0] 

        primary_syn, primary_token = matrix_priors.get_synset_and_strip(primary)
        primary_token = nlp(primary_token)
        
        a = X[case][1]
        a_syn, a_token = matrix_priors.get_synset_and_strip(a)
        a_token = nlp(a_token)

        b = X[case][2]
        b_syn, b_token = matrix_priors.get_synset_and_strip(b)
        b_token = nlp(b_token)

        c = X[case][3]
        c_syn, c_token = matrix_priors.get_synset_and_strip(b)
        c_token = nlp(c_token)     
         
        embedding_sim_vector = []
        embedding_sim_vector.append(primary_token.similarity(a_token))
        embedding_sim_vector.append(primary_token.similarity(b_token))
        embedding_sim_vector.append(primary_token.similarity(c_token))

        wordnet_sim_vector = []
        wordnet_sim_vector.append(
    
        if(np.argmax(embedding_sim_vector)+1 == X[case].index(y[case])):
            total_correct += 1

    return total_correct / len(X)

def evaluate_wordnet(X, y, test, rows_dict):
    
    total_correct = 0

    for case in test:
        
        primary = X[case][0]
        



print("Total accuracy of word2vec is:")
print(str(embedding_correct_counter / total_row_counter))

print("\nTotal accuracy of wordnet is:")
print(str(wordnet_correct_counter / total_row_counter))

print("\nTotal accuracy of matrix is:")
print(str(matrix_correct_counter / total_row_counter))

print("\nTotal accuracy of word2vec matrix is:")
print(str(word2vec_matrix_correct_counter / total_row_counter)) 

print("\nTotal accuracy of empirical matrix is:")
print(str(empirical_matrix_correct_counter / total_row_counter))
