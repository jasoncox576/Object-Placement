import csv
from nltk.corpus import wordnet as wn
import spacy
import numpy as np

print("Loading SPACY in matrix_priors")
nlp = spacy.load('en_core_web_lg')
print("Loading complete (matrix_priors)")

rows_dict = {}
PRIOR_MATRIX = []


def strip(word):
    if word[-2] == '_':
        word = word[:-2]
    
    return word






def get_synset_and_strip(word):
        
    synset_num = 0 

    if word[-2] == '_':
        synset_num = int(word[-1]) 
        word = word[:-2]
    
    synset = wn.synsets(word.replace(" ", "_"))[synset_num]    
    return (synset, word)




def fetch_sim(w1, w2, prior_matrix):
    row = rows_dict.get(w1)
    col = rows_dict.get(w2)
    
    if (row == None):
        w1_token = nlp(w1)    
        next_closest_word = ''
        max_sim = 0.0
        for word in rows_dict:
            stripped_word = strip(word)
            word_token = nlp(stripped_word)
            sim = w1_token.similarity(word_token)
            if sim >= max_sim:
                max_sim = sim 
                next_closest_word = word 
        
        print(w1, next_closest_word)
        row = rows_dict.get(next_closest_word)

    # TODO clean up later

    if (col == None):
        w2_token = nlp(w2)    
        next_closest_word = ''
        max_sim = 0.0
        for word in rows_dict:
            stripped_word = strip(word)
            word_token = nlp(stripped_word)
            sim = w2_token.similarity(word_token)
            if sim >= max_sim:
                max_sim = sim 
                next_closest_word = word 
        print(w2, next_closest_word)
        col = rows_dict.get(next_closest_word)
   
    return prior_matrix[row][col]


def fill_matrix(filename):
        with open(filename) as csvfile:
            reader = csv.reader(csvfile)
            #number of rows of the sparse matrix
            row_count = 0

            # comparison of A to A is always 1, hence prior is identity
            
            #just a declaration

            for row in reader:
                if reader.line_num == 1:
                    row_count = int(row[0])
                    prior_matrix = np.identity(row_count)
                    continue

                if reader.line_num <= row_count+1:
                    rows_dict[row[0]] = int(row[1]) 
                
                #symmetric matrix, so we fill both entries
                else:
                    obj_1 = row[0] 
                    obj_2 = row[1][1:]
                    prob = float(row[2])
                    row = rows_dict.get(obj_1)
                    col = rows_dict.get(obj_2)

                    prior_matrix[row][col] = prob
                    prior_matrix[col][row] = prob
        

        return (prior_matrix, rows_dict)

def fill_empirical_matrix(X, y, train, rows_dict):

        probs_matrix = np.identity(len(rows_dict))


        for case in train:

            primary = X[case][0]
            primary_syn, primary_token = get_synset_and_strip(primary) 
            primary_token = nlp(primary_token)


            _, token1 = get_synset_and_strip(X[case][1])
            token1 = nlp(token1)

            _, token2 = get_synset_and_strip(X[case][2])
            token2 = nlp(token2)

            _, token3 = get_synset_and_strip(X[case][3])
            token3 = nlp(token3)


            object_vector = [X[case][1], X[case][2], X[case][3]]
                    

            embedding_sim_vector = []
            embedding_sim_vector.append(primary_token.similarity(token1))
            embedding_sim_vector.append(primary_token.similarity(token2))
            embedding_sim_vector.append(primary_token.similarity(token3))

            predicted_object = object_vector[np.argmax(embedding_sim_vector)]     
                    
                        
            probs_matrix[rows_dict.get(primary)][rows_dict.get(predicted_object)] += 1 
            probs_matrix[rows_dict.get(primary)][rows_dict.get(primary)] += 1
        
        #print("PRE_DIVIDE: ", probs_matrix)
        for row in range(len(probs_matrix)):
            divide_val = probs_matrix[row][row]-1
            probs_matrix[row][row] -= 1
            if divide_val != 0:
                probs_matrix[row] = [x / divide_val for x in probs_matrix[row]]
        
        #print("POST_DIVIDE: ", probs_matrix)
        return probs_matrix         


def fill_matrix_word2vec(rows_dict):

    """ Fills a matrix with similarities between
        word embeddings, except treats similarities
        as non-deterministic probablities
    """
    
    #NOTE: Initializing to identity is redundant, because diagonal gets
    # filled in anyway as 1
    prior_matrix = np.identity(len(rows_dict))
    for word1 in rows_dict:
        for word2 in rows_dict: 
        
            word1_ind = rows_dict[word1]
            word2_ind = rows_dict[word2]


            token1 = nlp(strip(word1)) 
            token2 = nlp(strip(word2))
                
            sim = token1.similarity(token2)
            prior_matrix[word1_ind][word2_ind] = sim 

    return prior_matrix    



#PRIOR_MATRIX, rows_dict = fill_matrix('dummy_priors.csv')
#word2vec_matrix = fill_matrix_word2vec(rows_dict)
#print(rows_dict)
#print(word2vec_matrix)



