import csv
import spacy
import numpy as np

print("Loading SPACY...")
nlp = spacy.load('en_core_web_lg')
print("Loading complete")

rows_dict = {}
PRIOR_MATRIX = []


def strip(word):
    if word[-2] == '_':
        word = word[:-2]
    
    return word

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
            row = rows_dict.get(word1)
            col = rows_dict.get(word2)

            token1 = nlp(strip(word1)) 
            token2 = nlp(strip(word2))

            prior_matrix[row][col] = token1.similarity(token2) 

    return prior_matrix    



PRIOR_MATRIX, rows_dict = fill_matrix('dummy_priors.csv')
print(rows_dict)
print(str(fetch_sim('gum', 'nut', PRIOR_MATRIX)))
print(str(fetch_sim('oatmeal', 'coffee', PRIOR_MATRIX)))




