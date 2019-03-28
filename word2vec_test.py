#from word2vec_basic import *
from word2vec_eval import *

import csv

if __name__=="__main__":
    bigram_filename = '/home/justin/Data/modified_text'
    turk_data_filename = 'official_results.csv'

    bigram_dictionaries = get_pretrain_dictionaries(bigram_filename) 
    bigram_unused_dictionary = bigram_dictionaries[2]

    accs_pretrain = np.zeros(4)
    accs = np.zeros(4)  

    X, y = get_train_test(turk_data_filename) 
  
    word2vec_acc = 0
    word2vec_alt_acc = 0

    initial_bigram_embeddings, initial_bigram_weights = word2vec_basic('log', bigram_filename, retraining=False, X=None, y=None, dictionaries=None, get_embeddings=True)

    word2vec_acc, word2vec_alt_acc = evaluate_word2vec(X, y, initial_bigram_embeddings, initial_bigram_weights, bigram_unused_dictionary)

    print("word2vec_acc: ", word2vec_acc)
    print("word2vec_alt_acc: ", word2vec_alt_acc)
