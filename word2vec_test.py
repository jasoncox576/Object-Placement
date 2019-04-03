#from word2vec_basic import *
from word2vec_eval import *

import csv

if __name__=="__main__":
    #bigram_filename = '/home/justin/Data/fil9_bigram'
    #bigram_filename = 'modified_text'
    bigram_filename = 'modified_text'
    #turk_data_filename = 'cleaned_results.csv'
    turk_data_filename = 'final_cleaned_results.csv'

    bigram_dictionaries = get_pretrain_dictionaries(bigram_filename) 
    bigram_unused_dictionary = bigram_dictionaries[2]

    accs_pretrain = np.zeros(4)
    accs = np.zeros(4)  

    X, y = get_train_test(turk_data_filename) 
    print(len(X))
    print(len(y))
  
    word2vec_acc = 0
    word2vec_alt_acc = 0

    #initial_bigram_embeddings, initial_bigram_weights = word2vec_basic('log/fil9_bigram', bigram_filename, retraining=False, X=None, y=None, dictionaries=None, get_embeddings=True)
    initial_bigram_embeddings, initial_bigram_weights = word2vec_turk('log', bigram_filename, retraining=False, X=None, y=None, dictionaries=None, get_embeddings=True)

    word2vec_acc, word2vec_alt_acc = evaluate_word2vec(X, y, initial_bigram_embeddings, initial_bigram_weights, bigram_unused_dictionary, "results.csv")

    print("word2vec_acc: ", word2vec_acc)
    print("word2vec_alt_acc: ", word2vec_alt_acc)




    #NOW we retrain the model and test it AGAIN.

    embeddings, weights = retrain_model_and_get_embeddings(X, y, bigram_dictionaries, 'modified_text')
    word2vec_acc, word2vec_alt_acc = evaluate_word2vec(X, y, embeddings, weights, bigram_unused_dictionary, "results.csv") 
    
    print('\n')
    print("AFTER RETRAINING:::")
    print("word2vec_acc: ", word2vec_acc)
    print("word2vec_alt_acc: ", word2vec_alt_acc)


    #Now we retrain using bigram-split method and test.

    print("\nTraining Using Bigram-Split:")
    embeddings, weights = retrain_model_and_get_embeddings(X, y, bigram_dictionaries, 'modified_text', bigram_split=True)
    word2vec_acc, word2vec_alt_acc = evaluate_word2vec(X, y, embeddings, weights, bigram_unused_dictionary, "results.csv")
    print('\n')
    print("AFTER RETRAINING:::")
    print("word2vec_acc: ", word2vec_acc)
    print("word2vec_alt_acc: ", word2vec_alt_acc)
