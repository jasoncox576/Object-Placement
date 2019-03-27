from word2vec_basic import *
from word2vec_eval import *

import csv

if __name__=="__main__":
  bigram_filename = '/home/justin/Data/fil9'
  turk_data_filename = 'official_results.csv'

  bigram_dictionaries = get_pretrain_dictionaries(bigram_filename) 
  bigram_unused_dictionary = bigram_dictionaries[2]

  accs_pretrain = np.zeros(4)
  accs = np.zeros(4)  

  X, y, split = get_train_test(bigram_unused_dictionary, turk_data_filename) 
  
  word2vec_acc = 0
  word2vec_alt_acc = 0

  train = []
  test = []

  initial_bigram_embeddings, initial_bigram_weights = word2vec_load('log', bigram_filename, retraining=False, X=None, y=None, dictionaries=None, get_embeddings=True)

  for train_i, test_i in split:
      train.append(train_i)
      test.append(test_i)

  print("Beginning to retrain word2vec and evaluate models...")
  #print(train)

  for test_num in range(len(test)):
     next_word2vec_acc, next_word2vec_alt_acc, new_accs_pretrain = evaluate_word2vec(X, y, test[test_num], initial_bigram_embeddings, initial_bigram_weights, bigram_unused_dictionary)

     print("next word2vec acc: ", next_word2vec_acc)
     print("next word2vec alt acc: ", next_word2vec_alt_acc)
     word2vec_acc += next_word2vec_acc
     word2vec_alt_acc += next_word2vec_alt_acc
 
  word2vec_acc /= len(test)
  word2vec_alt_acc /= len(test)

  print("Averaged accuracy of word2vec is:")
  print(str(word2vec_acc))

  print("Averaged accuracy of word2vec alt is: ")
  print(str(word2vec_alt_acc))
