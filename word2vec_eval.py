import csv
import spacy
from nltk.corpus import wordnet as wn
from sklearn.model_selection import StratifiedKFold
from object_placement_turk import *
#from word2vec_basic import *
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
  #index = dictionary.get(synset_dic.get_synset_and_strip(in_word)[1])
  if '_' in in_word:
      middle = in_word.index('_')
      index1 = dictionary.get(matrix_priors.get_synset_and_strip(in_word[:middle])[1])
      index2 = dictionary.get(matrix_priors.get_synset_and_strip(in_word[middle+1:])[1])
    
      embed = (embeddings[index1] + embeddings[index2])/2
      n_embed = embed/np.linalg.norm(embed) 

      index = 0
      print("BIGRAM WORD! TOOK MEAN EMBEDDING of ", in_word[:middle], "and", in_word[middle+1:])
       
  else:
      index = dictionary.get(matrix_priors.get_synset_and_strip(in_word)[1])
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


"""

if __name__=="__main__":
    bigram_filename = '/home/justin/Data/fil9'
    #bigram_filename = 'modified_text'
    #original_filename = 'text8'

    #train_original_model(filename=original_filename, bigrams=False)
    if not MODEL_EXISTS_ALREADY:
        train_original_model(filename=bigram_filename) 
    
    bigram_dictionaries = get_pretrain_dictionaries(bigram_filename) 
    bigram_unused_dictionary = bigram_dictionaries[2]

    accs_pretrain = np.zeros(4)
    accs = np.zeros(4)
    #regular_dictionaries = get_pretrain_dictionaries(original_filename)
    #regular_unused_dictionary = regular_dictionaries[2]
    

    X, y, split = get_train_test(bigram_unused_dictionary, filename) 
    
    word2vec_acc = 0
    word2vec_alt_acc = 0
    wordnet_acc = 0
    retrained_acc = 0
    retrained_alt_acc = 0
    empirical_acc = 0
    random_acc = 0


    train = []
    test = []

    #Does no training, just a hacky way to get the embeddings off of the original base model without trying to load the tensor individually.
    initial_bigram_embeddings, initial_bigram_weights = word2vec_basic('log', bigram_filename, retraining=False, X=None, y=None, dictionaries=None, get_embeddings=True)
    #initial_regular_embeddings = word2vec_basic('log2', original_filename, retraining=False, X=None, y=None, dictionaries=None, get_embeddings=True)
    
    
    for train_i, test_i in split:
        train.append(train_i)
        test.append(test_i)

    print("Beginning to retrain word2vec and evaluate models...")
    #print(train)

    for test_num in range(len(test)):
       next_word2vec_acc, next_word2vec_alt_acc, new_accs_pretrain = evaluate_word2vec(X, y, test[test_num], initial_bigram_embeddings, initial_bigram_weights, bigram_unused_dictionary)
       final_bigram_embeddings, final_bigram_weights = retrain_model(bigram_filename, X, y, train[test_num], bigram_dictionaries) 
       next_retrained_acc, next_retrained_alt_acc, new_accs = evaluate_word2vec(X, y, test[test_num], final_bigram_embeddings, final_bigram_weights, bigram_unused_dictionary)
       next_wordnet_acc = evaluate_wordnet(X, y, test[test_num], bigram_unused_dictionary)
       next_random_acc = eval_random(X, y, test[test_num])

       print("next word2vec acc: ", next_word2vec_acc)
       print("next word2vec alt acc: ", next_word2vec_alt_acc)
       print("next retrained word2vec acc: ", next_retrained_acc)
       print("next retrained alt acc: ", next_retrained_alt_acc)
       print("next wordnet acc: ", next_wordnet_acc)
       print("next random acc: ", next_random_acc)
       word2vec_acc += next_word2vec_acc
       word2vec_alt_acc += next_word2vec_alt_acc
       retrained_acc += next_retrained_acc
       retrained_alt_acc += next_retrained_alt_acc
       wordnet_acc += next_wordnet_acc
       random_acc += next_random_acc
       accs = np.add(accs, new_accs)
       accs_pretrain = np.add(accs_pretrain, new_accs_pretrain)
   
    word2vec_acc /= len(test)
    word2vec_alt_acc /= len(test)
    retrained_acc /= len(test)
    retrained_alt_acc /= len(test)
    wordnet_acc /= len(test)
    random_acc /= len(test)
    accs = np.divide(accs, len(test))
    accs_pretrain = np.divide(accs_pretrain, len(test))
       

    #for test_num in range(len(test)):
    #    empirical_matrix = matrix_priors.fill_empirical_matrix(X, y, train[test_num], rows_dict)
    #
    #    next_empirical_acc = evaluate_empirical(empirical_matrix, X, y, test[test_num], rows_dict)
    #    print("next empirical acc: ", next_empirical_acc)
    #    empirical_acc += next_empirical_acc
    #empirical_acc /= len(test)


    print("Averaged accuracy of word2vec is:")
    print(str(word2vec_acc))

    print("Averaged accuracy of word2vec alt is: ")
    print(str(word2vec_alt_acc))

    print("Averaged accuracy of retrained word2vec is:")
    print(str(retrained_acc))

    print("Averaged accuracy of retrained alt is:")
    print(str(retrained_alt_acc))

    print("Averaged accuracy of wordnet is:")
    print(str(wordnet_acc))

    print("Averaged accuracy of empirical is:")
    print(str(empirical_acc))
    

    print("Averaged accuracy of random guessing is:")
    print(str(random_acc))


    print("Averaged accs bigram/unigram")
    print(accs)

    print("Averaged pretrain accs bigram/unigram")
    print(accs_pretrain)

    #print("\nTotal accuracy of matrix is:")
    #print(str(matrix_correct_counter / total_row_counter))

    #print("\nTotal accuracy of word2vec matrix is:")
    #print(str(word2vec_matrix_correct_counter / total_row_counter)) 

    #print("\nTotal accuracy of empirical matrix is:")
    #print(str(empirical_matrix_correct_counter / total_row_counter))




    print("================================================")
    print("Looking at disagreeance in data")

    #instances_disagree(X, y)
"""    


