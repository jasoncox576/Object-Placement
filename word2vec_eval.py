import csv
#import spacy
#from nltk.corpus import wordnet as wn
from object_placement_turk import *
from word2vec_basic import *
from word2vec_train_tuned import *
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


def get_word(in_word, dictionary, synset_dic, embeddings, bigram_split=False):
    
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
        index = dictionary.get(matrix_priors.get_synset_and_strip(in_word)[1])
        print("INDEX!! ", index)
        print(matrix_priors.get_synset_and_strip(in_word)[1])
        embed = embeddings[index]
        embeds = (embed, None)
        n_embed = embed/ np.linalg.norm(embed)
        n_embeds = (n_embed, None)
        indices = (index, None)
        
    #n_embed = embed / np.linalg.norm(embed)
    return indices, embeds, n_embeds
        
    

def get_word_primary(in_word, dictionary, synset_dic, embeddings, bigram_split=False):
    #index = dictionary.get(synset_dic.get_synset_and_strip(in_word)[1])
    if '_' in in_word:
        w1, w2 = in_word.split("_")
        index1 = dictionary.get(matrix_priors.get_synset_and_strip(w1)[1])
        index2 = dictionary.get(matrix_priors.get_synset_and_strip(w2)[1])
    
        embed = (embeddings[index1] + embeddings[index2])/2
        n_embed = embed/np.linalg.norm(embed) 

        indices = (index1,index2)
    else:
        index = dictionary.get(matrix_priors.get_synset_and_strip(in_word)[1])
        embed = embeddings[index]
        n_embed = embed/np.linalg.norm(embed)

        indices = (index, None)

    return indices, embed, n_embed





def evaluate_word2vec_cosine(X, y, embeddings, weights, dictionary, outfile_name, bigram_split=False, discard_instances=False):
    out_file = open(outfile_name, 'w') 
    hist_file = open("choices_res.csv", 'a')

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




def evaluate_word2vec_output(X, y, embeddings, weights, dictionary, outfile_name, bigram_split=False):
    out_file = open(outfile_name, 'w')
    hist_file = open("choices_res.csv", 'a') 

    total_correct = 0
    first_choice = 0
    second_choice = 0
    third_choice = 0

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

        output_vector = np.matmul([p_embedding], np.transpose(weights))
        output_vector = np.reshape(output_vector, (len(output_vector[0])))
        

        output_sim_vector = []

        for index_set in indices:
            if index_set[-1] == None:
                output_sim_vector.append(output_vector[index_set[0]])
            else:
                index1, index2 = index_set
                mean = np.mean([output_vector[index1], output_vector[index2]])
                output_sim_vector.append(mean)

        if(np.argmax(output_sim_vector) == X[case][1:].index(y[case])):
            total_correct += 1
        else:
            if X[case][0] in X[case][1:]:
                total_correct += 1

        first_choice_index = np.argmax(output_sim_vector)
        num_choices = len(output_sim_vector) 
        second_choice_index = -1
        third_choice_index = -1
            
        if (num_choices >= 3):
            third_choice_index = output_sim_vector.index(np.partition(output_sim_vector, -3)[-3])
        if (num_choices >= 2):
            second_choice_index = output_sim_vector.index(np.partition(output_sim_vector, -2)[-2])

        if (first_choice_index == X[case][1:].index(y[case])):
            first_choice += 1
        elif (second_choice_index == X[case][1:].index(y[case])):
            second_choice += 1
        elif (third_choice_index == X[case][1:].index(y[case])):
            third_choice += 1
        print(first_choice, second_choice, third_choice)

    res = ("first choice: " + str(first_choice) + " second choice: " + str(second_choice) + " third choice: " + str(third_choice) + "\n")
    hist_file.write(res)  
    hist_file.close()   
    out_file.close();


    return total_correct/len(X)


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
