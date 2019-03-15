import csv
import spacy
from nltk.corpus import wordnet as wn
from sklearn.model_selection import StratifiedKFold
import word2vec_basic
import numpy as np
import matrix_priors
import random

print("Loading SPACY in eval..")
#nlp = spacy.load('en_core_web_lg')
nlp = matrix_priors.nlp
print("Loading complete: eval beginning")

#important columns:
#27-31 starting from 0

matrix_correct_counter = 0

random_correct_counter = 0

def roll_probs(matrix_prob_vector):

        matrix_prob_vector /= np.sum(matrix_prob_vector) 
        
        selected_index = np.argmax(np.random.multinomial(1, matrix_prob_vector))
    
        return selected_index


filename = "dummy_results.csv"
print("trying to get prior matrix")
#prior_matrix, rows_dict = matrix_priors.fill_matrix("dummy_priors.csv")
#print(rows_dict)
#word2vec_matrix = matrix_priors.fill_matrix_word2vec(rows_dict)

#print("WORD2VEC MATRIX:", word2vec_matrix)


def verify_data():

    """
    Data is 'bad' if object A is not placed with itself
    when there is an opportunity to do so

    mistakes is the number of times this occurs,
    total_sames is the number of times an object is already
    on the shelf
    """

    mistakes = 0
    total_sames = 0

    with open(filename) as csvfile:
        #reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        reader = csv.reader(csvfile)
        for row in reader:
            if reader.line_num == 1:
                continue

            row_result = row[27:32]


            answer_label = row_result[4]
            answer = (["Top", "Middle", "Bottom"].index(answer_label)) + 1
            
            if row_result[0] in row_result[1:]:
                if row_result[0] != row_result[answer]:
                    print("BAD DATA!!")
                    print(row_result)
                    print(row_result[0], row_result[answer])
                    print("====================================")
                    mistakes += 1
                total_sames += 1

    return mistakes, total_sames, mistakes/total_sames


def instances_disagree(X, y):
   
    used_indices = []
    
    for x1 in range(len(X)):
        if x1 in used_indices: continue
        class_disagreements = [(X[x1], y[x1])]
        for x2 in range(len(X)):
            if (x2 in used_indices) or (x1 == x2): continue
            if (X[x1] == X[x2]) and (y[x1] != y[x2]):
                class_disagreements.append((X[x2], y[x2]))
                used_indices.append(x2)
        used_indices.append(x1)
        print("CLASS DISAGREEMENTS----------------")
        print(class_disagreements)
    





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


                answer_label = row_result[4]
                answer = (["Top", "Middle", "Bottom"].index(answer_label)) + 1

                X.append([row_result[0], row_result[1], row_result[2], row_result[3]])
                #Fetch the answer to the most recent instance
                y.append(X[-1][answer])
                
        

        skf = StratifiedKFold(n_splits=5, shuffle=True)
        print(rows_dict)
        y_numerical = y[:]
        for index, word in enumerate(y):
            row = rows_dict[word]
            y_numerical[index] = row 
        """
        for train, test in skf.split(X, y_numerical):
            print(train, test)
        """
        
        return X, y, skf.split(X, y_numerical)

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

def retrain_model(X, y, train):

    train_X = [X[x] for x in train]
    train_y = [y[x] for x in train]

    final_embeddings, dictionary = word2vec_basic('.log', retraining=True, X=train_X, y=train_y)
    return final_embeddings, dictionary



def evaluate_word2vec_wordnet(X, y, test, rows_dict, final_embeddings, dictionary):

    # test is a one-dimensional array.
    # Run this as many times as you want, average the accuracy.

    total_correct_w2v = 0
    total_correct_wordnet = 0

    
    for case in test:

        primary = X[case][0] 

        primary_syn, primary_token = matrix_priors.get_synset_and_strip(primary)
        primary_embedding = final_embeddings[dictionary.get(primary_token)]
        #primary_token = nlp(primary_token)
        
        a = X[case][1]
        a_syn, a_token = matrix_priors.get_synset_and_strip(a)
        a_embedding = final_embeddings[dictionary.get(a_token)]
        #a_token = nlp(a_token)

        b = X[case][2]
        b_syn, b_token = matrix_priors.get_synset_and_strip(b)
        b_embedding = final_embeddings[dictionary.get(b_token)]
        #b_token = nlp(b_token)

        c = X[case][3]
        c_syn, c_token = matrix_priors.get_synset_and_strip(b)
        c_embedding = final_embeddings[dictionary.get(c_token)]
        #c_token = nlp(c_token)     
         
        embedding_sim_vector = []
        embedding_sim_vector.append(tf.matmul(primary_embedding, a_embedding, transpose_b=True))
        embedding_sim_vector.append(tf.matmul(primary_embedding, b_embedding, transpose_b=True))
        embedding_sim_vector.append(tf.matmul(primary_embedding, c_embedding, transpose_b=True))

        wordnet_sim_vector = []
        wordnet_sim_vector.append(primary_syn.path_similarity(a_syn))
        wordnet_sim_vector.append(primary_syn.path_similarity(b_syn))
        wordnet_sim_vector.append(primary_syn.path_similarity(c_syn))
    
        if(np.argmax(embedding_sim_vector)+1 == X[case].index(y[case])):
            total_correct_w2v += 1
        if(np.argmax(wordnet_sim_vector)+1 == X[case].index(y[case])):
            total_correct_wordnet += 1

    return (total_correct_w2v/len(test), total_correct_wordnet/len(test))


if __name__=="__main__":

    #print(verify_data())
    #exit()


    X, y, split = get_train_test() 
    
    word2vec_acc = 0
    wordnet_acc = 0
    empirical_acc = 0
    random_acc = 0

    train = []
    test = []
    
    for train_i, test_i in split:
        train.append(train_i)
        test.append(test_i)

    print("TRAIN")
    print(train)

    for test_num in range(len(test)):
       final_embeddings, dictionary = retrain_model(X, y, train[test_num]) 
       next_word2vec_acc, next_wordnet_acc = evaluate_word2vec_wordnet(X, y, test[test_num], rows_dict, final_embeddings, dictionary) 
       next_random_acc = eval_random(X, y, test[test_num])

       print("next word2vec acc: ", next_word2vec_acc)
       print("next wordnet acc: ", next_wordnet_acc)
       print("next random acc: ", next_random_acc)
       word2vec_acc += next_word2vec_acc
       wordnet_acc += next_wordnet_acc
       random_acc += next_random_acc
   
    word2vec_acc /= len(test)
    wordnet_acc /= len(test)
    random_acc /= len(test)
       

    for test_num in range(len(test)):
        empirical_matrix = matrix_priors.fill_empirical_matrix(X, y, train[test_num], rows_dict)

        next_empirical_acc = evaluate_empirical(empirical_matrix, X, y, test[test_num], rows_dict)
        print("next empirical acc: ", next_empirical_acc)
        empirical_acc += next_empirical_acc
    empirical_acc /= len(test)

    print("Averaged accuracy of word2vec is:")
    print(str(word2vec_acc))

    print("Averaged accuracy of wordnet is:")
    print(str(wordnet_acc))

    print("Averaged accuracy of empirical is:")
    print(str(empirical_acc))
    

    print("Averaged accuracy of random guessing is:")
    print(str(random_acc))
    """
    print("\nTotal accuracy of matrix is:")
    print(str(matrix_correct_counter / total_row_counter))

    print("\nTotal accuracy of word2vec matrix is:")
    print(str(word2vec_matrix_correct_counter / total_row_counter)) 

    print("\nTotal accuracy of empirical matrix is:")
    print(str(empirical_matrix_correct_counter / total_row_counter))
    """




    print("================================================")
    print("Looking at disagreeance in data")

    instances_disagree(X, y)
    











