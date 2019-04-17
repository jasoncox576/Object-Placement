#from word2vec_basic import *
from word2vec_eval import *
from wordnet_test import *

import csv

if __name__=="__main__":
    #bigram_filename = '/home/justin/Data/fil9_bigram'
    #bigram_filename = 'modified_text'
    bigram_filename = 'modified_text'
    #turk_data_filename = 'cleaned_results.csv'
    turk_data_filename = 'final_cleaned_results.csv'

    bigram_dictionaries = get_pretrain_dictionaries(bigram_filename) 
    bigram_unused_dictionary = bigram_dictionaries[2]

    with open("accs_final.csv", 'w') as accuracies_file:
            results_writer = csv.writer(accuracies_file)

            #X, y = get_train_test(turk_data_filename) 
            test1 = read_csv_train_test("data/1_test.csv")
            test2 = read_csv_train_test("data/2_test.csv")
            test3 = read_csv_train_test("data/3_test.csv")
            test4 = read_csv_train_test("data/4_test.csv")
            test5 = read_csv_train_test("data/5_test.csv")


            test_sets = [test1, test2, test3, test4, test5]

            for set_num in range(len(test_sets)):
                        

                print("TEST SET #" + str(set_num+1) + ":::")
                print("=================================================================")
                test_x, test_y = test_sets[set_num]
                

                # THIS IS FOR TEST #1
                # Model trained exclusively on wikipedia

                initial_bigram_embeddings, initial_bigram_weights = get_embeddings("wiki", bigram_filename, bigram_dictionaries)

                cosine_acc = evaluate_word2vec_cosine(test_x, test_y, initial_bigram_embeddings, initial_bigram_weights, bigram_unused_dictionary, "results.csv")
                output_acc = evaluate_word2vec_output(test_x, test_y, initial_bigram_embeddings, initial_bigram_weights, bigram_unused_dictionary, "results.csv")


                print("TEST #1:::")
                print("cosine acc: ", cosine_acc)
                print("output acc: ", output_acc)
                results_writer.writerow([cosine_acc, output_acc])


                # THIS IS FOR TEST #2
                # Model trained on wikipedia and retrained on turk
                cosine_embeddings, cosine_weights = get_embeddings(str(set_num+1)+"_wiki+turk_cosine", bigram_filename, bigram_dictionaries)
                output_embeddings, output_weights = get_embeddings(str(set_num+1)+"wiki+turk_output", bigram_filename, bigram_dictionaries)

                cosine_acc = evaluate_word2vec_cosine(test_x, test_y, cosine_embeddings, cosine_weights, bigram_unused_dictionary, "results.csv") 
                output_acc = evaluate_word2vec_output(test_x, test_y, output_embeddings, output_weights, bigram_unused_dictionary, "results.csv") 
                
                print('\n')
                print("TEST #2:::")
                print("cosine acc: ", cosine_acc)
                print("output acc: ", output_acc)
                results_writer.writerow([cosine_acc, output_acc])


                # THIS IS FOR TEST #3
                # Model trained exclusively on turk
                cosine_embeddings, cosine_weights = get_embeddings(str(set_num+1)+"_turk_cosine", bigram_filename, bigram_dictionaries)
                output_embeddings, output_weights = get_embeddings(str(set_num+1)+"_turk_output", bigram_filename, bigram_dictionaries)
                cosine_acc = evaluate_word2vec_cosine(test_x, test_y, cosine_embeddings, cosine_weights, bigram_unused_dictionary, "results.csv")
                output_acc = evaluate_word2vec_output(test_x, test_y, output_embeddings, output_weights, bigram_unused_dictionary, "results.csv")

                print("TEST #3:::")
                print("cosine acc: ", cosine_acc)
                print("output acc: ", output_acc)
                results_writer.writerow([cosine_acc, output_acc])


                # THIS IS FOR TEST #4
                # load the model that was just trained on turk, train it on wikipedia,
                # test.
                cosine_embeddings, cosine_weights = word2vec_turk('log', bigram_filename, retraining=False, X=None, y=None, dictionaries=None, get_embeddings=True)
                cosine_embeddings, cosine_weights = get_embeddings(str(set_num+1)+"turk+wiki_cosine", bigram_filename, bigram_dictionaries)
                output_embeddings, output_weights = get_embeddings(str(set_num+1)+"turk+wiki_output", bigram_filename, bigram_dictionaries)
                

                cosine_acc = evaluate_word2vec_cosine(test_x, test_y, cosine_embeddings, cosine_weights, bigram_unused_dictionary, "results.csv")
                output_acc = evaluate_word2vec_output(test_x, test_y, output_embeddings, output_weights, bigram_unused_dictionary, "results.csv")


                print("TEST #4:::")
                print("cosine acc: ", cosine_acc)
                print("output acc: ", output_acc)
                results_writer.writerow([cosine_acc, output_acc])


                # THIS IS FOR TEST #5
                #Now we retrain using bigram-split method and test on model trained on wikipedia + turk.

                cosine_embeddings, cosine_weights = get_embeddings(str(set_num+1)+"_bigram_cosine", bigram_filename, bigram_dictionaries) 
                output_embeddings, output_weights= get_embeddings(str(set_num+1)+"_bigram_output", bigram_filename, bigram_dictionaries) 



                cosine_acc = evaluate_word2vec_cosine(test_x, test_y, cosine_embeddings, cosine_weights, bigram_unused_dictionary, "results.csv")
                output_acc = evaluate_word2vec_output(test_x, test_y, output_embeddings, output_weights, bigram_unused_dictionary, "results.csv")
                print('\n')
                print("TEST #5:::")
                print("cosine acc: ", cosine_acc)
                print("output acc: ", output_acc)
                results_writer.writerow([cosine_acc, output_acc])


                
                # THIS IS FOR TEST #6
                # WordNet evaluation
                wordnet_acc = evaluate_wordnet(test_x, test_y, bigram_unused_dictionary)
                print('\n')
                print("TEST #6:::")
                print("WordNet acc: ", wordnet_acc)
                results_writer.writerow([wordnet_acc])
                


                joint_embeddings, joint_weights = get_embeddings(str(set_num+1)+"_joint", bigram_filename, bigram_dictionaries)
                cosine_acc = evaluate_word2vec_cosine(test_x, test_y, joint_embeddings, joint_weights, bigram_unused_dictionary, "results.csv")
                output_acc = evaluate_word2vec_output(test_x, test_y, joint_embeddings, joint_weights, bigram_unused_dictionary, "results.csv")
                
                print("TEST #7:::")
                print("cosine acc: ", cosine_acc)
                print("output acc: ", output_acc)
                results_writer.writerow([cosine_acc, output_acc])



                print("=================================================================")
    


