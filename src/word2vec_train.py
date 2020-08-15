from word2vec_basic import *
from word2vec_eval import *
from data_gen import *
from nltk.corpus import wordnet as wn
import pandas as pd
import cosine_heatmap
import pickle
import sys
import os


#NOTE: This procedure trains using validation sets

def train_original_model(filename, save_dir, load_dir=None, load=False):
    return word2vec_turk(save_dir, load_dir=load_dir, filename=filename, retraining=False, X=None, y=None, dictionaries=None, bigram_split=False, load=load)

def retrain_model_and_get_embeddings(X, y, dictionaries, filename, cosine, load_dir, save_dir, bigram_split=True, joint_training=False, data_index_dir="data_index"):
    return word2vec_turk(save_dir, load_dir, filename=filename, retraining=True, X=X, y=y, dictionaries=dictionaries, bigram_split=bigram_split, load=True, cosine=cosine, joint_training=joint_training, data_index_dir=data_index_dir)

def get_embeddings(load_dir, filename, dictionaries):
    return word2vec_turk(load_dir, load_dir=load_dir, filename=filename, retraining=False, X=None, y=None, dictionaries=dictionaries, get_embeddings=True)



def train_by_name(X, y, dictionaries, filename, training_set, model, load, load_dir=None):


    if model == 'wiki_cosine' or model == 'wiki_output':
        directory = model
        #trained the same way, just evaluted differently
        print("TRAINING WIKI")
        return train_original_model(filename, save_dir=directory, load_dir=directory, load=load)

    if model == 'wiki+turk_cosine':
        directory = training_set+'_'+model
        if load:
            print(directory)
            return retrain_model_and_get_embeddings(X, y, dictionaries, filename, cosine=True, save_dir=directory, load_dir=directory, bigram_split=False, joint_training=True, data_index_dir=training_set+"_data_index")
        else:
            temp_load_dir = "wiki_output"
            return retrain_model_and_get_embeddings(X, y, dictionaries, filename, cosine=True, save_dir=directory, load_dir=temp_load_dir, bigram_split=False, joint_training=True)

def print_similarities(object_labels, cosine_matrix):

	# PRINTS A LIST, IN DESCENDING ORDER, OF PAIRWISE COSINE SIMILARITIES OF ALL OBJECT PAIRS
    size = len(cosine_matrix)
    cosine_similarities = {}
    for i in range(size):
    	for j in range(i):
    		cosine_similarities[object_labels[i] + " and " + object_labels[j]] = cosine_matrix[i][j]

    cosine_similarities = sorted(cosine_similarities.items(), key=lambda x: x[1], reverse=True)
    for i in cosine_similarities:
    	print("{}: {}".format(i[0], i[1]))

    # GENERATES A TABLE OF OBJECT PAIRINGS AND THEIR RESPECTIVE COSINE SIMILARITIES--FOR EACH OBJECT
    # THE COSINE SIMILARITIES ARE IN DESCENDING ORDER
    cosine_similarities_table = pd.DataFrame(columns=object_labels)
    for i in range(size):
    	cosine_similarities = {}
    	curr_food_item = object_labels[i]
    	for j in range(size):
    		if (curr_food_item != object_labels[j]):
    			cosine_similarities[object_labels[j]] = cosine_matrix[i][j]

    	cosine_similarities = sorted(cosine_similarities.items(), key=lambda x: x[1], reverse=True)
    	for j in range(len(cosine_similarities)):
    		cosine_similarities_table.loc[j, curr_food_item] = "{}: {}".format(cosine_similarities[j][0], cosine_similarities[j][1])

    # print(cosine_similarities_table)
    cosine_similarities_table.to_csv(path_or_buf="../cosine_similarities_table.csv")


def main(argv):

    if len(argv) > 1 and argv[1] == "train_wiki":
        train_wiki=True
        print("Training Wiki model first")
    else:
        train_wiki=False
        print("Wiki is already trained: retraining on shelving dataset.")

    filename = os.path.join(os.path.abspath(os.getcwd()),'recipe_corpus.txt')

    train_x = []
    train_y = []
    validate_x = []
    validate_y = []
    optimals = []

    num_ind = 0
    for num in list([3,4]):
        train1 = read_csv_train_test("data_human/"+str(num+1)+"_train.csv")
        validate_x.append([])
        validate_y.append([])
        train_x.append([])
        train_y.append([])
        for i in range(int(len(train1[0])/3)):
            index = random.choice(range(len(train1[0])))
            """
            validate_x[num].append(train1[0][index])
            validate_y[num].append(train1[1][index])
            """
            validate_x[num_ind].append(train1[0][index])
            validate_y[num_ind].append(train1[1][index])
            del train1[0][index]
            del train1[1][index]
        train_x[num_ind].append(train1[0])
        train_y[num_ind].append(train1[1])
        num_ind += 1
    cosine_heatmap.object_labels = cosine_heatmap.get_object_labels(train_x[0][0])

    print("==========================================================")


    models = ['wiki_output', 'wiki+turk_cosine']

    """
    Training Procedure:
    ------------------
    -Train a w2v model from recipes corpus.
    -Grab the embeddings, figure out how well they do
    -Train cosine off of that & keep going.
    -alternate between w2v and cosine.
    """

    dictionaries_dir = os.path.join(os.getcwd(), "..", "dictionaries")

    if train_wiki:
        bigram_dictionaries = get_pretrain_dictionaries(filename)
        bigram_unused_dictionary = bigram_dictionaries[2]
        embeddings, weights = train_by_name(None, None, bigram_dictionaries, filename, "1", "wiki_output", load=False)
        with open(dictionaries_dir, 'wb') as dict_file:
            pickle.dump(bigram_dictionaries, dict_file)
        sys.exit(0)
    else:
        with open(dictionaries_dir, 'rb') as dict_file:
            bigram_dictionaries = pickle.load(dict_file)
            bigram_unused_dictionary = bigram_dictionaries[2]
        embeddings, weights = get_embeddings("wiki_output", filename, bigram_dictionaries) 

        ### GENERATING HEATMAP OF COSINE SIMILARITIES BETWEEN 50 FOOD WORDS (AFTER TRAINING ON THE RECIPE CORPUS)

        # print(wn.synsets('food', pos=wn.NOUN))
        # solid_food = wn.synset('food.n.02')
        # solid_food_hypos = solid_food.hyponyms()
        # for food in solid_food_hypos: 
        #     print(food.lemmas()[0].name())
        object_labels = ["chocolate", "coconut", "loaf", "bread", "pasta", "cake", "pastry", "crackers", "muffin", "cereal", "muesli", "liquor", "sake", "wine", "cider", "brew", "beer", "aperitif", "coffee", "cocoa", "water", "juice", "lime_juice", "grape_juice", "orange_juice", "cola", "coke", "gingerale", "apple", "apricot", "avocado", "banana", "kiwi", "mango", "orange", "tomato", "artichoke", "corn", "cucumber", "onion", "squash", "chips", "potato_chips", "jelly", "peanut_butter", "ketchup", "chilisauce", "mustard", "salsa", "soy_sauce"]
        cosine_matrix = cosine_heatmap.pairwise_sim_grid(embeddings, bigram_unused_dictionary, object_labels)

        # PRINTS MORE STATISTICS OF COSINE SIMILARITIES OF ALL OBJECT PAIRS
        print_similarities(object_labels, cosine_matrix)

        cosine_heatmap.gen_cosine_heatmap(object_labels, [cosine_matrix], "heatmap")

        ###

    current_model = models[1]


    num_ind = 0
    for num in list([3,4]):
        print("TRAINING NEW MODEL:", current_model)
        orig_x, orig_y = train_x[num_ind][0], train_y[num_ind][0] 
        current_x, current_y = copy.deepcopy(orig_x), copy.deepcopy(orig_y)
        optimal_n_epochs = 0
        old_acc = 0.0
        new_acc = 0.0
        load=False
        n_equals = 0
        cosine_model_initialized=False
        iterations = 0
        while True:
            if new_acc <= old_acc:
                n_equals += 1
                #if n_equals > 4:
                if n_equals > 1:
                    break
            else:
                n_equals = 0
            old_acc = new_acc
            if cosine_model_initialized:
                embeddings, weights = get_embeddings(str(num+1)+"_"+current_model, filename, bigram_dictionaries)
            else:
                embeddings, weights = get_embeddings("wiki_output", filename, bigram_dictionaries) 

            
            ## Debugging: to verify that the training set trim-down is working
            len_old_filtered = 0
            if cosine_model_initialized:
                len_old_filtered = len(current_x) 
                current_x, current_y = evaluate_word2vec_cosine(orig_x, orig_y, embeddings, weights, bigram_unused_dictionary, "results.csv", bigram_split=False, discard_instances=True)
            
            filtered_diff = len_old_filtered - len(current_x) 
            print("DIFFERENCE OF FILTERED DATASET: Old:", str(len_old_filtered), "New:", str(len(current_x)))

            print("ABOUT TO TRAIN NEW SET")
            embeddings, weights = train_by_name(current_x, current_y, bigram_dictionaries, filename, str(num+1), current_model, load)
            cosine_model_initialized = True

            new_acc = evaluate_word2vec_cosine(validate_x[num_ind], validate_y[num_ind], embeddings, weights, bigram_unused_dictionary, "results.csv", bigram_split=False)


            optimal_n_epochs += 1000
            load=True
            print("BATCH FINISHED")
            print("OLD ACC:", old_acc)
            print("NEW ACC:", new_acc)

            iterations += 10
            cosine_heatmap.gen_cosine_heatmap(cosine_heatmap.object_labels, cosine_heatmap.grid, "set_{}_iter{}".format(num+1, iterations))
            cosine_heatmap.grid = []


        optimal_n_epochs -= 2000

        print("OPTIMAL NUMBER OF EPOCHS: ", optimal_n_epochs)
        print("Accuracy: ", new_acc)
        #print('heatmap len', len(cosine_heatmap.grid))
        print('*******************************************************************')
        print()

        optimals.append(optimal_n_epochs)
        num_ind += 1



if __name__=="__main__":
    main(sys.argv)
