from word2vec_basic import *
from word2vec_eval import *
from data_gen import *
import pickle
import sys
import os
from sklearn.model_selection import KFold
import cosine_heatmap

#NOTE: This procedure trains using validation sets

def train_original_model(filename, save_dir, load_dir=None, load=False):
    return word2vec_turk(save_dir, load_dir=load_dir, filename=filename, retraining=False, X=None, y=None, dictionaries=None, bigram_split=False, load=load)

def retrain_model_and_get_embeddings(X, y, dictionaries, filename, cosine, load_dir, save_dir, bigram_split=True, joint_training=False, data_index_dir="data_index",lr=1.0):
    return word2vec_turk(save_dir, load_dir, filename=filename, retraining=True, X=X, y=y, dictionaries=dictionaries, bigram_split=bigram_split, load=True, cosine=cosine, joint_training=joint_training, data_index_dir=data_index_dir, lr=lr)

def get_embeddings(load_dir, filename, dictionaries):
    return word2vec_turk(load_dir, load_dir=load_dir, filename=filename, retraining=False, X=None, y=None, dictionaries=dictionaries, get_embeddings=True)



def train_by_name(X, y, dictionaries, filename, training_set, model, load, load_dir=None, lr=1.0):


    if model == 'wiki_cosine' or model == 'wiki_output':
        directory = model
        #trained the same way, just evaluted differently
        print("TRAINING WIKI")
        return train_original_model(filename, save_dir=directory, load_dir=directory, load=load)

    if model == 'wiki+turk_cosine':
        directory = training_set+'_'+model
        if load:
            print(directory)
            return retrain_model_and_get_embeddings(X, y, dictionaries, filename, cosine=True, save_dir=directory, load_dir=directory, bigram_split=False, joint_training=True, data_index_dir=training_set+"_data_index", lr=lr)
        else:
            temp_load_dir = "wiki_output"
            return retrain_model_and_get_embeddings(X, y, dictionaries, filename, cosine=True, save_dir=directory, load_dir=temp_load_dir, bigram_split=False, joint_training=True, lr=lr)



def main(argv):

    if len(argv) > 1 and argv[1] == "train_wiki":
        train_wiki=True
        print("Training Wiki model first")
    else:
        train_wiki=False
        print("Wiki is already trained: retraining on shelving dataset.")

    filename = os.path.join(os.path.abspath(os.getcwd()),'fil9_bigram')
    #filename = os.path.join(os.path.abspath(os.getcwd()),'instacart_train_2.csv')
    #filename = wiki_filename
    turk_filename = "final_cleaned_results.csv"


    train_x = []
    train_y = []
    validate_x = []
    validate_y = []
    optimals = []

    num_ind = 0
    for num in list([0,3,4]):
        train1 = read_csv_train_test("data/"+str(num+1)+"_train.csv")
        #validate_x.append([])
        #validate_y.append([])
        train_x.append([])
        train_y.append([])
        #for i in range(int(len(train1[0])/3)):
            #index = random.choice(range(len(train1[0])))
        """
            validate_x[num].append(train1[0][index])
            validate_y[num].append(train1[1][index])
        """
            #validate_x[num_ind].append(train1[0][index])
            #validate_y[num_ind].append(train1[1][index])
            #del train1[0][index]
            #del train1[1][index]
        train_x[num_ind].append(train1[0])
        train_y[num_ind].append(train1[1])
        num_ind += 1


    print("==========================================================")
    

    # CREATE K-FOLD CROSS VALIDATION SETS
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    print(len(train_x[0][0]))
    for i in range(len(train_x)):
        K = 10
        X_train.append([])
        X_train[i] = ([[] for i in range(K)])
        X_test.append([])
        X_test[i] = ([[] for i in range(K)])
        y_train.append([])
        y_train[i] = ([[] for i in range(K)])
        y_test.append([])
        y_test[i] = ([[] for i in range(K)])
        for k, (train_index, test_index) in enumerate(KFold(K).split(train_x[i][0])):
            #print('adding another fold')
            print(train_index.shape)
            X_train[i][k], X_test[i][k] = np.array(train_x[i][0])[train_index], np.array(train_x[i][0])[test_index]
            y_train[i][k], y_test[i][k] = np.array(train_y[i][0])[train_index], np.array(train_y[i][0])[test_index]


    print(len(X_train))
    print(len(X_train[0]))
    print(len(X_train[0][0]))
    models = ['wiki_output', 'wiki+turk_cosine']



    cosine_heatmap.object_labels = cosine_heatmap.get_object_labels(train_x[0][0])

    """
    Training Procedure:
    ------------------
    -Train a w2v model from wikipedia.
    -Grab the embeddings, figure out how well they do
    -Train cosine off of that & keep going.
    -alternate between w2v and cosine.
    """

    dictionaries_dir = os.path.join(os.getcwd(), "..", "dictionaries")

    if train_wiki:
        bigram_dictionaries = get_pretrain_dictionaries(filename)
        bigram_unused_dictionary = bigram_dictionaries[2]
        embeddings, weights = train_by_name(None, None, bigram_dictionaries, filename, "1", "wiki_output", load=False)
        #embeddings, weights = train_by_name(None, None, bigram_dictionaries, filename, "1", "wiki_output", load=False)
        #embeddings, weights = train_by_name(None, None, bigram_dictionaries, filename, "1", "wiki_output", load=True)
        with open(dictionaries_dir, 'wb') as dict_file:
            pickle.dump(bigram_dictionaries, dict_file)
        sys.exit(0)
    else:
        with open(dictionaries_dir, 'rb') as dict_file:
            bigram_dictionaries = pickle.load(dict_file)
            bigram_unused_dictionary = bigram_dictionaries[2]
        embeddings, weights = get_embeddings("wiki_output", filename, bigram_dictionaries) 
        
    current_model = models[1]


    num_ind = 0
    for num in list([0,3,4]):
        print("TRAINING NEW MODEL:", current_model)
        optimal_n_epochs = 0
        load=False
        cosine_model_initialized=False
        all_orig_x, all_orig_y = train_x[num_ind][0], train_y[num_ind][0]
        loss_iter = 0
        for i in range(len(X_train[0])):
            orig_x, orig_y = X_train[num_ind][i].tolist().copy(), y_train[num_ind][i].tolist().copy()
            current_x, current_y = X_train[num_ind][i].tolist().copy(), y_train[num_ind][i].tolist().copy()

            validate_x = X_test[num_ind][i].tolist()
            validate_y = y_test[num_ind][i].tolist()
            print("On fold {}".format(i))
            old_acc = 0.0
            new_acc = 0.0
            n_equals = 0

            while True:
                if new_acc <= old_acc:
                    n_equals += 1
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
                train_acc = evaluate_word2vec_cosine(current_x, current_y, embeddings, weights, bigram_unused_dictionary, "results.csv", bigram_split=False)
                #current_x, current_y = evaluate_word2vec_cosine(current_x, current_y, embeddings, weights, bigram_unused_dictionary, "results.csv", bigram_split=False, discard_instances=True)
                #if cosine_model_initialized:
                    #current_x, current_y = evaluate_word2vec_cosine(orig_x, orig_y, embeddings, weights, bigram_unused_dictionary, "results.csv", bigram_split=False, discard_instances=True)
                if len(current_x) == 0 or len(current_x) == 1:
                    break
                
                filtered_diff = len_old_filtered - len(current_x) 
                print("DIFFERENCE OF FILTERED DATASET: Old:", str(len_old_filtered), "New:", str(len(current_x)))

                #print("ABOUT TO TRAIN NEW SET")
                decayed_learning_rate = 1 * .9 ** (loss_iter / 100)
                #decayed_learning_rate = 1
                print('learning rate', decayed_learning_rate)
                embeddings, weights = train_by_name(current_x, current_y, bigram_dictionaries, filename, str(num+1), current_model, load,lr=decayed_learning_rate)
                loss_iter += 10
                cosine_model_initialized = True

                new_acc = evaluate_word2vec_cosine(validate_x, validate_y, embeddings, weights, bigram_unused_dictionary, "results.csv", bigram_split=False)
                orig_acc = evaluate_word2vec_cosine(all_orig_x, all_orig_y, embeddings, weights, bigram_unused_dictionary, "results.csv", bigram_split=False)



                optimal_n_epochs += 1000
                load=True
                #print("BATCH FINISHED")
                #print("OLD ACC:", old_acc)
                print("NEW ACC:", new_acc)
                print("TRAIN FILTER ACC:", train_acc)
                print("TRAIN ORIG ACC:", orig_acc)
                cosine_heatmap.gen_cosine_heatmap(cosine_heatmap.object_labels, cosine_heatmap.grid, "set_{}_fold_{}_iter_{}".format(num,i, loss_iter))
                cosine_heatmap.grid = []



        optimal_n_epochs -= 2000

        print("OPTIMAL NUMBER OF EPOCHS: ", optimal_n_epochs)
        print("Accuracy: ", new_acc)
        print('*********************************************************\n\n')
        optimals.append(optimal_n_epochs)
        num_ind += 1



if __name__=="__main__":
    main(sys.argv)
