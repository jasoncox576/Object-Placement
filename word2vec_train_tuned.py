from word2vec_basic import *
from word2vec_eval import *
from object_placement_turk import *
import os

def train_original_model(filename, save_dir, load_dir=None, load=False):
    return word2vec_turk(save_dir, load_dir=load_dir, filename=filename, retraining=False, X=None, y=None, dictionaries=None, bigram_split=False, load=load)


def train_original_bigram_split(filename, save_dir, load_dir=None, load=False):
    return word2vec_turk(save_dir, load_dir=load_dir, filename=filename, retraining=False, X=None, y=None, dictionaries=None, bigram_split=True, load=load)


def retrain_model_and_get_embeddings(X, y, dictionaries, filename, cosine, load_dir, save_dir, bigram_split=True):
    return word2vec_turk(save_dir, load_dir, filename=filename, retraining=True, X=X, y=y, dictionaries=dictionaries, bigram_split=bigram_split, load=True, cosine=cosine)


def train_on_turk_exclusively(X, y, dictionaries, filename, cosine,save_dir, load_dir=None, load=False,bigram_split=False):
    return word2vec_turk(save_dir, load_dir=load_dir, filename=filename, retraining=True, X=X, y=y, dictionaries=dictionaries,bigram_split=bigram_split, load=load, cosine=cosine)


def train_joint_loss(X, y, dictionaries, filename, bigram_split, save_dir, load_dir=None, load=False, a=0.5, b=0.5):
    return word2vec_turk(save_dir, load_dir=load_dir, filename=filename, X=X, y=y, dictionaries=dictionaries, bigram_split=bigram_split, load=load, cosine=False, joint_training=True, load_early=False, a=a, b=b)

def get_embeddings(load_dir, filename, dictionaries):
    return word2vec_turk(load_dir, load_dir=load_dir, filename=filename, retraining=False, X=None, y=None, dictionaries=dictionaries, get_embeddings=True)



def train_by_name(X, y, dictionaries, filename, training_set, model, load, load_dir=None):


    if model == 'wiki_cosine' or model == 'wiki_output':
        directory = model
        #trained the same way, just evaluted differently
        print("========================================================================")
        print("TRAINING WIKI")
        return train_original_model(filename, save_dir=directory, load_dir=directory, load=load)



    if model == 'wiki+turk_cosine':
        directory = training_set+'_'+model
        if load:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print(directory)
            return retrain_model_and_get_embeddings(X, y, dictionaries, filename, cosine=True, save_dir=directory, load_dir=directory, bigram_split=False)
        else:
            #temp_load_dir = training_set+'_wiki_cosine'
            print("))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))")
            temp_load_dir = "wiki_output"
            return retrain_model_and_get_embeddings(X, y, dictionaries, filename, cosine=True, save_dir=directory, load_dir=temp_load_dir, bigram_split=False)



if __name__=="__main__":
    filename = os.path.join(os.path.abspath(os.getcwd()),'fil9_bigram')
    turk_filename = "final_cleaned_results.csv"

    train_wiki=False


    train_x = []
    train_y = []
    validate_x = []
    validate_y = []
    optimals = []
    for num in range(3):
        train1 = read_csv_train_test("data/"+str(num+1)+"_train.csv")
        validate_x.append([])
        validate_y.append([])
        train_x.append([])
        train_y.append([])
        for i in range(int(len(train1[0])/3)):
            index = random.choice(range(len(train1[0])))
            validate_x[num].append(train1[0][index])
            validate_y[num].append(train1[1][index])
            del train1[0][index]
            del train1[1][index]
        train_x[num].append(train1[0])
        train_y[num].append(train1[1])

    print("==========================================================")

    bigram_dictionaries = get_pretrain_dictionaries(filename)
    bigram_unused_dictionary = bigram_dictionaries[2]

    models = ['wiki_output', 'wiki+turk_cosine']

    """
    Training Procedure:
    ------------------
    -Train a w2v model from wikipedia.
    -Grab the embeddings, figure out how well they do
    -Train cosine off of that & keep going.
    -alternate between w2v and cosine.
    """

    if train_wiki:
        embeddings, weights = train_by_name(None, None, bigram_dictionaries, filename, "1", "wiki_output", load=False)
    else:
        embeddings, weights = get_embeddings("wiki_output", filename, bigram_dictionaries) 

    current_model = models[1]


    for num in range(3):
        print("TRAINING NEW MODEL:", current_model)
        current_x, current_y = train_x[num][0], train_y[num][0] 
        optimal_n_epochs = 0
        old_acc = 0.0
        new_acc = 0.0
        load=False
        n_equals = 0
        weights_generated=False
        while True:
            if new_acc <= old_acc:
                n_equals += 1
                if n_equals > 1:
                    break
            else:
                n_equals = 0
            old_acc = new_acc
            if weights_generated:
                embeddings = get_embeddings(str(num+1)+"_"+current_model, filename, bigram_dictionaries)[0]
            
            ## Debugging: to verify that the training set trim-down is working
            len_old_filtered = 0
            if weights_generated:
                len_old_filtered = len(current_x) 

            current_x, current_y = evaluate_word2vec_cosine(current_x, current_y, embeddings, weights, bigram_unused_dictionary, "results.csv", bigram_split=False, discard_instances=True)
            
            filtered_diff = len_old_filtered - len(current_x) 
            print("DIFFERENCE OF FILTERED DATASET: Old:", str(len_old_filtered), "New:", str(len(current_x)))

            print("ABOUT TO TRAIN NEW SET")
            embeddings, weights = train_by_name(current_x, current_y, bigram_dictionaries, filename, str(num+1), current_model, load)
            weights_generated = True

            new_acc = evaluate_word2vec_cosine(validate_x[num], validate_y[num], embeddings, weights, bigram_unused_dictionary, "results.csv", bigram_split=False)


            optimal_n_epochs += 1000
            load=True
            print("BATCH FINISHED")
            print("OLD ACC:", old_acc)
            print("NEW ACC:", new_acc)


        optimal_n_epochs -= 2000

        print("OPTIMAL NUMBER OF EPOCHS: ", optimal_n_epochs)
        print("Accuracy: ", new_acc)
        optimals.append(optimal_n_epochs)
