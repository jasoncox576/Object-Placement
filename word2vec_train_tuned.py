from word2vec_scratch import *
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

    directory = training_set+'_'+model

    if model == 'wiki_cosine' or model == 'wiki_output':
        #trained the same way, just evaluted differently
        return train_original_model(filename, save_dir=directory, load_dir=directory, load=load)

    if model == 'bigram_wiki_cosine' or model == 'bigram_wiki_output':
        #trained the same way, just evaluted differently
        return train_original_bigram_split(filename, save_dir=directory, load_dir=directory, load=load)

    if model == 'turk_cosine':
        return train_on_turk_exclusively(X, y, dictionaries, filename, cosine=True, save_dir=directory, load=load, load_dir=directory, bigram_split=False)
    if model == 'turk_output':
        return train_on_turk_exclusively(X, y, dictionaries, filename, cosine=False, save_dir=directory, load=load, load_dir=directory, bigram_split=False)

    if model == 'turk_bigram_cosine':
        return train_on_turk_exclusively(X, y, dictionaries, filename, cosine=True, load=load, load_dir=directory, save_dir=directory, bigram_split=True)

    if model == 'turk_bigram_output':
        return train_on_turk_exclusively(X, y, dictionaries, filename, cosine=False, save_dir=directory, load=load, load_dir=load_dir, bigram_split=True)

    if model == 'wiki+turk_bigram_output':
        if load:
            return retrain_model_and_get_embeddings(X, y, dictionaries, filename, cosine=False, save_dir=directory, load_dir=directory, bigram_split=True)
        else:
            temp_load_dir = training_set+'_bigram_wiki_output'
            return retrain_model_and_get_embeddings(X, y, dictionaries, filename, cosine=False, save_dir=directory, load_dir=temp_load_dir, bigram_split=True)

    if model == 'wiki+turk_bigram_cosine':
        if load:
            return retrain_model_and_get_embeddings(X, y, dictionaries, filename, cosine=True, save_dir=directory, load_dir=directory, bigram_split=True)
        else:
            temp_load_dir = training_set+'_bigram_wiki_cosine'
            return retrain_model_and_get_embeddings(X, y, dictionaries, filename, cosine=True, save_dir=directory, load_dir=temp_load_dir, bigram_split=True)

    if model == 'wiki+turk_output':
        if load:
            return retrain_model_and_get_embeddings(X, y, dictionaries, filename, cosine=False, save_dir=directory, load_dir=directory, bigram_split=False)
        else:
            temp_load_dir = training_set+'_wiki_output'
            return retrain_model_and_get_embeddings(X, y, dictionaries, filename, cosine=False, save_dir=directory, load_dir=temp_load_dir, bigram_split=False)

    if model == 'wiki+turk_cosine':
        if load:
            return retrain_model_and_get_embeddings(X, y, dictionaries, filename, cosine=True, save_dir=directory, load_dir=directory, bigram_split=False)
        else:
            temp_load_dir = training_set+'_wiki_cosine'
            return retrain_model_and_get_embeddings(X, y, dictionaries, filename, cosine=True, save_dir=directory, load_dir=temp_load_dir, bigram_split=False)


    if model == 'turk+wiki_bigram_output':
        if load:
            return train_original_bigram_split(filename, save_dir=directory, load_dir=directory, load=True)
        else:
            return train_original_bigram_split(filename, save_dir=directory, load_dir=training_set+'_turk_bigram_output', load=True)

    if model == 'turk+wiki_bigram_cosine':
        if load:
            return train_original_bigram_split(filename, save_dir=directory, load_dir=directory, load=True)
        else:
            return train_original_bigram_split(filename, save_dir=directory, load_dir=training_set+'_turk_bigram_cosine', load=True)

    if model == 'turk+wiki_output':
        if load:
            return train_original_model(filename, save_dir=directory, load_dir=directory, load=True)
        else:
            return train_original_model(filename, save_dir=directory, load_dir=training_set+'_turk_output', load=True)

    if model == 'turk+wiki_cosine':
        if load:
            return train_original_model(filename, save_dir=directory, load_dir=directory, load=True)
        else:
            return train_original_model(filename, save_dir=directory, load_dir=training_set+'_turk_cosine', load=True)

    if 'joint' in model:
        float_index = model.index('.')
        a = float(model[float_index+1:float_index+2])/10
        b = 1-a
        if 'bigram' in model:
            return train_joint_loss(X, y, dictionaries, filename, bigram_split=True, save_dir=directory, load_dir=directory, load=load, a=a, b=b)
        else:
            return train_joint_loss(X, y, dictionaries, filename, bigram_split=False, save_dir=directory, load_dir=directory, load=load, a=a, b=b)


    if model == 'turk+wiki+turk_bigram_cosine':
        if load:
            return train_on_turk_exclusively(X, y, dictionaries, filename, cosine=True, save_dir=directory, bigram_split=True, load_dir = directory, load=True)
        else:
            temp_load_dir = training_set+'_'+'turk+wiki_bigram_cosine'
            return train_on_turk_exclusively(X, y, dictionaries, filename, cosine=True, save_dir=directory, bigram_split=True, load_dir = temp_load_dir, load=True)

    if model == 'turk+wiki+turk_bigram_output':
        if load:
            return train_on_turk_exclusively(X, y, dictionaries, filename, cosine=False, save_dir=directory, bigram_split=True, load_dir = directory, load=True)
        else:
            temp_load_dir = training_set+'_'+'turk+wiki_bigram_output'
            return train_on_turk_exclusively(X, y, dictionaries, filename, cosine=False, save_dir=directory, bigram_split=True, load_dir = temp_load_dir, load=True)
    if model == 'turk+wiki+turk_cosine':
        if load:
            return train_on_turk_exclusively(X, y, dictionaries, filename, cosine=True, save_dir=directory, bigram_split=False, load_dir = directory, load=True)
        else:
            temp_load_dir = training_set+'_'+'turk+wiki_cosine'
            return train_on_turk_exclusively(X, y, dictionaries, filename, cosine=True, save_dir=directory, bigram_split=False, load_dir = temp_load_dir, load=True)

    if model == 'turk+wiki+turk_output':
        if load:
            return train_on_turk_exclusively(X, y, dictionaries, filename, cosine=False, save_dir=directory, bigram_split=False, load_dir = directory, load=True)
        else:
            temp_load_dir = training_set+'_'+'turk+wiki_output'
            return train_on_turk_exclusively(X, y, dictionaries, filename, cosine=False, save_dir=directory, bigram_split=False, load_dir = temp_load_dir, load=True)

if __name__=="__main__":
    filename = os.path.join(os.path.abspath(os.getcwd()),'fil9_bigram')
    #filename = 'text8'
    #filename = 'modified_text'
    turk_filename = "final_cleaned_results.csv"



    train_x = []
    train_y = []
    validate_x = []
    validate_y = []
    optimals = []
    for num in range(3):
        train1 = read_csv_train_test("data/"+str(num+1)+"_train.csv")
        print(len(train1[0]))
        print(len(train1[1]))
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
    print(len(train_x))
    print(len(train_x[0]))
    print(len(train_x[0][0]))
    print(len(train_x[0][0][0]))
    print(train_x[0][0])

    print(len(validate_x[0][0][0]))
    print((validate_x[0][0][0]))

    bigram_dictionaries = get_pretrain_dictionaries(filename)
    bigram_unused_dictionary = bigram_dictionaries[2]

    #models = ['wiki_cosine', 'wiki_output', 'bigram_wiki_cosine', 'bigram_wiki_output', 'wiki+turk_cosine', 'wiki+turk_output', 'wiki+turk_bigram_cosine', 'wiki+turk_bigram_output', 'turk_cosine', 'turk_output', 'turk_bigram_cosine',  'turk_bigram_output', 'turk+wiki_cosine', 'turk+wiki_output', 'turk_wiki_bigram_cosine', 'turk+wiki_bigram_output', 'turk+wiki+turk_cosine', 'turk+wiki+turk_output', 'turk+wiki+turk_bigram_cosine', 'turk+wiki+turk_bigram_output']
    models = ['wiki_output', 'wiki+turk_cosine']
    """
    for i in range(1,10):
        models.append('joint_.'+str(i) + '_cosine')
        models.append('joint_.'+str(i) + '_output')
        models.append('joint_.'+str(i) + '_bigram_cosine')
        models.append('joint_.'+str(i) + '_bigram_output')
    """


    """
    Training Procedure:
    ------------------
    -Train a w2v model from wikipedia.
    -Grab the embeddings, figure out how well they do
    -Train cosine off of that & keep going.
    -alternate between w2v and cosine.
    """

    embeddings, weights = train_by_name(None, None, bigram_dictionaries, filename, str(num+1), "wiki_output", load=False)

    current_model = models[1]

    for num in range(3):
        print("TRAINING NEW MODEL:", current_model)
        optimal_n_epochs = 0
        old_acc = 0.0
        new_acc = 0.0
        load=True
        n_equals = 0
        while True:
            if new_acc <= old_acc:
                n_equals += 1
                if n_equals > 1:
                    break
            else:
                n_equals = 0
            old_acc = new_acc
            embeddings = get_embeddings(current_model, filename, bigram_dictionaries)

            filtered_x, filtered_y = evaluate_word2vec_cosine(train_x[num][0], train_y[num][0], embeddings, weights, bigram_unused_dictionary, "results.csv", bigram_split=False, discard_instances=True)
            embeddings, weights = train_by_name(filtered_x, filtered_y, bigram_dictionaries, filename, str(num+1), current_model, load)
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
