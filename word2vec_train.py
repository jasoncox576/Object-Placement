from word2vec_basic import *

def train_original_model(filename, load=False):
    word2vec_turk('log', filename, retraining=False, X=None, y=None, dictionaries=None, bigram_split=False, load=load)
    return

def retrain_model_and_get_embeddings(X, y, dictionaries, filename, bigram_split=False):
    return word2vec_turk('log', filename, retraining=True, X=X, y=y, dictionaries=dictionaries, bigram_split=bigram_split, load=True) 
    

def train_on_turk_exclusively(X, y, dictionaries, filename):
    return word2vec_turk('log', filename, retraining=true, X=X, y=y, dictionaries=dictionaries,bigram_split=False, load=False)


if __name__=="__main__":
  #filename = '/home/justin/Data/modified_text'
  #filename = 'text8'
  filename = 'modified_text'
  train_original_model(filename=filename) 

