from word2vec_basic import *

def train_original_model(filename, load=False):
    word2vec_turk('log', filename, retraining=False, X=None, y=None, dictionaries=None, bigram_split=False, load=load)
    return

def train_original_bigram_split(filename, load=False):
    word2vec_turk('log', filename, retraining=False, X=None, y=None, dictionaries=None, bigram_split=True, load=load)
    return
    

def retrain_model_and_get_embeddings(X, y, dictionaries, filename, cosine, bigram_split=False):
    return word2vec_turk('log', filename, retraining=True, X=X, y=y, dictionaries=dictionaries, bigram_split=bigram_split, load=True, cosine=cosine) 
    

def train_on_turk_exclusively(X, y, dictionaries, filename, cosine):
    return word2vec_turk('log', filename=filename, retraining=True, X=X, y=y, dictionaries=dictionaries,bigram_split=False, load=False, cosine=cosine)


if __name__=="__main__":
  #filename = '/home/justin/Data/modified_text'
  #filename = 'text8'
  filename = 'modified_text'
  train_original_model(filename=filename) 

