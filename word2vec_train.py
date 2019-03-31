from word2vec_basic import *

def train_original_model(filename):
    word2vec_turk('log', filename, retraining=False, X=None, y=None, dictionaries=None)
    return

def retrain_model_and_get_embeddings(X, y, dictionaries, filename):
    return word2vec_turk('log', filename, retraining=True, X=X, y=y, dictionaries=dictionaries) 
    


if __name__=="__main__":
  #filename = '/home/justin/Data/modified_text'
  #filename = 'text8'
  filename = 'modified_text'
  train_original_model(filename=filename) 

