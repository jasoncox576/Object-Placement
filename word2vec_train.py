from word2vec_basic import *

def train_original_model(filename):
    word2vec_turk('log', filename, retraining=False, X=None, y=None, dictionaries=None)
    return

if __name__=="__main__":
  filename = '/home/justin/Data/modified_text'
  train_original_model(filename=filename) 

