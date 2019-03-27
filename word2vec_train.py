from word2vec_basic import *

def train_original_model(filename):
    word2vec_basic('log', filename, retraining=False, X=None, y=None, dictionaries=None)
    return

if __name__=="__main__":
  filename = '/home/justin/Data/fil9'
  train_original_model(filename=filename) 

