from word2vec_basic import *
from object_placement_turk import *

def train_original_model(filename, save_dir, load_dir, load=False):
    word2vec_turk(save_dir, load_dir=save_dir, filename=filename, retraining=False, X=None, y=None, dictionaries=None, bigram_split=False, load=load)
    return

def train_original_bigram_split(filename, save_dir, load=False):
    word2vec_turk(save_dir, load_dir=save_dir, filename=filename, retraining=False, X=None, y=None, dictionaries=None, bigram_split=True, load=load)
    return
    

def retrain_model_and_get_embeddings(X, y, dictionaries, filename, cosine, load_dir, save_dir, bigram_split=False):
    return word2vec_turk(save_dir, load_dir, filename=filename, retraining=True, X=X, y=y, dictionaries=dictionaries, bigram_split=bigram_split, load=True, cosine=cosine) 
    

def train_on_turk_exclusively(X, y, dictionaries, filename, cosine,save_dir):
    return word2vec_turk(save_dir, load_dir=save_dir, filename=filename, retraining=True, X=X, y=y, dictionaries=dictionaries,bigram_split=False, load=False, cosine=cosine)


def train_joint_loss(X, y, dictionaries, filename, bigram_split, save_dir):
    return word2vec_turk(save_dir, load_dir=save_dir, filename=filename, X=X, y=y, dictionaries=dictionaries, bigram_split=bigram_split, load=False, cosine=False, joint_training=True)

def get_embeddings(load_dir, filename, dictionaries):
    return word2vec_turk(load_dir, load_dir=load_dir, filename=filename, retraining=False, X=None, y=None, dictionaries=dictionaries, get_embeddings=True)


if __name__=="__main__":
  #filename = '/home/justin/Data/modified_text'
  #filename = 'text8'
  filename = 'modified_text'
  turk_filename = "final_cleaned_results.csv"
  
  train1 = read_csv_train_test("data/1_train.csv")
  train2 = read_csv_train_test("data/2_train.csv")
  train3 = read_csv_train_test("data/3_train.csv")
  train4 = read_csv_train_test("data/4_train.csv")
  train5 = read_csv_train_test("data/5_train.csv")

  train_sets = [train1, train2, train3, train4, train5]


  bigram_dictionaries = get_pretrain_dictionaries(filename)
  bigram_unused_dictionary = bigram_dictionaries[2]

  wiki_dir = "wiki"
  bigram_dir = "bigram_wiki"
  print("TRAINING ON WIKIPEDIA")
  train_original_model(filename, load_dir=None, save_dir=wiki_dir)
  print("TRAINING BIGRAM-SPLIT ON WIKIPEDIA")
  train_original_bigram_split(filename, save_dir=bigram_dir)
  


  for set_num in range(len(train_sets)):
      print("TRAIN SET #" + str(set_num+1) + ":::")


      train_x, train_y = train_sets[set_num]

      #MODEL #2: Retrain on cosine. 
      print("TRAINING WIKIPEDIA->TURK")
      retrain_model_and_get_embeddings(train_x, train_y, bigram_dictionaries, cosine=True, load_dir=wiki_dir, save_dir= str(set_num+1)+"_wiki+turk_cosine") 
      retrain_model_and_get_embeddings(train_x, train_y, bigram_dictionaries, cosine=False, load_dir=wiki_dir, save_dir=str(set_num+1)+"_wiki+turk_output") 

      print("TRAINING TURK EXCLUSIVE")
      #MODEL #3: Just train on turk.
      train_on_turk_exclusively(train_x, train_y, bigram_dictionaries, turk_filename, cosine=True, save_dir=str(set_num+1)+"_turk_cosine") 
      train_on_turk_exclusively(train_x, train_y, bigram_dictionaries, turk_filename, cosine=False, save_dir=str(set_num+1)+"_turk_output") 

      print("TRAINING TURK->WIKIPEDIA")
      #MODEL #4: Trained on turk, now train on wiki.
      train_original_model(filename, save_dir=str(set_num+1)+"_turk+wiki_cosine", load_dir=str(set_num+1)+"_turk_cosine")
      train_original_model(filename, save_dir=str(set_num+1)+"_turk+wiki_output", load_dir=str(set_num+1)+"_turk_output")

      print("TRAINING USING BIGRAM SPLIT")
      #MODEL #5: Bigram
      retrain_model_and_get_embeddings(train_x, train_y, bigram_dictionaries, filename, bigram_split=True, cosine=True, load_dir=bigram_dir, save_dir=str(set_num+1)+"_bigram_cosine")
      retrain_model_and_get_embeddings(train_x, train_y, bigram_dictionaries, filename, bigram_split=True, cosine=False, load_dir=bigram_dir, save_dir=str(set_num+1)+"_bigram_output")
      
      print("TRAINING JOINT LOSS")
      #MODEL #7:  Joint loss
      train_joint_loss(train_x, train_y, bigram_dictionaries, filename, bigram_split=False, save_dir=str(set_num+1)+"_joint") 
      
      print("TRAINING JOINT LOSS (BIGRAM_SPLIT)")
      #MODEL #8: Joint loss (bigram split)
      train_joint_loss(train_x, train_y, bigram_dictionaries, filename, bigram_split=True, save_dir=str(set_num+1)+"_joint_bigram") 





