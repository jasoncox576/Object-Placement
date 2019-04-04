from sklearn.model_selection import StratifiedKFold
import matrix_priors
from make_train_test import *
import numpy as np
import csv
import copy
import sys


def instances_disagree_process(X, y):
    
    """
    Need to find cases for which the data is 4/4 and combine those
    into one instance
    """

    print("Discarding bad instances....")
    used_indices = {}
    new_X = []
    new_y = []

    for x1 in range(len(X)):
        if used_indices.get(x1): continue
        answer_count = [0,0,0]
        answer_count[X[x1][1:].index(y[x1])] += 1
        current_indices = [x1]

        for x2 in range(len(X)):
            if (used_indices.get(x2)) or (x1 == x2): continue
            if(X[x1] == X[x2]):
                answer_count[X[x2][1:].index(y[x2])] += 1
                used_indices[x2] = 1
                current_indices.append(x2)
        used_indices[x1] = 1
        if max(answer_count) == 4:
            new_X.append(X[x1])
            new_y.append(X[x1][np.argmax(answer_count)+1])
        """
        else:
            for index in current_indices:
                new_X.append(X[index])
                new_y.append(y[index])
        """
        
    print("Finished discarding bad instances.")
    return new_X, new_y


def remove_perfect(X, y):
    
    """
    Need to find cases for which the data is NOT 4/4
    and add those to our list
    """

    used_indices = {}
    new_X = []
    new_y = []

    for x1 in range(len(X)):
        if used_indices.get(x1): continue
        answer_count = [0,0,0]
        answer_count[X[x1][1:].index(y[x1])] += 1
        current_indices = [x1]

        for x2 in range(len(X)):
            if (used_indices.get(x2)) or (x1 == x2): continue
            if(X[x1] == X[x2]):
                answer_count[X[x2][1:].index(y[x2])] += 1
                used_indices[x2] = 1
                current_indices.append(x2)
        used_indices[x1] = 1
        
        # If NOT Perfect
        if max(answer_count) != 4 and np.sum(answer_count) == 4:
            print("Appending #2 ", X[x1])
            new_X.append(X[x1])
            new_y.append(y[x1])

    return new_X, new_y

def remove_similars(X, y):

    #taking out any instances with these objects, as they are
    #similar to 'grape_juice', 'bread', and 'orange' respectively.
    similar_objects = ["orange_juice", "crackers", "apple"]

    train_x = []
    train_y = []

    test_x = []
    test_y = []

    discarded_indices = {}
    
    for i in range(len(X)):
        for obj in X[i]:
            if obj in similar_objects:
                test_x.append(X[i])
                test_y.append(y[i])
                print("Test x, y append ", X[i], y[i])
                discarded_indices[i] = 1
    
    for i in range(len(X)):
        if not discarded_indices.get(i):
            train_x.append(X[i])
            train_y.append(y[i])            


    train = train_x, train_y
    test = test_x, test_y

    return train, test


def add_instances(test1, test2):
    
    x1, y1 = test1
    x2, y2 = test2

    for i in range(len(x1)):
        for j in range(4):
            x2.append(x1[i])
            y2.append(y1[i])



    return x2, y2



def test_partition(X, y, assure_p):
    
    """
    Split the data into 75% train, 25% test.
    Makes sure that each object is 'p' at least once in
    the train set
    """

    base_indices = range(len(X))
    
    
    object_list = []
    #NOTE: The below number is hardcoded. This is however many objects are in your total dataset.
    # we just need to make sure that every single object is 'p' at least once in the training set
    num_test_indices = (int)(len(base_indices)/4)
    while len(object_list) != 12:
        test_indices = np.random.choice(base_indices, num_test_indices, replace=False)
        train_indices = [x for x in base_indices if x not in test_indices]
        for index in train_indices:
            if X[index][0] not in object_list:
                object_list.append(X[index][0])
        if not assure_p:
            break
    
    train_x = [X[i] for i in train_indices]
    train_y = [y[i] for i in train_indices]
    
    test_x = [X[i] for i in test_indices]
    test_y = [y[i] for i in test_indices] 
       

    for i in X:
        print(i)
    
    
    return (train_x, train_y), (test_x, test_y) 



def read_csv_train_test(filename):
    
    X = []
    y = []

    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            X.append(row[:3])
            y.append(row[-1])
    
    return X, y





def get_train_test(filename):

  #Yes, this is a horrible way to do this. I'm over it.
  X = []
  y = []
  word_substitution_set = {
    'orange' : 'orange',
    'apple' : 'apple',
    'corn' : 'corn',
    'cereal_2' : 'cereal',
    'jelly' : 'jelly',
    'orange_juice' : 'orange_juice',
    'grape_juice' : 'grape_juice',
    'onion' : 'onion',
    'crackers' : 'crackers',
    'bread' : 'bread',
    'potato_chips' : 'potato_chips',
    'coke' : 'coke'
  }

  answer_word_set = set({'orange', 'apple', 'corn', 'cereal', 'jelly', 'orange_juice', 'grape_juice', 'onion', 'crackers', 'bread', 'potato_chips', 'coke'})
  

  """
  Cross validation of the data

  Uses standard k-fold algorithm.
  """

  with open(filename) as csvfile:
      #reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
      reader = csv.reader(csvfile)
      for row in reader:
          if reader.line_num == 1:
              continue

          row_result = row[27:32]
          answer_label = row_result[4]
          answer = (["Top", "Middle", "Bottom"].index(answer_label)) + 1

          X.append([word_substitution_set[row_result[0]], word_substitution_set[row_result[1]], word_substitution_set[row_result[2]], word_substitution_set[row_result[3]]])
          #Fetch the answer to the most recent instance
          answer_word = X[-1][answer]
          y.append(answer_word)


          if not answer_word in answer_word_set:
            sys.exit()
          
  #This is train/test set #1
  X1 = copy.deepcopy(X)
  y1 = copy.deepcopy(y) 
  X_perfect, y_perfect = instances_disagree_process(X1, y1) 
  train1, test1 = test_partition(X_perfect, y_perfect, assure_p=True) 
  make_train_test_csv(copy.deepcopy(train1), copy.deepcopy(test1), "1") 
  print("MADE SET #1")

  #This is train/test set #2
  X1 = copy.deepcopy(X)
  y1 = copy.deepcopy(y) 
  train2 = [None, None]
  test2 = [None, None]
  train2[0] = copy.deepcopy(train1[0])
  train2[1] = copy.deepcopy(train1[1])

  test2[0] = copy.deepcopy(test1[0])
  test2[1] = copy.deepcopy(test1[1])

  test2 = remove_perfect(X1, y1) 

  make_train_test_csv(copy.deepcopy(train2), copy.deepcopy(test2), "2")
  print("MADE SET #2")

  #This is train/test set #3
  train3 = [None, None]
  train3[0] = copy.deepcopy(train1[0])
  train3[1] = copy.deepcopy(train1[1])

  test3 = add_instances(copy.deepcopy(test1), copy.deepcopy(test2)) 
  make_train_test_csv(copy.deepcopy(train3), copy.deepcopy(test3), "3")
  print("MADE SET #3")

  #This is train/test set #4
  X1 = copy.deepcopy(X)
  y1 = copy.deepcopy(y) 
  train4, test4 = remove_similars(X1, y1)   
  make_train_test_csv(train4, test4, "4")
  print("MADE SET #4")


  #This is train/test set #5
  X1 = copy.deepcopy(X)
  y1 = copy.deepcopy(y) 
  train5, test5 = test_partition(X1,y1, assure_p=False) 
  make_train_test_csv(train5, test5, "5")
  print("MADE SET #5")




if __name__=='__main__':
    get_train_test("final_cleaned_results.csv") 











