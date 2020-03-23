#from sklearn.model_selection import StratifiedKFold
#import matrix_priors
from make_train_test import *
from sklearn.model_selection import StratifiedKFold
import numpy as np
import csv
import copy
import sys
import os


def get_agreements(X, y):
    used_indices = {}

    X = X.tolist()
    y = y.tolist()

    four_count = 0
    three_count = 0
    two_count = 0
    one_count = 0


    fours = []
    threes = []
    twos = []
    ones = []

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
            four_count += 1
            if len(fours) < 10:
                fours.append((X[x1], answer_count))
        if max(answer_count) == 3:
            three_count += 1
            if len(threes) < 10:
                threes.append((X[x1], answer_count))
        if max(answer_count) == 2:
            two_count += 1
            if len(twos) < 10:
                twos.append((X[x1], answer_count))
        if max(answer_count) == 1:
            one_count += 1
            if len(ones) < 10:
                ones.append((X[x1], answer_count))
    
    return four_count, three_count, two_count, one_count, fours, threes, twos, ones 




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
            #print("Appending #2 ", X[x1])
            for index in current_indices:
                new_X.append(X[index])
                new_y.append(y[index])

    return new_X, new_y

def remove_similars(X, y):

    #taking out any instances with these objects, as they are
    #similar to 'grape_juice', 'bread', and 'orange' respectively.
    similar_objects = ["orange"]

    train_x = []
    train_y = []

    test_x = []
    test_y = []

    discarded_indices = {}
    
    for i in range(len(X)):
        for obj in X[i]:
            if obj in similar_objects and i not in discarded_indices:
                test_x.append(X[i])
                test_y.append(y[i])
                #print("Test x, y append ", X[i], y[i])
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
       

    
    
    return (train_x, train_y), (test_x, test_y) 



def read_csv_train_test(filename):
    
    X = []
    y = []

    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            X.append(row[:-1])
            y.append(row[-1])
    
    return X, y





def get_train_test(filename):
   
  #NOTE::: ALL THIS FUNCTION DOES IS CREATE THE TRAIN/TEST CSV files.

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

  #answer_word_set = set({'orange', 'apple', 'corn', 'cereal', 'jelly', 'orange_juice', 'grape_juice', 'onion', 'crackers', 'bread', 'potato_chips', 'coke'})
  
  with open(filename) as csvfile:
      #reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
      reader = csv.reader(csvfile)
      for row in reader:

          #NOTE:::: Comment out the particular code depending on which dataset you're using. Turk=use second block, human=use first.
          """

          primary_items = row[0]
          secondary_items = row[1]

          # remove unnecessary quotes from the outside of comma-separted-value
          primary_items = primary_items.replace('"', '')
          secondary_items = secondary_items.replace('"', '')

          # split items delineated by comma into list of strings
          primary_items = primary_items.split(",")
          secondary_items = secondary_items.split(",")

          primary_object = primary_items[0]
          answer_label = primary_items[1]

          secondary_items.insert(0, primary_object)
          X.append(secondary_items) 
          print(X[-1])
          y.append(answer_label)
          print(y[-1])
          """
          row_result = row[27:32]
          answer_label = row_result[4]
          print(row_result)
          try:
            answer = (["Top", "Middle", "Bottom"].index(answer_label)) + 1
          except:
            continue
          X.append([word_substitution_set[row_result[0]], word_substitution_set[row_result[1]], word_substitution_set[row_result[2]], word_substitution_set[row_result[3]]])
          #Fetch the answer to the most recent instance
          answer_word = X[-1][answer]
          y.append(answer_word)

          """

          if not answer_word in answer_word_set:
            sys.exit()
          """
  #This is train/test set #1
  #25% perfect instances
  X1 = copy.deepcopy(X)
  y1 = copy.deepcopy(y) 
  print("LEN X1", len(X1))
  X_perfect, y_perfect = instances_disagree_process(X1, y1) 
  train1, test1 = test_partition(X_perfect, y_perfect, assure_p=True) 
  train1_copy = copy.deepcopy(train1[0]), copy.deepcopy(train1[1])
  test1_copy = copy.deepcopy(test1[0]), copy.deepcopy(test1[1])
  
  make_train_test_csv(copy.deepcopy(train1), copy.deepcopy(test1), "1") 
  print("MADE SET #1")
  """ 

  #This is train/test set #2
  #All 3/4
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
  #All 2/4
  train3 = [None, None]
  train3[0] = copy.deepcopy(train1[0])
  train3[1] = copy.deepcopy(train1[1])

  test3 = add_instances(copy.deepcopy(test1), copy.deepcopy(test2)) 

  train3_copy = copy.deepcopy(train3[0]), copy.deepcopy(train3[1])
  test3_copy = copy.deepcopy(test3[0]), copy.deepcopy(test3[1])
  print(len(test3_copy[0]))
  print(len(test3_copy[1]))
  make_train_test_csv(train3_copy, test3_copy, "3")
  #make_train_test_csv(train3, test3, "3")
  print("MADE SET #3")
  """

  #This is train/test set #4
  #cousin set
  X1 = copy.deepcopy(X)
  y1 = copy.deepcopy(y) 
  train4, test4 = remove_similars(X1, y1)   
  make_train_test_csv(train4, test4, "4")
  print("MADE SET #4")


  #This is train/test set #5
  #total set
  X1 = copy.deepcopy(X)
  y1 = copy.deepcopy(y) 
  train5, test5 = test_partition(X1,y1, assure_p=False) 
  make_train_test_csv(train5, test5, "5")
  print("MADE SET #5")

def get_validation_train_test(filename):

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


	with open(filename) as csvfile:
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

	X = np.array(X)
	y = np.array(y)
	skf = StratifiedKFold(n_splits=3)
	print(skf.get_n_splits(X, y))
	print(X)
    # X = np.array(X)
	# n = 0
	for train_index, test_index in skf.split(X, y):
		# n = n + 1
		print("TRAIN:", train_index, "TEST:", test_index)
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		# make_train_test_csv([X_train, y_train], [X_test, y_test], "validation" + n)
		# print(X_train)

	#for x in X_train:
	#	print(x)

	make_train_test_csv(copy.deepcopy(train1), copy.deepcopy(test1), "validation") 



if __name__=='__main__':
    
    filename = "final_cleaned_results.csv"
    filename_dir = os.path.join(os.getcwd(), "..", filename)

    get_train_test(filename_dir) 
