from sklearn.model_selection import StratifiedKFold
import matrix_priors
import numpy as np
import csv
import sys


def instances_disagree_process(X, y):

    print("Discarding bad instances....")
    used_indices = {}
    new_X = []
    new_y = []

    for x1 in range(len(X)):
        if used_indices.get(x1): continue
        answer_count = [0,0,0]
        answer_count[X[x1][1:].index(y[x1])] += 1

        for x2 in range(len(X)):
            if (used_indices.get(x2)) or (x1 == x2): continue
            if(X[x1] == X[x2]):
                answer_count[X[x2][1:].index(y[x2])] += 1
                used_indices[x2] = 1
        used_indices[x1] = 1
        if max(answer_count) > 2:
            new_X.append(X[x1])
            new_y.append(X[x1][np.argmax(answer_count)+1])
    print("Finished discarding bad instances.")
    return new_X, new_y


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
          
  """
  skf = StratifiedKFold(n_splits=5, shuffle=True)
  y_numerical = y[:]
  for index, word in enumerate(y):
      #the word in the csv may be something like.. cereal_2. We need to
      #take off the _2 to get the dictionary index of the word. 
      row = dictionary[word]
      y_numerical[index] = row 
  """
  """
  for train, test in skf.split(X, y_numerical):
      print(train, test)
  """

  #X, y = instances_disagree_process(X, y) 
  return X, y
