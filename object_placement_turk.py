from sklearn.model_selection import StratifiedKFold
import matrix_priors

import csv
import sys

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

  return X, y
