from sklearn.model_selection import StratifiedKFold
import matrix_priors

import csv

def get_train_test(dictionary, filename):

        X = []
        y = []

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

                X.append([row_result[0], row_result[1], row_result[2], row_result[3]])
                #Fetch the answer to the most recent instance
                y.append(X[-1][answer])
                
        

        skf = StratifiedKFold(n_splits=5, shuffle=True)
        #print(rows_dict)
        y_numerical = y[:]
        for index, word in enumerate(y):
            #the word in the csv may be something like.. cereal_2. We need to
            #take off the _2 to get the dictionary index of the word. 
            word = matrix_priors.get_synset_and_strip(word)[1]
            row = dictionary[word]
            y_numerical[index] = row 
        """
        for train, test in skf.split(X, y_numerical):
            print(train, test)
        """
        
        return X, y, skf.split(X, y_numerical)
