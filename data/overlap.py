import os
import csv
import numpy as np
import sys
def read_csv_train_test(filename):
    
    X = []
    y = []

    filename_dir = os.path.join(os.getcwd(), filename) 
    with open(filename_dir, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            X.append(row[:-1])
            y.append(row[-1])
    
    return X, y
if __name__ == "__main__":
    origX, origY = read_csv_train_test(sys.argv[1])
    origX = np.array(origX)
    origY = np.array(origY)
    X = np.array(origX)[:,0]
    y = np.array(origY)

    testX, testY = read_csv_train_test(sys.argv[2])
    testX = np.array(testX)
    testY = np.array(testY)

    

    store = {}
    freq = {}
    for i in range(len(X)):
        key = X[i]
        value = y[i]
        try:
            store[key].add(value)
        except KeyError:
            store[key] = {value}

        bigram = (key, value)
        if bigram not in freq:
            freq[bigram] = 0
        freq[bigram] += 1

    overlapCount = 0
    lowerBound = 0
    for i, words in enumerate(testX):
        shelf = words[1:]
        if words[0] in shelf: # if the word itself is in the shelf, it is guaranteed to be right
            continue          # so just skip it
        try:
            overlap = list(store[words[0]] & set(shelf))
        except KeyError:
            # if it's not there, we'll just assume it's wrong
            lowerBound += 1
            overlapCount += 1
        if len(overlap) <= 1:
            continue          # skip this case cause there is no overlap
        lowerBound += 1
        # now we want to look for the most common one. if the most common one is the right answer, then don't add to overlap count
        most_common = None
        maxi = 0
        for item in shelf:
            bigram = (words[0], item)
            try:
                count = freq[bigram]
            except KeyError:
                continue

            if count > maxi:
                maxi = count
                most_common = item
        if most_common == testY[i]:
            #print(words[0], origY[i], i, maxi)
            continue      # skip because most common word is the right answer
        #print(testX[i], testY[i])

        overlapCount += 1
    #print(freq)
    print('word itself, overlap, most frequent', overlapCount, len(origX), 1-(overlapCount/len(origX)))
    print('word itself, overlap', lowerBound, len(origX), 1-(lowerBound/len(origX)))