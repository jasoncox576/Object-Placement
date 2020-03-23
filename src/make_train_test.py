import csv
import copy
import os

def make_train_test_csv(train, test, filename_prefix):
    #seen_permutations = []
    test_filename = os.path.join(os.getcwd(), "..", (filename_prefix + "_test.csv"))
    train_filename = os.path.join(os.getcwd(), "..", (filename_prefix + "_train.csv"))
    test_csv = open(test_filename, "w")
    train_csv = open(train_filename, "w")
    
    test_writer = csv.writer(test_csv)
    train_writer = csv.writer(train_csv)


    train_x = train[0]
    train_y = train[1]
    
    test_x = test[0]
    test_y = test[1]

    for index in range(len(train_x)):
        new_row_train = train_x[index]
        new_row_train.append(train_y[index])
        train_writer.writerow(new_row_train)

        
    for index in range(len(test_x)):
        new_row_test = copy.deepcopy(test_x[index])
        #print("XX", new_row_test)
        new_row_test.append(test_y[index]) 
        #print("XX", new_row_test)
        test_writer.writerow(new_row_test)
        new_row_test.clear()

    test_csv.close()
    train_csv.close()


