import csv
filename = "official_results.csv"

def make_train_test_csv(filename):
    seen_permutations = []
    test_csv = open("test.csv", "w")
    train_csv = open("train.csv", "w")
    
    with open(filename, "r+") as csvfile:
        reader = csv.reader(csvfile)
        test_writer = csv.writer(test_csv)
        train_writer = csv.writer(train_csv)
        
        for row in reader:
            if reader.line_num == 1: continue
            row_result = row[27:32]
            if row_result[:-1] not in seen_permutations:
                seen_permutations.append(row_result[:-1])
                test_writer.writerow(row_result)
            else:
                train_writer.writerow(row_result)

    test_csv.close()
    train_csv.close()


if __name__=="__main__":
    make_train_test_csv(filename)




