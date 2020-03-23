import csv
import os
cwd = os.getcwd()

filename = "official_results.csv"
write_filename = "cleaned_results.csv"
new_csv_filename = "new_turk_rows.csv"

filename_dir = os.path.join(cwd, "..", filename)
write_filename_dir = os.path.join(cwd, "..", write_filename)
new_csv_filename_dir = os.path.join(cwd, "..", new_csv_filename)




def instances_disagree(X, y):
   
    used_indices = []
    
    for x1 in range(len(X)):
        if x1 in used_indices: continue
        class_disagreements = [(X[x1], y[x1])]
        for x2 in range(len(X)):
            if (x2 in used_indices) or (x1 == x2): continue
            if (X[x1] == X[x2]) and (y[x1] != y[x2]):
                class_disagreements.append((X[x2], y[x2]))
                used_indices.append(x2)
        used_indices.append(x1)
        print("CLASS DISAGREEMENTS----------------")
        print(class_disagreements)



def make_new_turk_csv(bad_rows):
    """
    Takes all of the messed up rows from
    the first data run and generates a csv
    for turk so that proper data may be
    obtained a second go-round
    """
    
    with open(new_csv_filename_dir, "w") as csvfile:
        writer = csv.writer(csvfile) 
        first_row = ["image_url", "image_url2", "image_url3", "image_url4"]
        writer.writerow(first_row)
        for row in bad_rows:
            writer.writerow(row)


def verify_and_clean_data():

    """
    Data is 'bad' if object A is not placed with itself
    when there is an opportunity to do so

    mistakes is the number of times this occurs,
    total_sames is the number of times an object is already
    on the shelf

    Will remove all of the bad instances from the csv, plus all
    the instances by workers flagged as 'bad'.
    """
    mistakes = 0

    bad_users = []
    bad_row_objects = []
    total_rows = 0

    with open(filename_dir, "r+") as csvfile:
        #reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        reader = csv.reader(csvfile)

        for row in reader:
            if reader.line_num == 1:
                continue

            row_result = row[27:32]

            workerID = row[15]

            answer_label = row_result[4]
            answer = (["Top", "Middle", "Bottom"].index(answer_label)) + 1
            
            if row_result[0] in row_result[1:]:
                if row_result[0] != row_result[answer]:
                    # In this case, the piece of data and worker are bad.
                    bad_users.append(workerID)

        
        #Once we have all of the bad users, second pass through the data to clear
        #out all of the instances by those users.
    with open(filename_dir, "r+") as csvfile:
        reader = csv.reader(csvfile)
        with open(write_filename_dir, "r+") as write_file:
            writer = csv.writer(write_file)
            for row in reader:
                row_objects = row[27:31]
                if reader.line_num != 1:  total_rows += 1
                print("ROW")
                workerID = row[15]
                if workerID not in bad_users:
                    writer.writerow(row)
                else:
                    mistakes+=1
                    print("Bad worker")
                    bad_row_objects.append(row_objects)



    return mistakes, total_rows, mistakes/total_rows, bad_row_objects

if __name__=="__main__":
    performance = verify_and_clean_data()
    bad_row_objects = performance[-1]
    print("VALIDITY OF DATA BEFORE REMOVING: ", 1-performance[2])
    filename = "cleaned_results.csv" 
    filename_dir = os.path.join(cwd, "..", filename)
    performance2 = verify_and_clean_data()
    print("VALIDITY OF DATA AFTER REMOVING: ", 1-performance2[2])

    print("\n\n Generating new CSV with HITS for corrupted instances")
    make_new_turk_csv(bad_row_objects) 






