import csv
import glob

def parse_csv(filename):
    write_dict = {}
    with open(filename, "r+") as csvfile:
        #reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        reader = csv.reader(csvfile)
        input_label = {}
        current_items = []
        empty = "empty"
        for row in reader:
            answer = row[0]
            answer_label = row[1]
            answer_label = answer_label[1:]

            if answer != empty and answer_label != empty:
                write_key = answer + "," + answer_label
                write_val = ','.join(current_items)
                # print(write_key)
                # print(write_val)
                # print('\n')
                write_dict[write_key] = write_val
                input_label[answer] = answer_label
            if answer != empty and answer not in current_items:
                current_items.append(answer)
            if answer_label != empty and answer_label not in current_items:
                current_items.append(answer_label)
    return write_dict

def write_csv(write_filename, all_write_dict):
    with open(write_filename, "w+") as write_file:
        writer = csv.writer(write_file)
        for write_dict in all_write_dict:
            for key, value in write_dict.items():
                writer.writerow([key, value])

if __name__=="__main__":
    path = "/home/users/fri/lina_ws/Object-Placement/annotations/*.csv"
    # path = "/u/linaly/Documents/annotations/*.csv"
    write_filename = "clean_annotations.csv"
    all_write_dict = []
    for fname in glob.glob(path):
        if write_filename not in fname:
            write_dict = parse_csv(fname)
            all_write_dict.append(write_dict)
    write_csv(write_filename, all_write_dict)
    print("Finish parsing")