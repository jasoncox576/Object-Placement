from nltk.corpus import wordnet as wn
import random

write_file = open("random_objects", "w+")
object_synsets = []

physical_object = wn.synsets('physical_object')[0]
all_synsets = list(wn.all_synsets('n'))

physical_objects = 0 

while physical_objects < 100:
    chosen_object = random.choice(all_synsets)
    if str(chosen_object.lowest_common_hypernyms(physical_object)[0]) == str(physical_object):
        object_synsets.append(str(chosen_object)) 
        write_file.write(str(chosen_object) + '\n')
        print(chosen_object)
        physical_objects += 1
    else:
        continue

write_file.close()

