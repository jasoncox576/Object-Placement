from nltk.corpus import wordnet as wn
import random

write_file = open("random_objects", "w+")
object_synsets = []

physical_object = wn.synsets('physical_object')[0]
all_synsets = list(wn.all_synsets('n'))

physical_objects = 0 





def find_bad_hypernyms(to_exclude, synset):

    for elem in to_exclude:
        if str(synset.lowest_common_hypernyms(elem)[0]) == str(elem):
            return True
    return False







while physical_objects < 55:
    chosen_object = random.choice(all_synsets)
    if str(chosen_object.lowest_common_hypernyms(physical_object)[0]) == str(physical_object):
        # Additional filtering
        
        to_exclude = []
        to_exclude.append(wn.synsets('animal')[0])
        to_exclude.append(wn.synsets('person')[0])
        to_exclude.append(wn.synsets('location')[0])
        to_exclude.append(wn.synsets('vehicle')[0]) 
        # This may or may not be a good idea as a lot of the objects end up being
        # random plants such as 'missouri goldenrod', but then again you could
        # conceivably have something useful like 'house plant'
        to_exclude.append(wn.synsets('living_thing')[0])

        to_exclude.append(wn.synsets('structure')[0])

        to_exclude.append(wn.synsets('land')[3])
        #If we want fruit, we should not exclude that.
        to_exclude.append(wn.synsets('plant_part')[0])

        to_exclude.append(wn.synsets('particle')[0])
        to_exclude.append(wn.synsets('particle')[1])
        to_exclude.append(wn.synsets('celestial_body')[0])
        to_exclude.append(wn.synsets('body')[0])
        to_exclude.append(wn.synsets('facility')[0])
        to_exclude.append(wn.synsets('way')[5])

        if find_bad_hypernyms(to_exclude, chosen_object):
            continue

        object_synsets.append(str(chosen_object)) 
        write_file.write(str(chosen_object) + '\n')
        print(chosen_object)
        physical_objects += 1

write_file.close()

