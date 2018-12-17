import csv
import spacy
from nltk.corpus import wordnet as wn
import numpy as np

print("Loading SPACY..")
nlp = spacy.load('en_core_web_lg')
print("Loading complete")

#important columns:
#27-31 starting from 0

total_row_counter = 0
embedding_correct_counter = 0
wordnet_correct_counter = 0


filename = "dummy_results.csv"
with open(filename, newline='') as csvfile:
    #reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    reader = csv.reader(csvfile)
    for row in reader:
        row_result = row[27:32]
        
        for i in range(len(row_result)-1):
            row_result[i] = row_result[i][:-6] 

        primary = row_result[0]

        primary_token = nlp(primary)
        
        primary_syn = wn.synsets(primary)[0]

        answer = row_result[4]

        token1 = nlp(row_result[1])
        synset1 = wn.synsets(row_result[1])[0]

        token2 = nlp(row_result[2])
        synset2 = wn.synsets(row_result[2])[0]

        token3 = nlp(row_result[3])
        synset3 = wn.synsets(row_result[3])[0]


        if answer == "Top":
            answer = 0
        elif answer == "Middle":
            answer = 1
        elif answer == "Bottom":
            answer = 2

        embedding_sim_vector = []
        embedding_sim_vector.append(primary_token.similarity(token1))
        embedding_sim_vector.append(primary_token.similarity(token2))
        embedding_sim_vector.append(primary_token.similarity(token3))

        wordnet_sim_vector = []
        wordnet_sim_vector.append(primary_syn.path_similarity(synset1))
        wordnet_sim_vector.append(primary_syn.path_similarity(synset2))
        wordnet_sim_vector.append(primary_syn.path_similarity(synset3))

        

        if np.argmax(embedding_sim_vector) == answer:
            embedding_correct_counter += 1
        if np.argmax(wordnet_sim_vector) == answer:
            wordnet_correct_counter += 1
        total_row_counter += 1

print("Total accuracy of word2vec is:")
print(str(embedding_correct_counter / total_row_counter))

print("\nTotal accuracy of wordnet is:")
print(str(wordnet_correct_counter / total_row_counter))


