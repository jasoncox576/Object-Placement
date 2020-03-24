import csv

samples_per_aisle = 1000
sent_len = 4



#from tensorflow.contrib.tensorboard.plugins import projector
from gensim.models.phrases import Phrases, Phraser
from spacy.lang.en.stop_words import STOP_WORDS
import re
import os

def get_sentences(input_file_pointer):
    sentences = []
    while True:
        line = input_file_pointer.readline()
        if not line:
            break
        #yield line
        sentences.append(line)
    return sentences

def clean_sentence(sentence):

    sentence = sentence.lower().strip()
    sentence = re.sub(r'[^a-z0-9\s]', '', sentence)
    return re.sub(r'\s{2,}', ' ', sentence)

def tokenize(sentence):
    return [token for token in sentence.split(" ") if token not in STOP_WORDS]
    #return [token for token in sentence.split()]

def build_phrases(sentences):

    for i in range(len(sentences)):
        sentences[i] = tokenize(clean_sentence(sentences[i]))

    #print(sentences)
    phrases = Phrases(sentences,
                      min_count=1,
                      threshold=10,
                      delimiter=b'_',
                      progress_per=1000)
    return Phraser(phrases)
    #return phrases

def sentence_to_bi_grams(phrases_model, sentence):
    return ' '.join(phrases_model[sentence])


def sentences_to_bi_grams(n_grams, sentences):
    sentence_counter = 0
    output = []
    for sentence in sentences:
        sentence_counter += 1 
        cleaned_sentence = clean_sentence(sentence)
        tokenized_sentence = tokenize(cleaned_sentence)
        #print(tokenized_sentence)
        parsed_sentence = sentence_to_bi_grams(n_grams, tokenized_sentence)
        #print(parsed_sentence)
        output.append(parsed_sentence)
    #print("TOTAL SENTENCES: " + str(sentence_counter))
    return output


# Read in the aisle data. 
# TODO: Potentially do this with department data so associations between
# data in similar groups happen as well such as breakfast items together 
# rather than all the juices together
aisle_dict = None
with open('aisles.csv', mode='r') as infile:
    reader = csv.reader(infile)
    aisle_dict = {rows[0]:rows[1] for rows in reader}
del aisle_dict['aisle_id']

aisles = [[] for x in range(len(aisle_dict))]

# read in product data
with open('products.csv', mode='r') as infile:
    reader = csv.reader(infile)
    firstLine = True
    for row in reader:
        if firstLine:
            firstLine = False
            continue
        
        aisles[int(row[2])-1].append(row[1])

all_data = [item for sublist in aisles for item in sublist]


with open('../Object-Placement/text8', 'r') as filename:
    sentences = get_sentences(filename)
    phraser = build_phrases(sentences)
# create bigrams of the sentences now
#sentences = all_data
#sents = sentences.copy()
#phraser = build_phrases(sents)

for i in range(len(aisles)):
    aisles[i] = sentences_to_bi_grams(phraser, aisles[i])




# create real sentences
from iteration_utilities import random_permutation

dataset = []
for i in range(len(aisles)):
    for j in range(samples_per_aisle):
        perm = random_permutation(aisles[i], sent_len)
        dataset.append(perm)

with open("instacart_train.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(dataset)
