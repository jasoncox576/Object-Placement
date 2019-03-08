from tensorflow.contrib.tensorboard.plugins import projector
from gensim.models.phrases import Phrases, Phraser
from spacy.lang.en.stop_words import STOP_WORDS
import re

#Current work towards extracting bigram phrases from data below
# E.G. Data file contains 'new','york', but we want 'new_york'


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

    print(sentences)
    phrases = Phrases(sentences,
                      min_count=1,
                      threshold=10,
                      delimiter=b'_',
                      progress_per=1000)
    return Phraser(phrases)
    #return phrases

def sentence_to_bi_grams(phrases_model, sentence):
    return ' '.join(phrases_model[sentence])


def sentences_to_bi_grams(n_grams, input_file_name, output_file_name):
    sentence_counter = 0
    with open(input_file_name, 'r') as input_file_pointer:
        with open(output_file_name, 'w+') as out_file:
            for sentence in get_sentences(input_file_pointer):
                sentence_counter += 1 
                cleaned_sentence = clean_sentence(sentence)
                tokenized_sentence = tokenize(cleaned_sentence)
                #print(tokenized_sentence)
                parsed_sentence = sentence_to_bi_grams(n_grams, tokenized_sentence)
                #print(parsed_sentence)
                if "new_york" in parsed_sentence:
                    print("NEW YORK")
                out_file.write(parsed_sentence + '\n')
    print("TOTAL SENTENCES: " + str(sentence_counter))


with open('text8', 'r') as filename:

    sentences = get_sentences(filename)
    phraser = build_phrases(sentences)
    sentences_to_bi_grams(phraser, 'text8', 'modified_text') 

