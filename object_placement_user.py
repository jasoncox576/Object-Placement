from word2vec_eval import *


def spacy_rank_object_sims(w1, object_labels):

        nlp = matrix_priors.nlp

        sim_ranking = []
        primary_token = nlp(w1)
        tokens = [nlp(label) for label in object_labels]
        for label in tokens:
                sim = primary_token.similarity(label) 
                sim_ranking.append((object_labels[tokens.index(label)], sim))

        sim_ranking.sort(key = lambda x: x[1], reverse=True)
        print(sim_ranking)

        return [elem[0] for elem in sim_ranking]



def rank_object_sims(w1, object_labels, embeddings, weights, dictionary, bigram_split=False):

        sim_ranking = []
        for label in object_labels:
                sim = cossim(w1, label, embeddings, weights, dictionary, bigram_split)
                sim_ranking.append((label, sim))

        sim_ranking.sort(key = lambda x: x[1], reverse=True)

        print(sim_ranking)

        return [elem[0] for elem in sim_ranking]


def cossim(w1, w2, embeddings, weights, dictionary, bigram_split=False):

        index1, _, n_embed1 = get_word_primary(w1, dictionary, matrix_priors, embeddings)
        index2, _, n_embed2 = get_word_primary(w2, dictionary, matrix_priors, embeddings)
        sim = np.dot(n_embed1, n_embed2)


        print("SIM BETWEEN " + str(w1) + " and " + str(w2) + ": "  + str(sim))

        #NOTE: may use bigram split later, something to implement

        return sim


def prepare_model():
        fil9_dir = '/home/rucksack/workspaces/jasoncox_ws/data/fil9_bigram'
        dictionaries = get_pretrain_dictionaries(fil9_dir)
        embeddings, weights = get_embeddings('/home/rucksack/workspaces/jasoncox_ws/models/2_turk+wiki+turk_cosine', fil9_dir, dictionaries)
        important_dictionary = dictionaries[2]
        reverse_dictionary = dictionaries[3]


        return embeddings, weights, important_dictionary, reverse_dictionary




if __name__ == '__main__':
        embeddings, weights, dictionary, reverse_dictionary = prepare_model()
        w1 = 'bread'
        words = ['water', 'tea', 'blueberry', 'banana', 'meat', 'apple juice', 'juice', 'pear', 'cereal', 'potato chips', 'crackers', 'corn']

        spacy_rank_object_sims(w1, words)       


