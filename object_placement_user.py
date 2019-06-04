from word2vec_eval import *

def cossim(w1, w2, embeddings, weights, dictionary, bigram_split=False):

        if not bigram_split:
                _, _, n_embed1 = get_word_primary(w1, dictionary, matrix_priors, embeddings)
                _, _, n_embed2 = get_word_primary(w1, dictionary, matrix_priors, embeddings)
                sim = np.dot(nembed1, nembed2)

        #NOTE: may use bigram split later, something to implement



def prepare_model():
	fil9_dir = 'path/to/fil9_bigram'
	dictionaries = get_pretrain_dictionaries(fil9_dir)
	embeddings, weights = get_embeddings('2_turk+wiki+turk_cosine', fil9_dir, dictionaries)
	important_dictionary = dictionaries[2]

	return embeddings, weights, important_dictionary
