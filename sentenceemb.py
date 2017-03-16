import spacy
import numpy as np
from sklearn.decomposition import TruncatedSVD

def get_sentence_vectors(sentences, a=1e-3, batch_size = 1000, n_threads = 4, nlp = None):
    """
    Builds sentence vectors using the method proposed by Arora, Liang and Ma:

    A Simple But Tough-To-Beat Baseline for Sentence Embeddings, Arora, Liang and Ma:
    https://openreview.net/forum?id=SyK00v5xx

    1. Weights and averages the GloVe vectors for each word in the sentence.
    2. Removes the first principal component.
    3. Normalizes the result for easy similarity checks.

    Note: You must call this function with all the sentences you wish to compare. Vectors generated
    in separate calls are not comparable.

    :param sentences: a list of all the sentences in the corpus, sentences are unicode strings
    :param a: the hyperparameter used for weighting sentence vectors, the default works resonably well
    :param batch_size: how to break the sentences across threads
    :param n_threads: how many threads to use for tokenization
    :param nlp: the spacy english object, loaded via spacy.load('en'), argument means you dont have
        to reload spacy if you use it elsewhere
    """

    if nlp is None:
        nlp = spacy.load('en')

    # use spacy to get the raw sentence vectors
    all_vs = []
    for i, sent in enumerate(nlp.pipe(sentences, batch_size=batch_size, n_threads=n_threads, 
                                     tag=False, parse=False, entity=False)):
        slen = len(sent)
        vw = np.array([w.vector for w in sent]) # word vectors
        pw = np.array([np.exp(w.prob) for w in sent]) # uni-gram probs
        vs = (1.0 / slen) * np.sum(a / (a + pw[:, None]) * vw, axis=0) # weighted sentence vector
        all_vs.append(vs)
    all_vs = np.array(all_vs)
    
    # get the first principal component
    svd = TruncatedSVD(n_components=1)
    svd.fit(all_vs)
    pc0 = svd.components_
    
    # remove the first principal component
    all_vs = all_vs - np.dot(all_vs, pc0.T) * pc0

    # normalize (so that cosine similarity is just the dot product)
    # you might want to remove this if you are using the vectors for something else
    all_vs /= np.linalg.norm(all_vs, axis=1, keepdims=True)

    return all_vs
