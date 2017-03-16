import pytest
import numpy as np
from sentenceemb import *

@pytest.fixture(scope="module")
def nlp():
    import spacy
    return spacy.load('en')

def test_simple_svec_sim():
    sents = [u"the", u"the", u"the"]
    vs = get_sentence_vectors(sents, nlp=nlp())
    sim = np.abs(np.dot(vs, vs.T))
    assert np.allclose(sim, 1.0)

def test_simple_svec_sim():
    sents = [u"the cat", u"the dog", u"the review"]
    vs = get_sentence_vectors(sents, nlp=nlp())
    sim = np.abs(np.dot(vs, vs.T))
    assert sim[0, 1] > sim[0, 2]
    assert sim[0, 1] > sim[1, 2]

def test_simple_svec_sim_demo():
    sents = [u"This is a test.", u"A simple test sentence.", u"Don't you like my test sentences."]
    vs = get_sentence_vectors(sents, nlp=nlp())
    print np.abs(np.dot(vs, vs.T))
