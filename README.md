Builds sentence vectors using the method proposed by Arora, Liang and Ma:

A Simple But Tough-To-Beat Baseline for Sentence Embeddings, Arora, Liang and Ma:
https://openreview.net/forum?id=SyK00v5xx

1. Weights and averages the GloVe vectors for each word in the sentence.
2. Removes the first principal component.
3. Normalizes the result for easy similarity checks.

Tested on SemEval'15 Sentence Similarity Task. See eval_semeval15_sts.py for
the test code.

The results are:
    test_evaluation_task2a/STS.gs.headlines.txt Pearson: 0.74894
    test_evaluation_task2a/STS.gs.answers-forums.txt Pearson: 0.66934
    test_evaluation_task2a/STS.gs.images.txt Pearson: 0.82016
    test_evaluation_task2a/STS.gs.belief.txt Pearson: 0.73342
    test_evaluation_task2a/STS.gs.answers-students.txt Pearson: 0.71448

These are at better than the results reported in the original paper.


Requires:
- spacy >= 1.6.0
- sklearn >= 0.18.1
- numpy

Quick example:

import numpy as np
from sentenceemb import get_sentence_vectors

sents = [u"This is a test.", u"A simple test sentence.", u"Don't you like my test sentences."]
vs = get_sentence_vectors(sents)
similarity = np.abs(np.dot(vs, vs.T))
print similarity

Also see demo.py and test_sentenceemb.py for more examples.
