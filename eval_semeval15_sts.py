"""
Evaluates the senence similarity code against the SemEval'15 Sentence Similarity Task

Requires the semeval test data to be in the directory: ./test_evaluation_task2a
It is available here:
    http://alt.qcri.org/semeval2015/task2/data/uploads/test_evaluation_task2a.tgz
More details here:
    http://alt.qcri.org/semeval2015/task2/

The results I get are:
    test_evaluation_task2a/STS.gs.headlines.txt Pearson: 0.74894
    test_evaluation_task2a/STS.gs.answers-forums.txt Pearson: 0.66934
    test_evaluation_task2a/STS.gs.images.txt Pearson: 0.82016
    test_evaluation_task2a/STS.gs.belief.txt Pearson: 0.73342
    test_evaluation_task2a/STS.gs.answers-students.txt Pearson: 0.71448

These are at better than the results reported in the original paper.
A Simple But Tough-To-Beat Baseline for Sentence Embeddings, Arora, Liang and Ma:
https://openreview.net/forum?id=SyK00v5xx

"""


import glob
from subprocess import Popen, PIPE
import spacy
import numpy as np
from sentenceemb import get_sentence_vectors

def do_sts(filename, nlp):
    all_text = []
    for line in open(filename, "r"):
        sent_a, sent_b = line.split('\t')
        all_text.append(sent_a.decode("ascii", "ignore"))
        all_text.append(sent_b.decode("ascii", "ignore"))
    sv = get_sentence_vectors(all_text, nlp = nlp)
    fout = open(filename+'.myanswer.txt', "w")
    for i in xrange(0, len(sv),2):
        sim = np.abs(np.dot(sv[i], sv[i+1]))
        fout.write("%f\n" % sim)
    fout.close()

if __name__ == "__main__":
    nlp = spacy.load('en')

    in_files = glob.glob("test_evaluation_task2a/STS.input.*.txt")
    in_files = filter(lambda x: "myanswer" not in x, in_files)
    for fname in in_files:
        do_sts(fname, nlp)
        fname_gold = fname.replace('input', 'gs')
        fname_answer = fname + ".myanswer.txt"
        proc = Popen(["perl", "./test_evaluation_task2a/correlation-noconfidence.pl", 
            fname_gold, fname_answer], stdout=PIPE)
        out, _ = proc.communicate()
        print fname_gold, out.strip()

