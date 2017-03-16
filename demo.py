import sys
import numpy as np
from sentenceemb import get_sentence_vectors

if __name__ == "__main__":

    if len(sys.argv) < 4:
        print "You must input at least 3 sentences"

    vs = get_sentence_vectors([sys.argv[1].decode(), sys.argv[2].decode(), 
        sys.argv[3].decode()])
    similarity = np.abs(np.dot(vs, vs.T))
    print similarity
