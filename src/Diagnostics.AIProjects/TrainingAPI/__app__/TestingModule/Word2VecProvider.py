import os
import numpy as np

class Word2VecProvider:
    def __init__(self, modelpackagepath):
        self.word2vec = np.load(os.path.join(modelpackagepath, "word2vec.dat.npy"))
        
    def tfIdf2W2V(self, weightedTokens):
        result = np.zeros(300)
        for idx, weight in weightedTokens:
            try:
                w2v = self.word2vec[idx, :]
                result += w2v*weight
            except Exception as e:
                pass
        return result