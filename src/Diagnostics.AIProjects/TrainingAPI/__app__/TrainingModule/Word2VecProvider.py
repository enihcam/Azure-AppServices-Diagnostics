import gensim, os, logging, gc
import numpy as np
from __app__.AppSettings.AppSettings import appSettings

class Word2VecProvider:
    def __init__(self):
        word2vecpath = appSettings.WORD2VEC_PATH
        fileName = appSettings.WORD2VEC_MODEL_NAME
        logging.info("Now loading word2vec model")
        self.w2vModel = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(word2vecpath, fileName), binary=True)
        self.subsetMatrix = None

    def extractSubset(self, gensimDictionary, writepath):
        alltokens = [(k,v) for k, v in gensimDictionary.token2id.items()]
        self.subsetMatrix = np.zeros((len(alltokens), 300))
        for token, idx in alltokens:
            try:
                self.subsetMatrix[idx, :] = self.w2vModel[token]
            except KeyError as e:
                pass
        fullpath = os.path.join(os.getcwd(), os.path.normpath(writepath))
        np.save(fullpath, self.subsetMatrix)
        del self.w2vModel
        gc.collect()
    
    def tfIdf2W2V(self, weightedTokens):
        result = np.zeros(300)
        for wordId, weight in weightedTokens:
            try:
                w2v = self.subsetMatrix[wordId, :]
                result += w2v*weight
            except Exception as e:
                pass
        return result

    def getSentenceVector(self, sentenceTokens):
        result = np.zeros((len(sentenceTokens), 300))
        for i in range(len(sentenceTokens)):
            result[i, :] = self.tfIdf2W2V(sentenceTokens[i])
        return result