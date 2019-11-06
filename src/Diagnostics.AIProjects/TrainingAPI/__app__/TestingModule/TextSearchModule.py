import os, gc
import logging
import json
from __app__.TestingModule.ModelInfo import ModelInfo
from __app__.TrainingModule.TokenizerModule import *
from __app__.TestingModule.Word2VecProvider import Word2VecProvider
import numpy as np
from gensim.models import TfidfModel
from gensim import corpora, similarities

SITE_ROOT = os.getcwd()
logging.info("SITE_ROOT: {0}".format(SITE_ROOT))

def absPath(path):
    return os.path.join(SITE_ROOT, path)

class ModelFileConfigFailed(Exception):
    pass
class ModelFileLoadFailed(Exception):
    pass
class ModelRefreshException(Exception):
    pass

packageFileNames = {
    "dictionaryFile": "dictionary.dict",
    "m1ModelFile": "m1.model",
    "m1IndexFile": "m1.index",
    "m2ModelFile": "m2.model",
    "m2IndexFile": "m2.index",
    "detectorsFile": "Detectors.json",
    "sampleUtterancesFile": "SampleUtterances.json",
    "mappingsFile": "Mappings.json",
    "modelInfo": "ModelInfo.json"
}

optionalFiles = ["mappingsFile", "modelInfo", "m1IndexFile"]

#### Text Search model for Queries ####
def verifyFile(filename, prelogMessage=""):
    try:
        with open(absPath(filename), "rb") as fp:
            fp.close()
        logging.info("TextSearchModule: {0}Verified File {1}".format(prelogMessage, absPath(filename)))
        return True
    except FileNotFoundError:
        logging.error("TextSearchModule: {0}Failed to Verify File {1}".format(prelogMessage, absPath(filename)))
        return False

class TextSearchModel:
    def __init__(self, modelpackagepath, packageFiles):
        for key in packageFiles.keys():
            packageFiles[key] = absPath(os.path.join(modelpackagepath, packageFiles[key]))
        self.packageFiles = packageFiles
        self.models = {"dictionary": None, "m1Model": None, "m1Index": None, "m2Model": None, "m2Index": None, "detectors": None, "sampleUtterances": None, "mappings": None, "modelInfo": None}
        try:
            with open(self.packageFiles["modelInfo"], "r") as fp:
                self.models["modelInfo"] = ModelInfo(json.loads(fp.read()))
                fp.close()
        except:
            self.models["modelInfo"] = ModelInfo({})
        try:
            self.models["dictionary"] = corpora.Dictionary.load(self.packageFiles["dictionaryFile"])
        except:
            raise ModelFileLoadFailed("Failed to load dictionary from file " + self.packageFiles["dictionaryFile"])
        try:
            self.models["m1Model"] = TfidfModel.load(self.packageFiles["m1ModelFile"])
        except:
            raise ModelFileLoadFailed("Failed to load model from file " + self.packageFiles["m1ModelFile"])
        try:
            if self.models["modelInfo"].useWord2Vec:
                self.models["m1Index"] = np.load(self.packageFiles["m1IndexFile"].replace("index", "dat.npy"))
                self.models["w2vProvider"] = Word2VecProvider(absPath(modelpackagepath))
            else:
                self.models["m1Index"] = similarities.MatrixSimilarity.load(self.packageFiles["m1IndexFile"])
        except:
            raise ModelFileLoadFailed("Failed to load index from file " + self.packageFiles["m1IndexFile"])
        try:
            self.models["m2Model"] = TfidfModel.load(self.packageFiles["m2ModelFile"])
            self.models["m2Model"] = None
            gc.collect()
        except:
            raise ModelFileLoadFailed("Failed to load model from file " + self.packageFiles["m2ModelFile"])
        try:
            self.models["m2Index"] = similarities.MatrixSimilarity.load(self.packageFiles["m2IndexFile"])
            self.models["m2Index"] = None
            gc.collect()
        except:
            raise ModelFileLoadFailed("Failed to load index from file " + self.packageFiles["m2IndexFile"])
        try:
            with open(self.packageFiles["detectorsFile"], "r") as f:
                self.models["detectors"] = json.loads(f.read())
                f.close()
        except:
            raise ModelFileLoadFailed("Failed to parse json from file " + self.packageFiles["detectorsFile"])
        if self.models["modelInfo"].detectorContentSplitted:
            try:
                with open(self.packageFiles["mappingsFile"], "r") as f:
                    self.models["mappings"] = json.loads(f.read())
                    f.close()
            except:
                raise ModelFileLoadFailed("Failed to parse json from file " + self.packageFiles["mappingsFile"])
        try:
            with open(self.packageFiles["sampleUtterancesFile"], "r") as f:
                self.models["sampleUtterances"] = json.loads(f.read())
                f.close()
                self.models["sampleUtterances"] = None
                gc.collect()
        except:
            raise ModelFileLoadFailed("Failed to parse json from file " + self.packageFiles["sampleUtterancesFile"])

    def getDetectorByIndex(self, index):
        detector = [x for x in self.models["mappings"] if (x["startindex"] <= index <= x["endindex"])]
        if detector and detector[0]:
            return detector[0]["id"]
        else:
            return None
    
    def cosineDistance(self, vector1, vector2):
        mag_1 = np.linalg.norm(vector1)
        mag_2 = np.linalg.norm(vector2)
        res = 0
        if mag_1>0 and mag_2>0:
            res = np.dot(vector1, vector2)/(mag_1*mag_2)
        return res

    def queryDetectors(self, query=None):
        if query:
            try:
                vector = self.models["m1Model"][self.models["dictionary"].doc2bow(getAllNGrams(query, self.models["modelInfo"].textNGrams))]
                if self.models["modelInfo"].detectorContentSplitted:
                    if self.models["modelInfo"].useWord2Vec:
                        query_w2v = self.models["w2vProvider"].tfIdf2W2V(vector)
                        similarities = np.apply_along_axis(lambda x: float(self.cosineDistance(x, query_w2v)), 1, self.models["m1Index"])
                        similar_doc_indices = sorted(list(enumerate(similarities)), key=lambda x: x[1], reverse=True)[:10]
                    else:
                        similar_doc_indices = sorted(enumerate(self.models["m1Index"][vector]), key=lambda item: -item[1])[:10]
                    similar_docs = []
                    for x in similar_doc_indices:
                        detector = self.getDetectorByIndex(x[0])
                        if detector and (not (detector in [p["detector"] for p in similar_docs])):
                            similar_docs.append({"detector": self.getDetectorByIndex(x[0]), "score": str(x[1])})
                else:
                    similar_doc_indices = sorted(enumerate(self.models["m1Index"][vector]), key=lambda item: -item[1])
                    similar_docs = list(map(lambda x: {"detector": self.models["detectors"][x[0]]["id"], "score": str(x[1])}, similar_doc_indices))
                return {"query": query, "results": similar_docs}
            except Exception as e:
                return {"query": query, "results": [], "exception": str(e)}
        return None

    def loadUtteranceModel(self):
        self.models["m2Model"] = TfidfModel.load(self.packageFiles["m2ModelFile"])
        self.models["m2Index"] = similarities.MatrixSimilarity.load(self.packageFiles["m2IndexFile"])
        with open(self.packageFiles["sampleUtterancesFile"], "r") as f:
            self.models["sampleUtterances"] = json.loads(f.read())
            f.close()

    def unloadUtteranceModel(self):
        self.models["m2Model"] = None
        self.models["m2Index"] = None
        self.models["sampleUtterances"] = None
        gc.collect()

    def queryUtterances(self, query=None, existing_utterances=[]):
        if query:
            query = query + " ".join(existing_utterances)
            self.loadUtteranceModel()
            try:
                vector = self.models["m2Model"][self.models["dictionary"].doc2bow(getAllNGrams(query, self.models["modelInfo"].textNGrams))]
                similar_doc_indices = sorted(enumerate(self.models["m2Index"][vector]), key=lambda item: -item[1])
                similar_doc_indices = [x for x in similar_doc_indices if self.models["sampleUtterances"][x[0]]["text"].lower() not in existing_utterances][:10]
                similar_docs = list(map(lambda x: {"sampleUtterance": self.models["sampleUtterances"][x[0]], "score": str(x[1])}, similar_doc_indices))
                return {"query": query, "results": similar_docs}
            except Exception as e:
                return {"query": query, "results": None}
            self.unloadUtteranceModel()
        return None
    
    def runTestCases(self, testCases, passThreshold=0.5, publishThreshold=0.95):
        numpassed = 0
        failedTestCases = []
        for testCase in testCases:
            testCase.run(self, passThreshold)
            if testCase.isPassed:
                numpassed += 1
        logging.info("Total test cases: {0}".format(len(testCases)))
        logging.info("Passed test cases: {0}".format(numpassed))
        logging.info("Failed test case details:\n{0}".format("\n".join(["{0}\t\t{1}".format(testCase.query, ",".join(testCase.failDetails)) for testCase in testCases if not testCase.isPassed])))
        if numpassed/len(testCases)>publishThreshold:
            return True
        return False

def loadModel(productid):
    modelpackagepath = os.path.normpath(productid)
    logging.info("TextSearchModule: Loading model for product {0}: Loading from folder {1}".format(productid, modelpackagepath))
    if not os.path.isdir(absPath(modelpackagepath)):
        logging.info("TextSearchModule: Loading model for product {0}: Could not find model folder.".format(productid))
    if not all([verifyFile(os.path.join(modelpackagepath, packageFileNames[x])) for x in packageFileNames.keys() if x not in optionalFiles]):
        logging.error("modelLoadTask: TextSearchModule: Loading model for product {0}: {1}".format(productid, "One or more of model file(s) are missing"))
        raise FileNotFoundError("One or more of model file(s) are missing")
    return TextSearchModel(modelpackagepath, dict(packageFileNames))