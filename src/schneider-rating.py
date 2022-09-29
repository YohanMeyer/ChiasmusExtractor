import sys
import json
from scipy.spatial import distance
import numpy as np
from tqdm import tqdm
# import fasttext
import os
import types
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeRegressor

from utility import *

# [-- ADDED CODE: BERTHOMET & MEYER --]

from embeddings import GloveEmbedding

os.environ['HOME'] = os.path.join('..', '..', 'GloVe')
glove = GloveEmbedding('common_crawl_48', d_emb=300, show_progress=True)


def glove_emb_ndarray(word: str) -> np.ndarray:
    return np.array(glove.emb(word.lower()))

# [--       END OF ADDED CODE       --]

class RatingMachine: 
    def __init__(self, gloveModel=None, verbose = False, negList = None, conjList = None, featureTypes = None, C=1, model_type = "logreg", chiasmus_regex_pattern = None, posBlacklist=None, rating_model=None):
       
        # [TODO][embedding] replace with GloVe ?
        # if fasttextModel is not None:
        #     self.fasttextModel = fasttext.load_model(fasttextModel)
        # else:
        #     self.fasttextModel = None
        if gloveModel is None:
            self.gloveModel = lambda w: np.array(glove.emb(w.lower()))
        elif isinstance(gloveModel, types.FunctionType):
            self.gloveModel = gloveModel
        else:  # Neither None nor function? Then we assume it's an array-like object
            self.gloveModel = lambda w: np.array(gloveModel[w])

        self.negList = negList
        self.conjList = conjList
        self.featureTypes = featureTypes

        self.chiasmus_regex_pattern = chiasmus_regex_pattern

        self.summary = None
        self.C = C # Inverse of regularization strength

        self.posBlacklist = posBlacklist
        if self.posBlacklist is None:
            self.posBlacklist = []

        self.model_type = model_type

        if isinstance(rating_model, str):
            with open(rating_model, 'rb') as f:
                self.model = pickle.load(f)
        else:
            self.model = rating_model

        self.positive_annotations = "TrueChiasmus"


    def rate_candidates(self, fileName, candidatesFolder = "candidates", idStart = ""):
        candidates = get_file_jsonlines(fileName, candidatesFolder)
        if(candidates == -1):
            print("Rating candidates aborted.")
            return

        assert(self.model is not None)
        print('\tget features...')
        
        features = np.asarray([
                self.get_features(candidate) for candidate in tqdm(candidates)
                ])

        ratings = self.model.decision_function(features)
        print('\trate...')

        with open(os.path.join("..", candidatesFolder, fileName), 'w') as fileOut:
            for index, candidate in enumerate(tqdm(candidates)):
                candidate['rating'] = ratings[index]
                fileOut.write(json.dumps(candidate))
                fileOut.write("\n")

            fileOut.close()
    
    def get_top(self, outputFile, topNumber, ratedFile, candidatesFolder = "candidates"):
        candidates = get_file_jsonlines(ratedFile, candidatesFolder)
        if(candidates == -1):
            print("Top rating aborted.")
        
        ratings = [candidate['rating'] for candidate in candidates]
        sorting = np.argsort(ratings)[::-1]
        numCandidates = min(topNumber, len(candidates))
        
        with open(os.path.join("..", candidatesFolder, outputFile), 'w') as fileOut:
            for i in range(numCandidates):
                json.dump(candidates[sorting[i]], fileOut, ensure_ascii=False, indent=4)

    def get_dubremetz_features(self, candidate):
        ids = candidate["index"]
        ia1 = ids[0]
        ia2 = ids[-1]
        ib1 = ids[int(len(ids)/2 - 1)]
        ib2 = ids[int(len(ids)/2)]

        words = candidate["words"]
        lemmas = candidate["lemmas"]
        dep = candidate["dep"]

        conjList = self.conjList
        negList = self.negList

        features = []


        hardp_list = ['.', '(', ')', "[", "]"] 
        softp_list = [',', ';']


        # Basic

        num_punct = 0
        for h in hardp_list:
            if h in words[ ia1+1 : ib1 ]: num_punct+=1
            if h in words[ ib2+1 : ia2 ]: num_punct+=1
        features.append(num_punct)

        num_punct = 0
        for h in hardp_list:
            if h in words[ ia1+1 : ib1 ]: num_punct+=1
            if h in words[ ib2+1 : ia2 ]: num_punct+=1
        features.append(num_punct)

        num_punct = 0
        for h in hardp_list:
            if h in words[ ib1+1 : ib2 ]: num_punct+=1
        features.append(num_punct)

        rep_a1 = -1
        if lemmas[ia1] == lemmas[ia2]:
            rep_a1 -= 1
        rep_a1 += lemmas.count(lemmas[ia1])
        features.append(rep_a1)

        rep_b1 = -1
        if lemmas[ib1] == lemmas[ib2]:
            rep_b1 -= 1
        rep_b1 += lemmas.count(lemmas[ib1])
        features.append(rep_b1)

        rep_b2 = -1
        if lemmas[ib1] == lemmas[ib2]:
            rep_b2 -= 1
        rep_b2 += lemmas.count(lemmas[ib2])
        features.append(rep_b2)

        rep_a2 = -1
        if lemmas[ia1] == lemmas[ia2]:
            rep_a2 -= 1
        rep_a2 += lemmas.count(lemmas[ia2])
        features.append(rep_b2)

        # Size

        diff_size = abs((ib1-ia1) - (ia2-ib2))
        features.append(diff_size)

        toks_in_bc = ia2-ib1
        features.append(toks_in_bc)

        # Similarity

        exact_match = ([" ".join(words[ia1+1 : ib1])] == [" ".join(words[ib2+1 : ia2])])
        features.append(exact_match)

        same_tok = 0
        for l in lemmas[ia1+1 : ib1]:
            if l in lemmas[ib2+1 : ia2]: same_tok += 1
        features.append(same_tok)

        sim_score = same_tok / (ib1-ia1)
        features.append(sim_score)

        num_bigrams = 0
        t1 = " ".join(words[ia1+1 : ib1])
        t2 = " ".join(words[ib2+1 : ia2])
        s1 = set()
        s2 = set()
        for t in range(len(t1)-1):
            bigram = t1[t:t+2]
            s1.add(bigram)
        for t in range(len(t2)-1):
            bigram = t2[t:t+2]
            s2.add(bigram)
        for b in s1:
            if b in s2: num_bigrams += 1
        bigrams_normed = (num_bigrams/max(len(s1)+1, len(s2)+1))
        features.append(bigrams_normed)

        num_trigrams = 0
        t1 = " ".join(words[ia1+1 : ib1])
        t2 = " ".join(words[ib2+1 : ia2])
        s1 = set()
        s2 = set()
        for t in range(len(t1)-2):
            trigram = t1[t:t+3]
            s1.add(trigram)
        for t in range(len(t2)-2):
            trigram = t2[t:t+3]
            s2.add(trigram)
        for t in s1:
            if t in s2: num_trigrams += 1
        trigrams_normed = (num_trigrams/max(len(s1)+1, len(s2)+1))
        features.append(trigrams_normed)

        same_cont = 0
        t1 = set(words[ia1+1:ib1])
        t2 = set(words[ib2+1:ia2])
        for t in t1:
            if t in t2: same_cont += 1
        features.append(same_cont)

        # Lexical clues

        conj = 0
        for c in conjList:
            if c in words[ib1+1:ib2]+lemmas[ib1+1:ib2]:
                conj = 1
        features.append(conj)


        neg = 0
        for n in negList:
            if n in words[ib1+1:ib2]+lemmas[ib1+1:ib2]:
                neg = 1
        features.append(neg)


        # Dependency score

        if dep[1] == dep[3]:
            features.append(1)  
        else: 
            features.append(0)

        if dep[0] == dep[2]:
            features.append(1)  
        else: 
            features.append(0)

        if dep[1] == dep[2]:
            features.append(1)  
        else: 
            features.append(0)

        if dep[0] == dep[3]:
            features.append(1)  
        else: 
            features.append(0)
        # Return
        return features

    def get_embedding_features(self, candidate):
        print("embedding features not implemented yet")
        exit(0)
        assert(self.negList is not None)
        assert(self.conjList is not None)
        ids = candidate["ids"] 
        ia1 = ids[0]-candidate["cont_ids"][0] 
        ib1 = ids[1]-candidate["cont_ids"][0] 
        ib2 = ids[2]-candidate["cont_ids"][0] 
        ia2 = ids[3]-candidate["cont_ids"][0] 
        tokens = candidate["tokens"]
        lemmas = candidate["lemmas"]
        vectors = candidate["vectors"]
        pos = candidate["pos"]
        dep = candidate["dep"]

        conjList = self.conjList
        negList = self.negList

        features = []


        hardp_list = ['.', '(', ')', "[", "]"]
        softp_list = [',', ';']

        for i in [ia1, ia2, ib1, ib2]:
            if vectors[i] is not None:
                assert(len(vectors[i] > 1))
            for j in [ia1, ia2, ib1, ib2]:
                if j <= i:
                    continue
                if vectors[i] is None or vectors[j] is None:
                    features.append(1)
                else:
                    features.append(distance.cosine(vectors[i], vectors[j]))

        return np.asarray(features)

    def get_lexical_features(self, candidate):
        # use only four words of the chiasmus to match all data instances for training
        ids = candidate["index"]
        wordsIds = [ids[0], ids[int(len(ids)/2 - 1)], ids[int(len(ids)/2)], ids[-1]]
    
        lemmas = candidate["lemmas"]
        nbWords = len(wordsIds)
        features = []
        
        for index in range(nbWords):
            for index2 in range(index, nbWords):
                features.append(int(
                        lemmas[wordsIds[index]] == 
                        lemmas[wordsIds[index2]]))

        return np.asarray(features)

    def get_features(self, candidate):
        assert(self.featureTypes is not None)
        funcs = {
                "embedding": self.get_embedding_features,
                "lexical": self.get_lexical_features,
                "dubremetz": self.get_dubremetz_features,
                }
        features = [funcs[ft](candidate) for ft in self.featureTypes]
        return np.concatenate(features, axis=0)

    def _preprocess_training_data(self, data):
        assert(self.gloveModel is not None)
        #print("compute vectors if needed")
        for instance in data:
            if "vectors" in instance:
                continue
            words = instance["words"]
            vectors = [self.gloveModel(word) for word in words]
            # [TODO][embedding] adapt with our embedding ?
            instance["vectors"] = vectors

        #print("turn into numpy arrays")


        x = []
        y = []
        for instance in data:
            x.append(self.get_features(instance))
            # "cats" is currently imposed by the usage of Doccano
            y.append(1 if self.positive_annotations in instance["cats"] else 0)

        x = np.asarray(x)
        y = np.asarray(y)

        return x, y
    
    def _train(self, x, y):
        model = None
        if self.model_type == "logreg":
            # default model type
            model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        class_weight = "balanced",
                        max_iter=1000,
                        C = self.C))
        elif self.model_type.lower() == 'rbf svm':
            model = make_pipeline(
                    StandardScaler(),
                    SVC(
                        class_weight = "balanced",
                        gamma='scale',
                        max_iter=1000,
                        C = self.C))
        elif self.model_type.lower() == 'decisiontree':
            model = make_pipeline(
                    StandardScaler(),
                    DecisionTreeRegressor()
                    )

        else:
            print("ERROR:", self.model_type, 'does not exist')
            exit(-1)
            
        assert(model is not None)
        model.fit(x, y)

        scores = model.decision_function(x)
        ap = average_precision_score(y, scores, average='macro')
        return model, ap

    def train(self, trainingFile, trainingFolder = "annotated", keep_model = True):
        data = get_file_jsonlines(trainingFile, trainingFolder)
        if(data == -1):
            print("Training aborted.")
        x, y = self._preprocess_training_data(data)
        
        model, train_ap = self._train(x, y)
        if keep_model:
            self.model = model

    def train_with_crossval(self, trainingFile, trainingFolder = "annotated", numRuns = 5):
        data = get_file_jsonlines(trainingFile, trainingFolder)
        if(data == -1):
            print("Training aborted.")
        x, y = self._preprocess_training_data(data)
        
        kf = StratifiedKFold(n_splits = numRuns)

        aps_test = []
        aps_train = []

        for train_index, test_index in kf.split(x, y):
            x_train = x[train_index, :]
            y_train = y[train_index]
            x_test = x[test_index, :]
            y_test = y[test_index]

            model, ap_train = self._train(x_train, y_train)
            scores = model.decision_function(x_test)
            ap_test = average_precision_score(y_test, scores, average='macro')
            aps_train.append(ap_train)
            aps_test.append(ap_test)

        ap_train = np.mean(np.asarray(aps_train))
        ap_test = np.mean(np.asarray(aps_test))
        ap_train_std = np.std(np.asarray(aps_train))
        ap_test_std = np.std(np.asarray(aps_test))

        self.summary = {
                "ap_train": ap_train,
                "ap_test": ap_test,
                "ap_train_std": ap_train_std,
                "ap_test_std": ap_test_std
                }





def main():

    # -- Initializing the project --

    if len(sys.argv) == 2:
        fileName = sys.argv[1]
    else:
        fileName = input('Enter the name of the file to process : ')
        
    if(fileName[:-17] != "-candidates.jsonl"):
        fileName = fileName + "-candidates.jsonl"
        
    content = get_file_content(fileName, "candidates")
    if(content == -1):
        print("File not found.")
        exit(0)
        
    print('initialize rating machine')
    chiRate = RatingMachine(
            # fasttextModel = '../schneider/fasttext_models/wiki.en.bin',
            gloveModel = None,
            # featureTypes = ['dubremetz', 'lexical', 'embedding'],
            featureTypes = ['dubremetz', 'lexical'],
            conjList = ["and", "so", "because", "neither", "nor", "but", "for", "yet"],
            negList = ["no", "not", "never", "nothing"],
            posBlacklist = ["SPACE", "PUNCT", "PROPN", "DET"],
            )

    print('train with crossvalidation')
    chiRate.train_with_crossval(
            trainingFile = "small-chiasmi-annotated.jsonl",
            trainingFolder = "annotated",
            numRuns = 5
            )

    print('train on whole dataset')
    chiRate.train(
            trainingFile = "small-chiasmi-annotated.jsonl",
            trainingFolder = "annotated",
            keep_model = True
            )

    print('rate candidates')
    chiRate.rate_candidates(
            fileName = fileName,
            candidatesFolder = "candidates",
            idStart = "test_"
            )

    chiRate.get_top(
            outputFile = fileName[:-16] + "top-results.jsonl", # remove "candidates.jsonl"
            topNumber = 100,
            ratedFile = fileName,
            candidatesFolder = "candidates")

if __name__ == "__main__":
    main()

