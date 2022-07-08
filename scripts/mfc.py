#################################################
# mfc.py
#
# Calculates moral foundations values for a list of strings.
# Last updated: 1.10.21
#################################################

# General Workflow:
#     1. Load model -> mfc.loadModel('../GoogleNews-vectors-negative300.bin')
#     2. Set dictionaires -> mfc.setDictionaries(mfc.mfd)
#     3. Calculate weights using the method of your choosing -> mfc.computeWeights

import gensim
import string
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt


mfd = {
    "Authority_virtue": ["authority", "obey", "respect", "tradition"],
    "Authority_vice": ["subversion", "disobey", "disrespect", "chaos"],
    "Care_virtue": ["kindness", "compassion", "nurture", "empathy"],
    "Care_vice": ["suffer", "cruel", "hurt", "harm"],
    "Fairness_virtue": ["fairness", "equality", "justice", "rights"],
    "Fairness_vice": ["cheat", "fraud", "unfair", "injustice"],
    "Loyalty_virtue": ["loyal", "solidarity", "patriot", "fidelity"],
    "Loyalty_vice": ["betray", "treason", "disloyal", "traitor"],
    "Sanctity_virtue": ["purity", "sanctity", "sacred", "wholesome"],
    "Sanctity_vice": ["impurity", "depravity", "degradation", "unnatural"]
}

mfd2 = {
    "Authority": mfd["Authority_virtue"]+mfd["Authority_vice"],
    "Care": mfd["Care_virtue"]+mfd["Care_vice"],
    "Fairness": mfd["Fairness_virtue"]+mfd["Fairness_vice"],
    "Loyalty": mfd["Loyalty_virtue"]+mfd["Loyalty_vice"],
    "Sanctity": mfd["Sanctity_virtue"]+mfd["Sanctity_vice"]
}

# initialize model and dictionary concepts and representations
model = False
dictionaries = {
    'concepts': False,
    'reps': False
}

# load the model to the global
def loadModel(model_path):
    global model
    model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)

def mfCosDistDict(inp):
    ''' (str) -> dict
    Returns a dictionary containing the inputted text and the value weights
    '''
    reps = dictionaries['reps']
    concepts = dictionaries['concepts']

    rar = {'text': inp}
    if isinstance(inp, float):
        for key in concepts:
            rar[key] = np.nan
        return rar
    
    words = inp.split()
    sent = ' '.join(words)
    sent = str(sent)
    sent = gensim.parsing.preprocessing.remove_stopwords(sent)
    
    translator = str.maketrans('', '', string.punctuation)
    sent = sent.lower().translate(translator)
    sentAr = sent.split(' ')

    srep = rep(sentAr)
    for key in sorted(reps):
        dist = cosdist(srep, reps[key])
        rar[key] = dist
    return rar

def cosdist(x, y):
    ''' (list, list) -> float
    Computes the cosine distance between two lists of values
    '''
    if x is None or y is None:
        return None
    return 1-spatial.distance.cosine(x, y)
    
def rep(wordAr):
    ''' (list) -> list
    Returns the representation of the phrase
    '''
    # start_time = time.time()
    
    try:
        model[wordAr]
    except:
        # I don't think this should be set(wordAr), because it's artificially 
        # reducing the number of words (by exluding duplicates)
        # I need to use set, because lists don't have intersection
        # but then I need to repopulate wordAr to include all words
        # wordAr = set(wordAr).intersection(allWordsInVocab)
        s = set(wordAr).intersection(model.vocab.keys())
        wordAr = [x for x in wordAr if x in s]
        
        # print wordAr
    if len(wordAr) == 0:
        return None
    numerator = np.sum(model[wordAr], axis=0)
    denominator = np.linalg.norm(numerator, axis=0)
    # print("--- %s seconds ---" % (time.time() - start_time))
    return numerator/denominator

def setDictionaries(concepts):    
    dictionaries['concepts'] = concepts
    # generate a representation for each word in the concept dictionary
    dictionaries['reps'] = {}
    for key in dictionaries['concepts']:
        dictionaries['reps'][key] = rep(dictionaries['concepts'][key])

def computeWeights(tar, logInterval=False):
    ''' (list, boolean) -> list
    Takes in a list of phrases and outputs their similarity to the set
    dictionary concepts.
    '''
    global model
    if model == False:
        print("Error: No model loaded")
        return
    if dictionaries['reps'] == False:
        print("Error: Concept dictionaries not set")
        return
    
    oar = [] # initialize output array
    for phrase in tar:
        if logInterval and ind%logInterval == 0:
            print("{}/{}".format((ind+1),len(tar)))
        phraseDict = mfCosDistDict(phrase)
        oar.append(phraseDict)
    return oar

def plotWeights(w):
    a = w.copy()
    del a['text']
    plt.figure()
    x = a.keys()
    y = a.values()
    plt.xticks(rotation=90)
    plt.axis([None, None, 0, 1])
    plt.bar(x,y)