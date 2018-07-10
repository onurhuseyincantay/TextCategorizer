import  nltk
import  re
import pandas as pd
from scipy.io import arff
import numpy as np
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import  glob
import  errno
import csv


csvFile = pd.read_csv("DuzenlenmisData.csv")
wpt = nltk.WordPunctTokenizer()
stop_word_list = nltk.corpus.stopwords.words('turkish')

def normalDoc(doc):
    doc = re.sub(" \d+"," ",doc)
    pattern = r"[{}]".format(",.;")
    doc = re.sub(pattern,"",doc)
    doc = doc.lower()
    doc = doc.strip()
    tokens = wpt.tokenize(doc)
    filteredTokens = [token for token in tokens if token not in stop_word_list]
    doc = ' '.join(filteredTokens)
    return doc
normalDocs = np.vectorize(normalDoc)
normalizedDocuments = normalDocs(csvFile["Description"])

bowVector = CountVectorizer(min_df=0.,max_df=1.)
bowMatrix = bowVector.fit_transform(normalizedDocuments)
#print(bowMatrix)
features = bowVector.get_feature_names()
# print("features[49228] :"+features[49228])
# print("features[37901] :"+features[37901])
bowMatrix = bowMatrix.toarray()
bowDf = pd.DataFrame(bowMatrix,columns=features)

tfIdDfVector = TfidfVectorizer(min_df=0.,max_df=1.,use_idf=True)
tfIdMatrix = tfIdDfVector.fit_transform(normalizedDocuments)
tfIdMatrix = tfIdMatrix.toarray()

features = tfIdDfVector.get_feature_names()
tfIdfId = pd.DataFrame(np.round(tfIdMatrix,3),columns=features)
numberOfTopics = 4
bowMatrix = bowVector.fit_transform(normalizedDocuments)
lDA = LatentDirichletAllocation(n_components=10,max_iter=10,learning_offset=50.,random_state=0,learning_method='online').fit(bowMatrix)
features = bowVector.get_feature_names()
for tId , topic in enumerate(lDA.components_):
    print("Topic %d:" % (tId))
    print(" ".join(features[i] for i in topic.argsort()[:-numberOfTopics - 1: -1]))


