'''
This is a demo script for testing our the job post detection system
'''
import json 
from pprint import pprint as pp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors.nearest_centroid import NearestCentroid # Ro classifier
from sklearn.neighbors import KNeighborsClassifier # KNN
import numpy as np

with open('corpus.json','r') as input:
    corpus = json.load(input)['data']

# Transform corpus to vector space model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(list(map(lambda e:e[0],corpus)))
Y = np.array(list(map(lambda e:e[1],corpus)))

def get_vectorizer_array(query):
    return vectorizer.transform([query]).toarray()

def do_testcase(classifier):
    # Test case 
    with open('testcase.json','r') as input:
        testdata = json.load(input)['data']
        # Fun fact, bool is the subclass of int, and 1 == true ;-)

        # Count how many it got correct
        correct = sum(str(classifier.predict(get_vectorizer_array(testcase[0]))[0]) == str(testcase[1]) for testcase in testdata)

        # Print the accurency in percentage
        result = str(correct / len(testdata) * 100) + "%"
        return result

def do_centroid():
    clf = NearestCentroid()
    clf.fit(X, Y)
    return do_testcase(clf)

def do_knn():
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(X, Y)
    return do_testcase(neigh)

# Finally, print the result
print("Rocchio classifier: " + do_centroid())
print("3-nearest neighbour: " + do_knn())