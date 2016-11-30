__author__ = 'gregor'

from bs4 import BeautifulSoup
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import os
import operator


def parseDocument(filePath):
    data = []
    for filename in os.listdir(filePath):
        name, file_extension = os.path.splitext(filename)
        if filename.endswith(".sgm"):
            print("Extracting: " + name)
            with open("resources/" + filename, 'r') as myfile:
                data.append(myfile.read())

    vrti(data[0] + ' ' + data[1] + ' ' + data[2] + ' ' + data[3] + ' ' + data[4])



def vrti(data):
    soup = BeautifulSoup(data, 'html.parser')
    shortword = re.compile(r'([^ a-zA-Z]+)|(\b[a-zA-Z]{1,2}\b)')
    data = []

    docs = soup.find_all('reuters')
    titles = soup.find_all('title')
    bodies = soup.find_all('body')
    topics = soup.find_all('topics')
    topicDict = {}
    r = topics.__len__()
    for i in range(r):
        if topics[i].contents.__len__() == 1:
            txt = topics[i].contents[0].text.strip()
            if not topicDict.__contains__(txt):
                topicDict.__setitem__(txt, 1)
            else:
                times = topicDict.__getitem__(txt)
                topicDict.__setitem__(txt, times + 1)

    sorted_x = sorted(topicDict.items(), key=operator.itemgetter(1))
    topicSet = set()
    topicToNum = {}
    k = 1

    while topicSet.__len__() < 20:
        topicSet.add(sorted_x[sorted_x.__len__() - k][0])
        topicToNum.__setitem__(sorted_x[sorted_x.__len__() - k][0], k)
        k += 1

    j = 0
    klaster = []
    for i in range(0, docs.__len__()):
        try:
            body = docs[i].find_all('body')
            if body.__len__() != 0 \
                    and topics[i].contents.__len__() == 1 \
                    and topicSet.__contains__(topics[i].contents[0].text.strip()):
                body = shortword.sub(' ', bodies[j].text)
                j += 1
                if body.split().__len__() > 10:
                    # print(body.split().__len__())
                    data.append(body + ' ' + shortword.sub(' ', titles[i].text))
                    klaster.append(topicToNum.__getitem__(topics[i].contents[0].text.strip()))
        except:
            continue

    tf_vectorizer = CountVectorizer()  # fit_transform(raw_documents, y=None)
    # Returns:
    #    X : sparse matrix, [n_samples, n_features]
    #    Tf-idf-weighted document-term matrix.
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tf = tfidf_vectorizer.fit_transform(data)
    # [n_samples, n_features]
    narray = tf.toarray().transpose()
    clasters = np.array(klaster)

    print(narray.shape)
    save_sparse_csr("resources/csv/matrica5", narray)
    save_sparse_csr("resources/csv/klasteri5", clasters)
    return 0


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data)
