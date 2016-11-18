__author__ = 'gregor'

from bs4 import BeautifulSoup
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
import os


def parseDocument(filePath):
    for filename in os.listdir(filePath):
        name, file_extension = os.path.splitext(filename)
        if filename.endswith(".sgm"):
            print("Extracting: " + name)
            with open("resources/" + filename, 'r') as myfile:
                data = myfile.read()
            soup = BeautifulSoup(data, 'html.parser')
            shortword = re.compile(r'([^ a-zA-Z]+)|(\b[a-zA-Z]{1,2}\b)')
            data = ["" for x in range(soup.find_all('body').__len__())]

            docs = soup.find_all('reuters')
            titles = soup.find_all('title')
            bodies = soup.find_all('body')
            j = 0
            for i in range(data.__len__()):
                if docs[i].find_all('body').__len__() != 0:
                    print(i)
                    data[i] = shortword.sub(' ', bodies[j].text) + ' ' + shortword.sub(' ', titles[i].text)
                    j += 1

        tf_vectorizer = CountVectorizer()

        tf = tf_vectorizer.fit_transform(data)
        save_sparse_csr("resources/csv/" + name, tf)
        return 0


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices, indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
