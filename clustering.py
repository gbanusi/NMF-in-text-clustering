__author__ = 'gregor'

import numpy as np

def results(H, klasters):
    res = np.array([np.argmax(H[:, i]) for i in range(H.shape[1])])
    topics = np.array([np.zeros(20) for i in range(0, 20)])

    for i in range(0, klasters.__len__()):
        topics[res[i]][klasters[i]] += 1

    for i in range(0, 20):
        print('Klaster ' + i.__str__() + ' ima većinu topica: ' + np.argmax(topics[i, :]).__str__() + ', postotak = ' + (
            np.max(topics[i, :]) / np.sum(topics[i, :])).__str__() + '%.')
