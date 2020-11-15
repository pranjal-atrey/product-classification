import csv
import random as rand
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as  plt
import classi as classipy
import os
import pickle

# Example to Store a Classifier

"""
    with open('classifiers/method_classifier_test.pkl', 'wb') as output:
    c = classipy.train('translated.csv', method, test)
    pickle.dump(c, output, pickle.HIGHEST_PROTOCOL)
"""


# Example to Make a Graph Using a Stored Classifier

x = []
y = []

plt.title('Accuracy By Classification Method')
plt.xlabel('% Data Withheld')
plt.ylabel('% Accuracy')

currentmethod = None

for file in os.listdir('classifiers/'):
    with open('classifiers/' + file, 'rb') as f:
        method = file.split('_')[0]
        test = int(file.split('classifiers')[1].split('.')[0])

        if method != currentmethod and currentmethod:
            s = UnivariateSpline(x,y)
            xs = np.linspace(1, x[len(x)-1], 100)
            ys = s(xs)
            plt.plot(xs, ys, label=currentmethod)
            x = []
            y = []

        currentmethod = method

        cw = pickle.load(f)
        x.append(test)
        y.append(cw['acc'])

s = UnivariateSpline(x,y)
xs = np.linspace(1, x[len(x)-1], 100)
ys = s(xs)
plt.plot(xs, ys, label=currentmethod)
plt.legend()
plt.show()
