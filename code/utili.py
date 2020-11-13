import csv
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as  plt
import classi as classipy

plt.title('Accuracy Over % Testing Data Withheld After Stratification')
plt.xlabel('% Testing Data Withheld')
plt.ylabel('% Accuracy')

methods = ['nb', 'lr', 'tree', 'svm']

for m in methods:
    x = []
    y = []
    for i in range(1,100):
        if not i%10 == 0:
            continue
        filename = 'testgamut/testdata' + str(i) + '.csv'
        acc = classipy.train('translated.csv', m, test=filename)['acc']
        x.append(i)
        y.append(acc)

    s = UnivariateSpline(x,y)
    xs = np.linspace(1, x[len(x)-1], 100)
    ys = s(xs)
    plt.plot(xs, ys, label=m)

plt.legend()
plt.show()
