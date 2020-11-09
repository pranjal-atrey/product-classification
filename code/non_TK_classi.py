import csv
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
import random as rand

def concat_files(files):
    return

def train(f, method, *args, **kwargs):

    # Holds whatever it is that the classifier will differentiate on
    cats = []
    testcats = []
    # Holds the values that will be tokenized for the bag of words
    values = []
    testvalues = []
    subcats = []

    if isinstance(f, list):
        file = concat_files(f)
    else:
        file = f
    test = kwargs.get('test', None)

    cw = {}

    with open(file, encoding='utf-8') as f:
        reader = csv.reader(f)
        header = reader.__next__()

        # Currently just pulls two elements (i.e. the large categories and the product names)
        for line in reader:
            cats.append(line[0])
            subcats.append(line[1])
            values.append(line[2])

        if test:
            # If a String is passed to test it's assumed that it is a filename containing a number of indices
            if isinstance(test, str):
                indices = []
                with open(test, encoding='utf-8') as testfile:
                    testreader = csv.reader(testfile)
                    for line in testreader:
                        index = int(line[0])
                        testcats.append(subcats.pop(index))
                        testvalues.append(values.pop(index))
                print(len(testcats), 'data items withheld for testing...')
            else:
                # Select the random percentage of test data, order it in reverse order, and append/pop accordingly
                rns = sorted(rand.sample(range(len(subcats)), int(len(subcats) * (test/100)) + 1), reverse=True)
                for r in rns:
                    testcats.append(subcats.pop(r))
                    testvalues.append(values.pop(r))
                print(len(testcats),'data items withheld for testing...')
    
        count_vect = CountVectorizer()
        # This is where the bag of words is created:
        # Replace values with whatever you want to tokenize
        vectorized = count_vect.fit_transform(values)

        # Fit estimator to the data and transform count-matrix to tf-idf representation
        tf_transformer = TfidfTransformer()
        train_tf = tf_transformer.fit_transform(vectorized)
        # Train the classifier
        classifier = MultinomialNB().fit(vectorized, subcats)

        cw = {
            'classifier': classifier,
            'count_vect': count_vect,
            'tft': tf_transformer,
            'acc': None
        }

        if test:
            # Fit the test data
            newcounts = count_vect.transform(testvalues)
            new_tfidf = tf_transformer.transform(newcounts)
            #Predict the test data's category
            predicted = classifier.predict(new_tfidf)
            # Display classifier accuracy
            acc = int(np.mean(predicted == testcats)*100)
            cw['acc'] = acc

            # Writes results to a file in case you want to see what the actual predictions look like
            # with open('testfile.csv', 'w+', encoding='utf-8', newline='') as f:
            #     writer = csv.writer(f)
            #     for doc, category in zip(testvalues, predicted):
            #         writer.writerow([doc, ':', category])
    

    return cw

def classify(classifand, classifier):
    if not isinstance(classifand, list):
        c = [classifand]
    else:
        c = classifand
    #Fit the String to be classified
    v = classifier['count_vect'].transform(c)
    n = classifier['tft'].transform(v)
    # Actual prediction step
    predicted = classifier['classifier'].predict(n)
    i = 0
    for p in predicted:
        print(classifand[i], ':', p)
        i += 1
