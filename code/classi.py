import csv
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics, svm
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import random as rand
import time

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
            # If the dataset has additional features, adds them to values
            try:
                line[3]
                for n in range(3, len(line)):
                    values[len(values)-1] += ' ' + line[n]
            except Exception as e:
                True

        # Replace 'cats' here if you want to train on subcats
        train_on_this = subcats.copy()

        if test:
            # If a String is passed to test it's assumed that it is a filename containing a number of indices
            if isinstance(test, str):
                indices = []
                with open(test, encoding='utf-8') as testfile:
                    testreader = csv.reader(testfile)
                    for line in testreader:
                        index = int(line[0])
                        testcats.append(train_on_this.pop(index))
                        testvalues.append(values.pop(index))
                print(len(testcats), 'data items withheld for testing...')
            elif isinstance(test, int):
                # Stratification: Randomly selects a percentage of elements for training from *each category* (rather than a blanket percentage over the whole dataset)
                # Commented out code is for creating csv files with indices

                # with open('amz_testgamut/testdata' + str(test) + '.csv', 'w+', encoding='utf-8', newline='') as testfile:
                #     writer = csv.writer(testfile)

                start = 0
                indices = []
                currentcat = train_on_this[0]
                for i in range(len(train_on_this)):
                    if train_on_this[i] == currentcat:
                        True
                    else:
                        # Get the test data for the curent category and add it to the test data indices
                        for r in rand.sample(range(start, i), int((i-start) * (test/100))):
                            indices.append(r)
                        currentcat = train_on_this[i+1]
                        start = i
                # Get the test percentage for the final category
                ran = rand.sample(range(start, i), int((i-start) * (test/100)))
                for r in ran:
                    indices.append(r)

                indices = sorted(indices, reverse=True)
                for i in indices:
                    # writer.writerow([i])
                    testcats.append(train_on_this.pop(i))
                    testvalues.append(values.pop(i))

                    print(len(testcats),'data items withheld for testing...')
        
        count_vect = CountVectorizer()
        # This is where the bag of words is created:
        # Replace values with whatever you want to tokenize
        vectorized = count_vect.fit_transform(values)

        # Fit estimator to the data and transform count-matrix to tf-idf representation
        tf_transformer = TfidfTransformer()
        train_tf = tf_transformer.fit_transform(vectorized)

        if method == 'nb':
            # Train the classifier
            train_start = time.time()
            cw['classifier'] = MultinomialNB().fit(train_tf, train_on_this)
            cw['train_time'] = time.time() - train_start

        elif method == 'lr':
            train_start = time.time()
            cw['classifier'] = LogisticRegression(max_iter = 10000, class_weight='balanced').fit(train_tf, train_on_this)
            cw['train_time'] = time.time() - train_start
            conf_matrix(train_tf, train_on_this)


        elif method == 'tree':
	        # Create Decision Tree classifer object
            cw['classifier'] = DecisionTreeClassifier(class_weight='balanced')
	        # Train Decision Tree Classifer
            cw['classifier'] = cw['classifier'].fit(train_tf, train_on_this)

        elif method == 'svm':

            # starts running the SVM classifier
            clf = svm.SVC(kernel = 'linear', class_weight='balanced') # Linear Kernel
            train_start = time.time()
            cw['classifier'] = clf.fit(train_tf, train_on_this)
            cw['train_time'] = time.time() - train_start

        else:
            print('Must enter an acceptable method')
            return

        cw['count_vect'] = count_vect
        cw['tft'] = tf_transformer

        if test:
            # Fit the test data
            newcounts = count_vect.transform(testvalues)
            new_tfidf = tf_transformer.transform(newcounts)
            #Predict the test data's category
            pstart = time.time()
            predicted = cw['classifier'].predict(new_tfidf)
            cw['predict_time'] = time.time() - pstart
            # Set classifier accuracy
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
