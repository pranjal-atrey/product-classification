import csv
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics, svm
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
                        testcats.append(cats.pop(index))
                        testvalues.append(values.pop(index))
                print(len(testcats), 'data items withheld for testing...')
            elif isinstance(test, int):
                # Stratification: Randomly selects a percentage of elements for training from *each category* (rather than a blanket percentage over the whole dataset)
                # Commented out code is for creating csv files with indices

                # with open('testgamut/testdata' + str(test) + '.csv', 'w+', encoding='utf-8', newline='') as testfile:
                    # writer = csv.writer(testfile)

                start = 0
                indices = []
                currentcat = cats[0]
                for i in range(len(cats)):
                    if cats[i] == currentcat:
                        True
                    else:
                        # Get the test data for the curent category and add it to the test data indices
                        for r in rand.sample(range(start, i), int((i-start) * (test/100))):
                            indices.append(r)
                        currentcat = cats[i+1]
                        start = i
                # Get the test percentage for the final category
                ran = rand.sample(range(start, i), int((i-start) * (test/100)))
                for r in ran:
                    indices.append(r)

                indices = sorted(indices, reverse=True)
                for i in indices:
                    # writer.writerow([i])
                    testcats.append(cats.pop(i))
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
            cw['classifier'] = MultinomialNB().fit(vectorized, cats)

        elif method == 'lr':
            cw['classifier'] = LogisticRegression(max_iter = 10000).fit(vectorized, cats)

            # Fit the test data
            newcounts = count_vect.transform(testvalues)
            new_tfidf = tf_transformer.transform(newcounts)
                
            #Predict the test data's category
            predicted = cw['classifier'].predict(new_tfidf)
                
            # Display classifier accuracy
            cw['testacc'] = int(np.mean(predicted == testcats)*100)

        elif method == 'tree':
            
            #split dataset in features and target variable
	        #feature_cols = ['category', 'subcategory', 'name', 'current_price', 'raw_price']
            X = vectorized #Features
            y = cats #Target variable

	        # Split dataset into training set and test set
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test, random_state=1) # 80% training and 20% test

	        # Create Decision Tree classifer object
            cw['classifier'] = DecisionTreeClassifier()

	        # Train Decision Tree Classifer
            cw['classifier'] = cw['classifier'].fit(X_train,y_train)

	        #Predict the response for test dataset
            y_pred = cw['classifier'].predict(X_test)

	        # Model Accuracy, how often is the classifier correct?
            cw['testacc'] = int(metrics.accuracy_score(y_test, y_pred) * 100)

        elif method == 'svm':

            # starts running the SVM classifier
            clf = svm.SVC(kernel = 'linear') # Linear Kernel
            cw['classifier'] = clf.fit(vectorized, cats)

            # Fit the test data
            newcounts = count_vect.transform(testvalues)
            new_tfidf = tf_transformer.transform(newcounts)
                
            #Predict the test data's category
            predicted = cw['classifier'].predict(new_tfidf)
                
            # Display classifier accuracy
            cw['testacc'] = int(np.mean(predicted == testcats)*100)
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
            predicted = cw['classifier'].predict(new_tfidf)
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
