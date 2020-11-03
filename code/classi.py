import csv
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import random as rand

# Holds whatever it is that the classifier will differentiate on
pridet = []
testpridet = []
# Holds the values that will be tokenized for the bag of words
values = []
testvalues = []

def concat_files(files):
    return

def train(f, method, *args, **kwargs):
    if isinstance(f, list):
        file = concat_files(f)
    else:
        file = f
    test = kwargs.get('test', None)

    with open(file, encoding='utf-8') as f:
        reader = csv.reader(f)
        header = reader.__next__()

        # Currently just pulls two elements (i.e. the large categories and the product names)
        for line in reader:
            pridet.append(line[0])
            values.append(line[2])
        # Select random test elements (if applicable)
        if test:
            for r in range(int(len(pridet) * (test/100))+1):
                randomNumber = rand.randint(0, len(pridet)-1)
                testpridet.append(pridet.pop(randomNumber))
                testvalues.append(values.pop(randomNumber))
            print(len(testpridet),'data items withheld for testing...')
    
        count_vect = CountVectorizer()
        # This is where the bag of words is created:
        # Replace values with whatever you want to tokenize
        vectorized = count_vect.fit_transform(values)

        # Fit estimator to the data and transform count-matrix to tf-idf representation
        tf_transformer = TfidfTransformer()
        train_tf = tf_transformer.fit_transform(vectorized)
        # Train the classifier
        classifier = MultinomialNB().fit(vectorized, pridet)

        cw = {
            'classifier': classifier,
            'count_vect': count_vect,
            'tft': tf_transformer
        }

        if test:
            # Fit the test data
            newcounts = count_vect.transform(testvalues)
            new_tfidf = tf_transformer.transform(newcounts)
            #Predict the test data's category
            predicted = classifier.predict(new_tfidf)
            # Display classifier accuracy
            print("Accuracy:",str(int(np.mean(predicted == testpridet)*100)) + '%')
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

# Example
c = train('translated.csv', 'nb', test=15)

