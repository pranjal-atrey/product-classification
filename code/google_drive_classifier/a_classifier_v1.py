import csv
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import random as rand

names = []
testnames = []
testcats = []
cats = []

# Read in the Amazon data
with open('parsed_data/first_cat.csv', encoding='utf-8') as file:
    reader = csv.reader(file)

    reader.__next__()

    for line in reader:
        # Assures that the item will have a category
        if not line[3]:
            continue
        # Randomly picks items to be part of the test data
        t = rand.randint(0,200)
        if t != 24:
            names.append(line[0])
            cats.append(line[3])   
        else:
            testnames.append(line[0])
            testcats.append(line[3])

count_vect = CountVectorizer()
vectorized = count_vect.fit_transform(names)

# Fit estimator to the data and transform count-matrix to tf-idf representation
tf_transformer = TfidfTransformer()
train_tf = tf_transformer.fit_transform(vectorized)

# Train the classifier
classifier = MultinomialNB().fit(vectorized, cats)

# Fit the test data
newcounts = count_vect.transform(testnames)
new_tfidf = tf_transformer.transform(newcounts)
#Predict the test data's category
predicted = classifier.predict(new_tfidf)

# Display results
# for doc, category in zip(testnames, predicted):
#       print(doc, ':', category)

# Display classifier accuracy (may not be correct)
# print(str(int(np.mean(predicted == testcats)*100)) + '%')
