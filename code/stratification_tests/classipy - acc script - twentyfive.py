import csv
import numpy as np
import tkinter as tk
import random as rand
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics, svm


# method to train a classifier on a data set
def train(file, test, method):

    global cw
    cw = {}

    # 1 = Naive Bayes
    # 2 = LR
    # 3 = TREE
    # 4 = SWM

    # opens the dataset
    f = open(file, 'r', encoding='utf-8')
    # cleans out arrays for every iteration
    cats.clear()
    subcats.clear()
    testcats.clear()
    values.clear()
    testvalues.clear()

    # reads data set and its header
    reader = csv.reader(f)
    header = reader.__next__()

    # Currently just pulls two elements (i.e. the large categories and the product names)
    for line in reader:
        cats.append(line[0])
        subcats.append(line[1])
        values.append(line[2])
    
    # If test is an int, randomly takes that % of the total data and withholds it for testing
    if isinstance(test, int):
	# Begin stratification implementation
        start = 0
        indices = []
        currentcat = cats[0]
        for i in range(len(cats)):
            if cats[i] == currentcat:
                True
            else:
                # Randomly select test data for the curent category and add it to the test data indices
                for r in rand.sample(range(start, i), int((i-start) * (test/100))):
                    indices.append(r)
                currentcat = cats[i+1]
                start = i
        # Get the test percentage for the final category
        ran = rand.sample(range(start, i), int((i-start) * (test/100)))
        for r in ran:
            indices.append(r)
	
	# Sort the indices from greatest to smallest so popping doesn't affect the overall order
        indices = sorted(indices, reverse=True)
        for i in indices:
            # writer.writerow([i])
            testcats.append(subcats.pop(i))		# If classifying on the larger category remember to change back to 'cats'
            testvalues.append(values.pop(i))
	
        print(f"{len(testcats)} data items ({test}%) withheld for testing...")
    # If test is a String, assumes it is a filename, reads the indices from that file, and pops those indices from the training arrays to the test arrays
    elif isinstance(test, str):
        with open(test, encoding='utf-8') as testfile:
                testreader = csv.reader(testfile)
                for line in testreader:
                    index = int(line[0])
                    testcats.append(subcats.pop(index))
                    testvalues.append(values.pop(index))
        print(f"{len(testcats)} data items withheld from {test} for testing...")

    # creates instance of Count Vectorizer
    count_vect = CountVectorizer()
    
    # This is where the bag of words is created:
    # Replace values with whatever you want to tokenize
    vectorized = count_vect.fit_transform(values)

    # Fit estimator to the data and transform count-matrix to tf-idf representation
    tf_transformer = TfidfTransformer()
    train_tf = tf_transformer.fit_transform(vectorized)

    # Train the classifier
    if method == 1:
        
        print("Starting training with a \"Naive Bayes\" method...")
        
        # runs the Naive Bayes classifier and vectorizes the data
        classifier = MultinomialNB().fit(vectorized, subcats)

        # Fit the test data
        newcounts = count_vect.transform(testvalues)
        new_tfidf = tf_transformer.transform(newcounts)
            
        #Predict the test data's category
        predicted = classifier.predict(new_tfidf)
            
        # Display classifier accuracy
        cw['testacc'] = int(np.mean(predicted == testcats)*100)
        print(f"Training Complete!\nThis dataset will have a {cw['testacc']}% accuracy in classifying products!")
        print("-----------------------------")
        

    elif method == 2:

        print("Starting training with a \"Logistic Regression\" method...")
        
        classifier = LogisticRegression(max_iter = 10000).fit(vectorized, subcats)

        # Fit the test data
        newcounts = count_vect.transform(testvalues)
        new_tfidf = tf_transformer.transform(newcounts)
            
        #Predict the test data's category
        predicted = classifier.predict(new_tfidf)
            
        # Display classifier accuracy
        cw['testacc'] = int(np.mean(predicted == testcats)*100)
        print(f"Training Complete!\nThis dataset will have a {cw['testacc']}% accuracy in classifying products!")
        print("-----------------------------")
        

    elif method == 3:

        print("Starting training with a \"Tree\" method...")
        
        #split dataset in features and target variable
        #feature_cols = ['category', 'subcategory', 'name', 'current_price', 'raw_price']
        X = vectorized #Features
        y = subcats #Target variable

        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test, random_state=1) # 80% training and 20% test

        # Create Decision Tree classifer object
        classifier = DecisionTreeClassifier()

        # Train Decision Tree Classifer
        classifier = classifier.fit(X_train,y_train)

        #Predict the response for test dataset
        y_pred = classifier.predict(X_test)

        # Model Accuracy, how often is the classifier correct?
        cw['testacc'] = int(metrics.accuracy_score(y_test, y_pred) * 100)
        print(f"Training Complete!\nThis dataset will have a {cw['testacc']}% accuracy in classifying products!")
        print("-----------------------------")

    elif method == 4:

        print("Starting training with a \"SVM\" method...")

        # starts running the SVM classifier
        clf = svm.SVC(kernel = 'linear') # Linear Kernel
        classifier = clf.fit(vectorized, subcats)

        # Fit the test data
        newcounts = count_vect.transform(testvalues)
        new_tfidf = tf_transformer.transform(newcounts)
            
        #Predict the test data's category
        predicted = classifier.predict(new_tfidf)
		
        # Setting accuracy
        cw['testacc'] = int(np.mean(predicted == testcats)*100)
            
        # Display classifier accuracy
        print(f"Training Complete!\nThis dataset will have a {cw['testacc']}% accuracy in classifying products!")
        print("-----------------------------")

    with open("25.txt", 'a') as file2:
        file2.write(f"{cw['testacc']}, ")


def classify():
    # grab variables from the input boxes
    product = product_name.get()
    method = method_name.get()

    # make the product a list as the methods below require it in list form
    if not isinstance(product, list):
        c = [product]
    else:
        c = product
    
    #Fit the String to be classified
    v = cw['count_vect'].transform(c)
    n = cw['tft'].transform(v)
    
    # Actual prediction step
    predicted = cw['classifier'].predict(n)

    # simply sets the method name for easy output
    if method == 1:
        method = "Naive Bayes"
    elif method == 2:
        method = "Logistic Regression"
    elif method == 3:
        method = "Tree"
    elif method == 4:
        method = "SVM"

    # prints a resulting category for every product entered
    for p in predicted:
        print(f"Using the '{method}' method, we believe that the item '{product}' would belong in the '{p}' category!")
        print("-----------------------------")


# Holds whatever it is that the classifier will differentiate on
cw = {}
cats = []
subcats = []
testcats = []
        
# Holds the values that will be tokenized for the bag of words
values = []
testvalues = []

#datasets[] is the name of the datasets, n = number of times to iterate each dataset
datasets = ['combined.csv', 'walmart_reformed.csv', 'amazon_reformed.csv']
n = 5
test_percent = 25

for dataset in datasets:                                # for every dataset,
    for method in range(1, 5):                          # use each method once
        for i in range(0, n):                           # n times over
            train(dataset, test_percent, method)        # and train it
            print(f"dataset {dataset}, method {method}, iteration {i + 1} out of {n}")
		
