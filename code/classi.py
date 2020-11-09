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
def train():

    # 1 = Naive Bayes
    # 2 = LR
    # 3 = TREE
    # 4 = SWM

    # grabs the user's input for the method they wish to use
    method = method_name.get()

    # sees if the user put a valid file name
    try:
        file = file_name.get()
    except:
        print("You entered an invalid file path. Please try again.")
        return

    test = test_name.get()
    if isinstance(test, int) and not (test > 0 and test < 99):
        print("You must enter an integer between 1% and 99%. Please try again.")

    # opens the dataset
    with open(file, encoding='utf-8') as f:
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
            rns = sorted(rand.sample(range(len(subcats)), int(len(subcats) * (test/100)) + 1), reverse=True)
            for r in rns:
                    testcats.append(subcats.pop(r))
                    testvalues.append(values.pop(r))
        # If test is a String, assumes it is a filename, reads the indices from that file, and pops those indices from the training arrays to the test arrays
        elif isinstance(test, str):
            with open(test, encoding='utf-8') as testfile:
                    testreader = csv.reader(testfile)
                    for line in testreader:
                        index = int(line[0])
                        testcats.append(subcats.pop(index))
                        testvalues.append(values.pop(index))

        print(f"{len(testcats)} data items ({test}%) withheld for testing...")

        # creates instance of Count Vectorizer
        count_vect = CountVectorizer()
        
        # This is where the bag of words is created:
        # Replace values with whatever you want to tokenize
        vectorized = count_vect.fit_transform(values)

        # Fit estimator to the data and transform count-matrix to tf-idf representation
        tf_transformer = TfidfTransformer()
        train_tf = tf_transformer.fit_transform(vectorized)

        # Train the classifier

        cw = {}

        if method == 1:
            
            # runs the Naive Bayes classifier and vectorizes the data
            classifier = MultinomialNB().fit(vectorized, cats)

            # Fit the test data
            newcounts = count_vect.transform(testvalues)
            new_tfidf = tf_transformer.transform(newcounts)
                
            #Predict the test data's category
            predicted = classifier.predict(new_tfidf)
                
            # Display classifier accuracy
            print("Accuracy:",str(int(np.mean(predicted == testcats)*100)) + '%')
            cw['testacc'] = int(np.mean(predicted == testcats)*100)

        elif method == 2:
            classifier = LogisticRegression(max_iter = 10000).fit(vectorized, cats)

            # Fit the test data
            newcounts = count_vect.transform(testvalues)
            new_tfidf = tf_transformer.transform(newcounts)
                
            #Predict the test data's category
            predicted = classifier.predict(new_tfidf)
                
            # Display classifier accuracy
            print(f"Accuracy: {str(int(np.mean(predicted == testcats)*100))}%")
            cw['testacc'] = int(np.mean(predicted == testcats)*100)

        elif method == 3:
            
            #split dataset in features and target variable
	    #feature_cols = ['category', 'subcategory', 'name', 'current_price', 'raw_price']
            X = vectorized #Features
            y = cats #Target variable

	    # Split dataset into training set and test set
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test, random_state=1) # 80% training and 20% test

	    # Create Decision Tree classifer object
            classifier = DecisionTreeClassifier()

	    # Train Decision Tree Classifer
            classifier = classifier.fit(X_train,y_train)

	    #Predict the response for test dataset
            y_pred = classifier.predict(X_test)

	    # Model Accuracy, how often is the classifier correct?
            print(f"Accuracy: {int(metrics.accuracy_score(y_test, y_pred) * 100)}%")
            cw['testacc'] = int(metrics.accuracy_score(y_test, y_pred) * 100)

        elif method == 4:

            # starts running the SVM classifier
            clf = svm.SVC(kernel = 'linear') # Linear Kernel
            cw['classifier'] = clf.fit(vectorized, cats)

            # Fit the test data
            newcounts = count_vect.transform(testvalues)
            new_tfidf = tf_transformer.transform(newcounts)
                
            #Predict the test data's category
            predicted = classifier.predict(new_tfidf)
                
            # Display classifier accuracy
            print(f"Accuracy: {str(int(np.mean(predicted == testcats)*100))}%")
            cw['testacc'] = int(np.mean(predicted == testcats)*100)

        # creates dictionary for easy transfer of variable names
        cw = {
                'classifier': classifier,
                'count_vect': count_vect,
                'tft': tf_transformer
            }

        # runs if the classifier succeeded in training but the user put an invalid string / no strong
        try:
            product = product_name.get()
        except:
            print("The classifier was able to run, but the string entered is invalid.")

        # attempts to classify input string into a category
        classify(product, cw, method)



def classify(product, classifier, method):
    # make the product a list as the methods below require it in list form
    if not isinstance(product, list):
        c = [product]
    else:
        c = product
    
    #Fit the String to be classified
    v = classifier['count_vect'].transform(c)
    n = classifier['tft'].transform(v)
    
    # Actual prediction step
    predicted = classifier['classifier'].predict(n)

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
cats = []
subcats = []
testcats = []
        
# Holds the values that will be tokenized for the bag of words
values = []
testvalues = []
        
# create the UI
# build a blank interface
interface = tk.Tk()
interface.geometry("420x550")

# create variables that will be used to grab data from UI
file_name = tk.StringVar()
product_name = tk.StringVar()
test_name = tk.IntVar()
method_name = tk.IntVar()

# create name at top
label_name = tk.Label(interface, text = "Classi.py, a Simple Classifier", font = ("Arial", 24)).grid(ipady = 5, row = 0, column = 1)
interface.title("Classi.py")

# create row for entering the file name
label_file = tk.Label(interface, text = "File Name", font = ("Arial", 16)).grid(pady = (10, 0), row = 1, column = 1)
entry_file = tk.Entry(interface, width = 30, textvariable = file_name).grid(ipady = 5, row = 2, column = 1)

# create row for entering a product name
label_product = tk.Label(interface, text = "Product to Classify", font = ("Arial", 16)).grid(pady = (10, 0), row = 4, column = 1)
entry_product = tk.Entry(interface, width = 30, textvariable = product_name).grid(ipady = 5, row = 5, column = 1)

# create row for entering the test percentage
label_test = tk.Label(interface, text = "Testing Percentage", font = ("Arial", 16)).grid(pady = (10, 0), row = 7, column = 1)
entry_test = tk.Entry(interface, width = 30, textvariable = test_name).grid(ipady = 5, row = 8, column = 1)

# create the radio button options for the classifier methods
label_method = tk.Label(interface, text = "Classifier Method", font = ("Arial", 16)).grid(pady = (25, 0), row = 10, column = 1)
radio_nb = tk.Radiobutton(interface, text = "Naive Bayes", variable = method_name, value = 1).grid(pady = (5, 0), row = 11, column = 1)
radio_lr = tk.Radiobutton(interface, text = "Logistic Regression", variable = method_name, value = 2).grid(pady = (5, 0), row = 12, column = 1)
radio_tree = tk.Radiobutton(interface, text = "Tree", variable = method_name, value = 3).grid(pady = (5, 0), row = 13, column = 1)
radio_swm = tk.Radiobutton(interface, text = "SWM", variable = method_name, value = 4).grid(pady = (5, 0), row = 14, column = 1)

# allow the user to submit the data to the program
button_submit = tk.Button(interface, text = "Submit", command = train, width = 15, height = 2).grid(pady = (50, 0), row = 15, column = 1)

# run the build infinitely until user closes out
interface.mainloop()
