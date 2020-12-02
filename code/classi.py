# classi.py - Team 4
# Member Contributions:
# Jordan Le: 25%
# Pranjal Atrey: 25%
# Alex Kisiel: 25%
# Erik O'Hara: 25%

# imports for our used libraries
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

# This function is used to get the top n words from our product names
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

# method to train a classifier on a data set
def train():

    # Holds whatever it is that the classifier will differentiate on
    cats = []
    testcats = []
    # Holds the values that will be tokenized for the bag of words
    values = []
    testvalues = []
    subcats = []

    # gets the file name from the UI
    try:
        file = file_name.get()
    except:
        print(f"File name {file_name.get()} not found. Try again.")
        return
    
    # gets the method selection from the UI
    try:
        method = int(method_name.get())
        if method == 1:
            method = "nb"
        elif method == 2:
            method = "lr"
        elif method == 3:
            method = "tree"
        elif method == 4:
            method = "svm"
    except:
        print("Please select a valid classifier method and try again.")
        return
    
    # gets the withheld testing data percentage from the UI
    try:
        test = int(test_name.get())
    except:
        print("Test value not loaded. Defaulting to 20%.")
        test = 20

    # Opens the CSV file to read every line
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
        train_on_this = cats.copy()


        # print("-------------------")
        # print("Amazon Dataset")
        # print("-------------------")
        # common_words = get_top_n_words(values, 10)
        # for word, freq in common_words:
        #     print(word, freq)

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
                    testcats.append(train_on_this.pop(i))
                    testvalues.append(values.pop(i))

                print(f"{len(testcats)} data items withheld for testing...")
        
        count_vect = CountVectorizer()
        # This is where the bag of words is created:
        # Replace values with whatever you want to tokenize
        vectorized = count_vect.fit_transform(values)

        # Fit estimator to the data and transform count-matrix to tf-idf representation
        tf_transformer = TfidfTransformer()
        train_tf = tf_transformer.fit_transform(vectorized)

        # selects Naive Bayes as classifier method
        if method == 'nb':
            # Train the classifier
            cw['classifier'] = MultinomialNB().fit(train_tf, train_on_this)

        # selects Logistic Regression as classifier method
        elif method == 'lr':
            cw['classifier'] = LogisticRegression(max_iter = 10000, class_weight='balanced').fit(train_tf, train_on_this)

        # selects Decision Tree as classifier method
        elif method == 'tree':
            # Create Decision Tree classifer object
            cw['classifier'] = DecisionTreeClassifier(class_weight='balanced')
            # Train Decision Tree Classifer
            cw['classifier'] = cw['classifier'].fit(train_tf, train_on_this)

        # selects SVM as classifier method
        elif method == 'svm':
            clf = svm.SVC(kernel = 'linear', class_weight='balanced') # Linear Kernel
            cw['classifier'] = clf.fit(train_tf, train_on_this)

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
            print(f"Success! Your classifier has a prediction accuracy of {acc}%. You can now classify items with this classifier using the panel on the right side of the user interface, or re-train a new classifier.")
            cw['acc'] = acc
    
    return cw


# classifier method, runs when user inputs product on righthand side
def classify():
    classifier = cw

    # gets product name from UI
    try:
        classifand = product_name.get()
        c = [classifand]
    except:
        print("You must enter a valid product name. Please try again.")
        return

    # adds color from UI if applicable
    try:
        c[0] = c[0] + " " + str(color_name.get())
    except:
        pass

    # adds price from UI if applicable
    try:
        c[0] = c[0] + " " + str(price_name.get())
    except:
        pass

    # Fit the String to be classified
    v = classifier['count_vect'].transform(c)
    n = classifier['tft'].transform(v)

    # Actual prediction step
    predicted = classifier['classifier'].predict(n)
    print("------------------------------")
    print("Prediction Complete!")
    print(f"Your product {classifand} belongs in the {predicted[0]} category!")
    return


# Holds whatever it is that the classifier will differentiate on
cw = {}
cats = []
subcats = []
testcats = []
        
# Holds the values that will be tokenized for the bag of words
values = []
testvalues = []
        
# create the UI
# build a blank interface
interface = tk.Tk()
interface.geometry("450x550")

# create variables that will be used to grab data from UI
file_name = tk.StringVar()
product_name = tk.StringVar()
test_name = tk.StringVar()
method_name = tk.IntVar()
color_name = tk.StringVar()
price_name = tk.StringVar()

# create name at top
label_name = tk.Label(interface, text = "Classi.py, a Simple Classifier", font = ("Arial", 24)).grid(ipady = 5, row = 0, column = 1, columnspan = 2)
interface.title("Classi.py")

# create two separate columns for different data entry
label_testing = tk.Label(interface, text = "Training Entry", font = ("Arial", 16)).grid(pady = (10, 0), padx = (5, 0), row = 1, column = 1)
label_data = tk.Label(interface, text = "Classify Entry", font = ("Arial", 16)).grid(pady = (10, 0), row = 1, column = 2)

# TESTING (LEFT) COLUMN DATA
# create row for entering the file name
label_file = tk.Label(interface, text = "CSV Data Set Path", font = ("Arial", 16)).grid(pady = (10, 0), padx = (15, 0), row = 3, column = 1)
entry_file = tk.Entry(interface, width = 30, textvariable = file_name).grid(ipady = 5, row = 4, column = 1)

# create row for entering the test percentage
label_test = tk.Label(interface, text = "Testing % Withheld", font = ("Arial", 16)).grid(pady = (10, 0), row = 5, column = 1)
entry_test = tk.Entry(interface, width = 30, textvariable = test_name).grid(ipady = 5, row = 6, column = 1)

# create the radio button options for the classifier methods
label_method = tk.Label(interface, text = "Classifier Method", font = ("Arial", 16)).grid(pady = (25, 0), row = 7, column = 1)
radio_nb = tk.Radiobutton(interface, text = "Naive Bayes", variable = method_name, value = 1).grid(pady = (5, 0), row = 8, column = 1)
radio_lr = tk.Radiobutton(interface, text = "Logistic Regression", variable = method_name, value = 2).grid(pady = (5, 0), row = 9, column = 1)
radio_tree = tk.Radiobutton(interface, text = "Tree", variable = method_name, value = 3).grid(pady = (5, 0), row = 10, column = 1)
radio_swm = tk.Radiobutton(interface, text = "SWM", variable = method_name, value = 4).grid(pady = (5, 0), row = 11, column = 1)

button_train = tk.Button(interface, text = "Train Classifier", command = train, width = 15, height = 2).grid(pady = (50, 0), row = 12, column = 1)

# CLASSIFYING (RIGHT) COLUMN DATA
# create row for entering a product name
label_product = tk.Label(interface, text = "Product Description", font = ("Arial", 16)).grid(pady = (10, 0), padx = 10, row = 3, column = 2)
entry_product = tk.Entry(interface, width = 30, textvariable = product_name).grid(ipady = 5, row = 4, column = 2)

label_color = tk.Label(interface, text = "Product Color", font = ("Arial", 16)).grid(pady = (10, 0), row = 5, column = 2)
entry_color = tk.Entry(interface, width = 30, textvariable = color_name).grid(ipady = 5, row = 6, column = 2)

label_price = tk.Label(interface, text = "Product Price", font = ("Arial", 16)).grid(pady = (10, 0), row = 7, column = 2)
entry_price = tk.Entry(interface, width = 30, textvariable = price_name).grid(ipady = 5, row = 8, column = 2)

# allow the user to submit the data to the program
button_classify = tk.Button(interface, text = "Classify Product", command = classify, width = 15, height = 2).grid(pady = (50, 0), row = 12, column = 2)


# run the build infinitely until user closes out
interface.mainloop()
