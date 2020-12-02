# Product Classification

## How Do I Implement Each Classifier?

Syntactically speaking, training a classifier on a file is as simple as passing it as an argument into train; train also accepts an array of files.
As an example, training a Naïve Bayes-based classifier on data in a file data_file would look like this: 

```python
train(data_file, "nb");
```

The *method* parameter dictates the type of classifier that will be trained on the data; see **Which Classifier Should I Use?** for help picking a method, as well as train’s documentation for more information.

Including the *test* parameter allows the user to set aside a certain percentage of their data as ‘test data.’ This test data will be randomly selected from the dataset and will not contribute to the classifier’s training. Once the classifier is trained on the non-test data, it will try to classify the test data based on what it has learned. The program will then compare the classifier’s guess with the actual value for each test element and print a numerical representation of its accuracy. Bear in mind that smaller test data pools may report lower-than-actual accuracy, and larger pools mean less actual training data for the classifier. What constitutes a good value depends on the overall data size, but generally ~10-15% should yield a fair representation of accuracy. 

Testing the accuracy of the classifier in the above example at 15% would look like this:

```python
train(data_file, ”nb”, test=15)
```

Once a classifier has been trained, classifying Strings is as simple as passing the Strings – either separately or as an array – along with the classifier object as arguments in *classify*.

Classifying a product ‘mint chocolate chip ice cream’ using the classifier above would look like this:

```python
classify(‘mint chocolate chip ice cream’, classifier)
```
## What Kinds of Data Files Are Acceptable?

Classifiers tend to perform better with larger pools of data; as they are exposed to more and more examples of correct classification, they become progressively better at making nuanced predictions themselves. That said, however, excessively large datasets will adversely affect performance (since the classifier has to expend time and resources to process all the data). A good dataset generally contains at least 5000 elements. 

As to the actual data content, classifiers generally perform best when the classes are distinct. If a classifier is asked to determine the category of a rubber duck, it’ll have more success when the options are “bath toy” and “shampoo” that it would if the options were “bath toy” and “shower toy” for example.

For best results, any files passed to train should adhere to the following guidelines:
- The file should be a .csv file in English, with the delimiter set as “,” and the encoding set as “utf-8”.
- The first line of each file should delineate the headers for the products contained therein. For example, a header line might look like this:
```python
category, name, color, price
```
- The first header should represent the desired primary determinant (i.e. whatever you’re trying to classify items into). Every product should have a meaningful, non-null – value for this attribute (otherwise the product can’t really contribute anything to the training process).
  - In most cases the primary determinant will be a product’s category: the classifier might be asked to place an item ‘lawn chair’ into one of a number of categories, for example.
- Every product in the file should have a value for every header. This means that even if a product doesn’t have a known value for an attribute that attribute should still be represented by an empty string or some other null value.
  - To follow the earlier example, a product without a known color would be listed as:
  ```python
  toys, “teddy bear”,, “$5”
  ```
- Attributes with units should have that unit included in their value.
  - i.e. “toys, ‘teddy bear’,, ‘$5’” rather than “toys, ‘teddy bear’,, 5” with the header being “price ($)”
  
When passing multiple files to train it’s important that the primary determinant for every file is the same; if the classifier is supposed to be recognizing/associating product categories for one dataset and prices for another it’s ultimately not likely to perform very well for either. 

## Method List

 - train(files, ”classifier_method_key”, test=%_test_data, svm_kernel=%_kernel_type, noise_len=%_str_len)
    - Parameters
      - files: The file(s) that the classifier will be trained on. Can be a single file or an array of files.
      - method: String: The specific classifier method that will be used. Potential values are as follows:
        - nb: Naïve Bayes
        - lr: Logistic Regression
        - tree: Tree Classifier
        - svm: SVM
      - test: 
        - int : Pulls aside a percentage of the data in files to use as test data. Prints a measure of accuracy for the trained classifier
        - String : Attempts to use the String as a file name, and pulls indices from that file that represent data elements to withhold for testing.
      - kernel_type: String: Defines the kernel type to be used for the SVM classification method.
      - noise_len: int: The length of a randomly-generated String that will be appended to the end of a product's name (used to test the classifier's robustness).
    - Returns
      - A dictionary containing classifier objects, trained on files using the specified method (also includes the accuracy and training time for the classifier if the 'test' variable was used).
  - classify(items, classifier=[classifier])
    - Parameters
      - items: The String(s) that the classifier will attempt to classify. Can be a single String or an array of Strings.
      - classifier:  A trained classifier (returned via train).
    - Returns
      - A String (or array of Strings if items was plural) representing the predicted value(s).


