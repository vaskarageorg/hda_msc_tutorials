'''
Tutorial 1: Introduction to Machine Learning with Python

The goal of this tutorial is to introduce a typical workflow in carrying out ML in Python. This includes,
1. accessing and organising data,
2. assessing the data,
3. visualising the data,
4. a) creating training, b) test datasets and c) learning a model using them and evaluating its performance.
'''

import numpy as np
import pandas

from matplotlib import pyplot as plt

# Set the default figure size.
plt.rcParams["figure.figsize"] = (10, 8)



'''
1) Load Data
Here, we shall load the Iris dataset from a publically available source.
This dataset consists of 150 samples of 3 classes of iris plants; each datapoint consists of 4 attributes, the "sepal-length", "sepal-width", "petal-length" and"petal-width".
Once the data has been downloaded, we can organise them into their classes.
'''

url = "./iris.csv"
column_names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
class_names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

raw_dataset = pandas.read_csv(url, names=column_names)

# print the raw dataset
print(raw_dataset)

# Organise data by class
dataset = raw_dataset.groupby('class')



'''
2) Statistics of the dataset
Pandas has some convenience methods that allow us to easily calculate statistical properties of a dataset.
'''

# Calculate the mean of each attribute. E.g
print(f"\nMean: \n{dataset.mean()}")

# a) Calculate the standard deviation of each attribute


# b) Show the minimum of each attribute


# c) Show the maximum of each attribute




'''
3) Visualise the dataset
Pandas has some convenience functions that allow us to easily visualise our dataset.
This is the documentation for the basic plotting tools available in Pandas: https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html
'''

# Try some of them here.
# For example, try to plot scatter graphs for the Iris-setosa class:




'''
4) Classification using Least Squares
Here we will be carrying out classification using the least squares formulation on 2 classes of the dataset.
'''

# a) Create separate datasets for the classes 'Iris-setosa' and 'Iris-versicolor'.
setosa = dataset.get_group("Iris-setosa")
versicolor = dataset.get_group("Iris-versicolor")

# b) create an output vector $Y^k$, for each class, where $y_i^k = 1$ if $k = $'Iris-setosa' and $-1$ otherwise.
# Insert code here to update 'setosa' and 'versicolor' DataFrames to include an extra column 'output'.



assert setosa.shape == (50, 5)
assert versicolor.shape == (50, 5)

# c) create training and test datasets, with 20% of the data for testing (80 training points and 20 testing points).
# Make sure that data from each class is equally distributed.
# Create 'training_data' and 'test_data' DataFrames that contain the appropriate number of samples from each class.



assert training_data.shape == (80, 5)
assert test_data.shape == (20, 5)

# d) apply the least squares solution to obtain an optimal solution for different combinations of the 4 available attributes.
# Create all possible combinations of attributes.
from itertools import chain, combinations

def all_combinations(attributes):
    return chain(*map(lambda i: combinations(attributes, i), range(1, len(attributes) + 1)))


_attributes = [name for name in column_names if name != 'class']
attribute_combinations = all_combinations(_attributes)  # Note that this is an iterable object.


# # Complete the function that takes in a list of attributes, and outputs the predictions after carrying out least squares.
def return_predictions(attributes, training_data=training_data, testing_data=test_data):


    return predictions

# e) evaluate which input attributes are the best.
# Complete the function below that takes in a predictions vector, and outputs the mean squared error.
def return_mse(predictions, testing_data=test_data):


    return mse

# evaluate
for attributes in attribute_combinations:
    preds = return_predictions(list(attributes))
    print(f"{str(attributes):<70} MSE: {return_mse(preds)}")

plt.show()
