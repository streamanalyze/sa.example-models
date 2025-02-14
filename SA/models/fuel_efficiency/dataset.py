import pandas as pd
import numpy as np

# Download and import dataset
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'

raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

dataset = raw_dataset.copy()

# Clean the dataset
dataset = dataset.dropna()
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')

# Split the dataset
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

# Features for linear regression
def get_horsepower_features():
    horsepower = np.array(train_features['Horsepower'])
    return horsepower

def get_all_features():
    # array created from panda dataframe is in Fortran order
    return np.array(train_features).copy(order='C').astype('float32')

def get_mpg_labels():
    return np.array(train_labels)

def get_horsepower_test_features():
    return np.array(test_features['Horsepower'])

def get_all_test_features():
    return np.array(test_features).copy(order='C').astype('float32')

def get_test_labels():
    return np.array(test_labels)