# Data Set Division

This Notebook will show you to divide our data set to fit our model avoiding overfitting and underfitting. Also, in the final there will be a defined function to do all the division at once &#128522;

# Set up


```python
import pandas as pd
from sklearn.model_selection import train_test_split
```

# First division

First of all, you need to separate the data set in 3 differents subsets: 1º for the fit, 2º for validation and 3º for test predictions. Sklearn has **train_test_split** function which do a random partition of our data set every time the script run. So, if we need 3 different subsets from our data set, we will run this function two time, one for the subset **train_split** and the subset **test_split**, and other one for the subset **val_set**.

The **train_split**, like it's called, is for train our algorithm.

The **val_set** is to check that our algorithm doesn't have **overfitting** or **underfitting**.

The **test_set** is to check that our algorithm has a good accuracy for predict new samples.


```python
# Imagine we have loaded our data set as "df", a DataFrame made by Pandas from our data file in .csv format.

file_path = "...\folder\data_set.csv"
df = pd.read_csv(file_path)

# Now we separate our data set in 60% train set and 40% test set for example, it can be changed without problem
# but it's better that train set > test set for a good prediction.
train_set, train_set = train_test_split(df, test_size = 0.4, random_state = 42)
```


```python
# We can see information about our new subsets
train_set.info()
train_set.info()
```


```python
# Now we will separate our test subset in validation set and test set
val_set, test_set = train_test_split(test_set, test_size = 0.5, random_state= 42)
```


```python
print("Training set lenght:",len(train_set))
print("Validation set lenght:",len(val_set))
print("Test set lenght:",len(test_set))
```

# Random partition and Stratified Sampling

There is a parameter called **shuffle** in **train_test_split** function which can help to separate our data set if it is very big (Big Data reference &#128562;). **Shuffle** is used to avoid after many tries our algorithm "doesn't watch" all the data set by random mixing the data before the division.


```python
# If shuffle = False, the function doesn't mix the data set before the partition
# train_set, test_set = train_test_split(df, test_size=0.4, random_state=42, shuffle=False)
```

If the data set isn't big enough, it can appear the risk to introduce **sampling bias**. **Sampling bias** is a type of bias that occurs when a sample selected for a study or analysis does not adequately represent the entire population. This leads to incorrect conclusions, as certain groups within the population may be overrepresented or underrepresented. 

To avoid that, sklearn introduces the **stratify** parameter which makes a split so that the proportion of values in the sample produced will be the same as the proportion of values provided to parameter stratify.

For example, if variable "y" is a binary categorical variable with values 0 and 1 and there are 25% of zeros and 75% of ones, stratify = y will make sure that your random split has 25% of 0's and 75% of 1's.


```python
# You need to change "column" for the feature which you want to hold the initial proportion
# train_set, test_set = train_test_split(df, test_size=0.4, random_state=42, stratify=df["column"])
```

# Defined Function for complete division


```python
def train_val_test_split(df, rstate = 42, shuffle = True, stratify = None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size = 0.4, random_state = rstate, shuffle = shuffle, stratify = strat)
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size = 0.5, random_state = rstate, shuffle = shuffle, stratifu = strat)
    return (train_set, val_set, test_set)
```

To check all of this work great, we can see the lenght of our set and subsets. Also, we can check that **stratify** hold the initial proportion of the feature chosen.


```python
# Change "column" for the feature to hold the initial proportion.
train_set, val_set, test_set = train_val_test_split(df, stratify='column')
print("Initial Data Set lenght:", len(df))
print("Training Set lenght:", len(train_set))
print("Validation Set lenght:", len(val_set))
print("Test Set lenght:", len(test_set))

import matplotlib.pyplot as plt
%matplotlib inline
plt.figure(figsize=(10,10)
df["column"].hist()
train_set["column"].hist()
val_set["column"].hist()
test_set["column"].hist()
plt.show()
```
