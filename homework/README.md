# Homework 2: k-NN Classifier


```python
import pandas as pd
import numpy as np
```


```python
#define column names
names = ['a', 'b', 'c', 'd', 'class']

# Read CSV file
df = pd.read_csv('iris.data.txt', header=None, names=names)
```

## k-NN Algorithm

The algorithm is implemented with vectorization in mind.

1. Square of difference matrix $${S} = {X} - \begin{bmatrix} X_{t} \\ X_{t} \\ \vdots \\ X_{t} \\\end{bmatrix}$$
   Each element is then squared.

2. Euclidean distance is the sum of all columns of ${S}$.
   $${D} = {S}\begin{bmatrix} 1 \\ 1 \\ \vdots \\ 1 \\\end{bmatrix}$$

3. Selection is implemented by doing k partition on ${D}$.

4. The indices of the selected ones is stored in `select`, which is passed to `y` to access their corresponding classes.
   
   `np.unique` passes each element and the number of times it was counted to `lablel` and `freq`.
   
   Return the element who has the most number of times counted.


```python
def knn_classifier(df, Xt, k):
    '''Given dataset (X, y), classifies object Xt using k-NN algorithm.'''
    # Split up features and labels
    X = np.float32(df.iloc[:, 0:4])
    y = np.array(df['class'])
    
    # Convert test example into float numpy array
    Xt =  np.float32(Xt.iloc[0:4]).reshape((1, -1))

    m, n = X.shape

    # Square of difference for each element
    sqdiff = (X - np.ones((m, 1)) @ Xt) ** 2

    # Euclidean distance for every row
    dist = np.sqrt(sqdiff @ np.ones((n, 1)))

    # Selection of k smallest distance
    select = np.argpartition(dist, k, axis=0)[:k]

    # Find the dominant label among the selection
    label, freq = np.unique(y[select], return_counts=True)

    return label[freq.argmax()]
```

## Problem 1

I took an example from the dataset.


```python
# Test data: label='Iris-virginica'
Xt = pd.DataFrame(np.array([
    [6.7,3.0,5.2,2.3]
]))

# Perform k-NN Classification
knn_classifier(df, Xt, 1)
```




    'Iris-virginica'



## Problem 2

1. Split the DataFrame into 2 sets: one contains 60% of each class example, the other contains the rest.

2. The prediction function is a lambda function that calls `knn_classifier` and compares the result with the answer.

3. `test_set.apply` applies the function to each row with vectorization.

4. `np.count_nonzero` counts number of successful predictions in the data.


```python
# Split DataFrame into training set and testing set
train_set = pd.concat([
    df.iloc[0:30, :], 
    df.iloc[50:80, :], 
    df.iloc[100:130, :]
])

test_set = pd.concat([
    df.iloc[30:50, :],
    df.iloc[80:100, :],
    df.iloc[130:150, :]
])

# Prediction function
f = lambda Xt: knn_classifier(train_set, Xt, 1) == Xt.iloc[-1]

# Apply function to each row of testing set 
pred = test_set.apply(pred, axis=1)

# Count number of successful predictions
stat = np.count_nonzero(pred)

print(f'Success: {stat}\nFailure: {test_set.shape[0] - stat}\nAccuracy: {stat / test_set.shape[0]:.3f}')
```

    Success: 58
    Failure: 2
    Accuracy: 0.967

