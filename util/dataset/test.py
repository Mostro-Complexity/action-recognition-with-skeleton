import numpy as np
from numba import jit
from sklearn.model_selection import train_test_split


@jit(nogil=True)
def split_dataset(y, x, ratio):
    """Divide dataset into training set and test set.  

    `y` : Array of labals. `shape`=`(N,)`  
    `x` : Array of samples. `shape`=`(N,)`  
    `ratio` : Proportion of test sets. `type`=`int`  
    `return` : (`y_test`, `x_test`, `y_train`, `x_train`) 
    """
    x_test, y_test = [], []
    x_train, y_train = [], []
    class_num = np.max(y) + 1

    for i in range(class_num):
        _x_train, _x_test = train_test_split(x[y == i, :, :], test_size=ratio)
        _y_train, _y_test = train_test_split(y[y == i], test_size=ratio)
        for j in range(_y_train.size):
            y_train.append(_y_train[j])
            x_train.append(_x_train[j, :, :])

        for j in range(_y_test.size):
            y_test.append(_y_test[j])
            x_test.append(_x_test[j, :, :])

    y_train, x_train = np.array(y_train), np.array(x_train)
    return np.array(y_test), np.array(x_test), y_train, x_train
