import os
import pickle

import numpy as np
from scipy.special import comb, perm
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from numba import jit


@jit(nogil=True)
def normalization(f_cc, f_cp, f_ci):
    """Normalize f_cc, f_cp, f_ci to [-1,1]

    `f_cc` : f_cc feature of N samples. `shape`=`(N,)`  
    `f_cp` : f_cp feature of N samples. `shape`=`(N,)`  
    `f_ci` : f_ci feature of N samples. `shape`=`(N,)`  
    `return` : `f_cc`, `f_cp`, `f_ci` after normalization. `shape`=`(N,)`
    """

    scaler = MinMaxScaler(feature_range=(-1, 1))

    f_cc_std = scaler.fit_transform(f_cc)
    f_cp_std = scaler.fit_transform(f_cp)
    f_ci_std = scaler.fit_transform(f_ci)
    return f_cc_std, f_cp_std, f_ci_std


@jit(nogil=True)
def extract_feature(joints):  # TODO:使用ctypes加速
    """Extract feature from single sample.  
    N is the number of sample.  

    `joints` : Time sequence of skeleton. shape=(T, joint_num, coordinate_dim)  

    `return` : `f_cc`, `f_cp`, `f_ci`. shape=(N,)  
    """
    # a = np.random.random(100)
    # a_ctypes_ptr = cast(a.ctypes.data, POINTER(c_double))
    # np.ctypeslib.as_array(a_ctypes_ptr, shape=(100,))

    # np.ctypeslib.as_array(
    #     (ctypes.c_double * 100).from_address(ctypes.addressof(a_ctypes_ptr.contents)))

    frame_num, joint_num, _ = joints.shape

    f_cc = np.zeros((frame_num, int(comb(joint_num, 2)), 3), dtype=np.float32)
    f_cp = np.zeros((frame_num, joint_num * joint_num, 3), dtype=np.float32)
    f_ci = np.zeros((frame_num, joint_num * joint_num, 3), dtype=np.float32)

    k = 0
    for i in range(joint_num):
        for j in range(i + 1, joint_num):
            f_cc[:, k, :] = joints[:, i, :] - joints[:, j, :]
            k += 1

    for k in range(1, frame_num):
        for i in range(joint_num):
            for j in range(joint_num):
                f_cp[k, i * 15 + j, :] = joints[k, i, :] - \
                    joints[k - 1, j, :]

    for k in range(1, frame_num):
        for i in range(joint_num):
            for j in range(joint_num):
                f_ci[k, i * 15 + j, :] = joints[k, i, :] - \
                    joints[0, j, :]

    return f_cc, f_cp, f_ci


def scatter_samples(x, sample_size, step):
    """Scatter the big sample to smaller one.

    N is the number of sample.  
    `x` : Array of original samples. `shape`=`(N,)`  
    `sample_size` : New sample size. `type`=`int`  
    `step` : How many steps to sampling. `type`=`int`  
    `return` : labels and new samples. `shape`=`(M,)`
    """
    labels, samples = [], []
    orig_labels, orig_samples = x['label'], x['sample']

    for i in range(len(orig_labels)):
        frame_num = orig_samples[i].shape[0]

        for j in range(0, frame_num, step):
            if j + sample_size > frame_num:
                break
            labels.append(orig_labels[i])
            samples.append(orig_samples[i][j:j + sample_size, :])
    return np.array(labels), np.array(samples)


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
