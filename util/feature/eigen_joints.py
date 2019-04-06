import os
import pickle

import numpy as np
from numba import jit
from scipy import linalg
from scipy.special import comb, perm
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


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
def extract_feature(joints):
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


@jit(nogil=True)
def defined_PCA(feature, n_dim=None):
    n, _ = feature.shape
    U, S, V = linalg.svd(feature, full_matrices=True)
    # U, V = svd_flip(U, V)

    V_max_ind = np.argmax(np.abs(V.T), axis=0)  # find max element each column
    V_sign = np.sign(V[np.arange(V_max_ind.size), V_max_ind])
    V *= V_sign[:, None]

    U *= S[None, :]
    U *= V_sign[None, :n]

    # scipy.io.savemat('V.mat', {'V': V})
    f_after_PCA = np.dot(feature, V.T)

    if n_dim is None:
        return f_after_PCA
    else:
        return f_after_PCA[:, :n_dim]
