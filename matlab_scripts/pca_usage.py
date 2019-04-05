import numpy as np
import scipy.io
from scipy import linalg


def defined_PCA(feature, n_dim=None):
    U, S, V = linalg.svd(feature, full_matrices=True)
    # U, V = svd_flip(U, V)

    V_max_ind = np.argmax(np.abs(V.T), axis=0)  # find max element each column
    V_sign = np.sign(V[np.arange(V_max_ind.size), V_max_ind])
    V *= V_sign[:, None]

    U *= S[None, :]
    U *= V_sign[None, :30]

    # scipy.io.savemat('V.mat', {'V': V})
    f_after_PCA = np.dot(feature, V.T)

    if n_dim is None:
        return f_after_PCA
    else:
        return f_after_PCA[:n_dim]


if __name__ == "__main__":
    feature = scipy.io.loadmat('feature.mat')['feature']
    coeff = scipy.io.loadmat('coeff.mat')['coeff']

    r = defined_PCA(feature)
    print(r.shape)