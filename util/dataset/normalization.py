import numpy as np
from numba import jit


@jit(nogil=True)
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
