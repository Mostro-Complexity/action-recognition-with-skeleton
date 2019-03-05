import argparse
import os
import pickle

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from util import *

ORIGINAL_DATA_PATH = 'data/original'


def integrate_features(X):
    """Convert all samples to features with single time.

    `X` : sample array, which contains N samples. shape=(N,)  
    `return` : feature array, which contains N features. shape=(N,) 
    """
    set_size = X.shape[0]
    features = []

    for i in range(set_size):  # 对每个训练（测试）样本提取特征
        example = X[i]
        frame_num, joint_num, _ = example.shape

        f_cc, f_cp, f_ci = extract_feature(example)

        f_cc = f_cc.reshape(f_cc.shape[0], -1)
        f_cp = f_cp.reshape(f_cp.shape[0], -1)
        f_ci = f_ci.reshape(f_ci.shape[0], -1)
        f_cc, f_cp, f_ci = normalization(f_cc, f_cp, f_ci)

        f_norm = np.hstack((f_cc, f_cp, f_ci))
        f_norm = PCA(n_components=200).fit_transform(f_norm)
        features.append(f_norm)

    features = np.array(features)
    features = features.reshape(features.shape[0], -1)
    return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Action recognition.')
    parser.add_argument('--load-model', action='store_true',
                        help='Load a pretrained model')
    parser.add_argument('--load-test', action='store_true',
                        help='Run trained model on test set')
    parser.add_argument('--input-dir', type=str, default='data/input',
                        help='Directory of the processed input')
    args = parser.parse_args()

    dataset = {}

    for fn in ACTION_FILE_NAMES.keys():
        dataset[fn] = np.load(os.path.join(args.input_dir, fn + '.npy'))

    indices, x = scatter_samples(dataset, 500, 50)  # 大样本分散为小样本
    y = mark_labels(indices)  # 把编号（文件名）换成标签
    y = LabelEncoder().fit_transform(y)  # 字符串标签转为数字标签

    y_test, x_test, y_train, x_train = split_dataset(y, x, 0.2)
    print('Total classes number:%d' % (np.max(y_train) + 1))

    features = integrate_features(x_train)

    if args.load_model:
        model = pickle.load(open('model/naive_bayes.pkl', 'rb'))
    else:
        model = train_and_save(features, y_train, 'model/naive_bayes.pkl')

    # Predict Output
    y_pred = model.predict(features)

    print('The Training set accuracy:%f' % accuracy_score(
        y_true=y_train, y_pred=y_pred, normalize=True))

    features = integrate_features(x_test)

    y_pred = model.predict(features)
    print('The testing set accuracy:%f' % accuracy_score(
        y_true=y_test, y_pred=y_pred, normalize=True))
