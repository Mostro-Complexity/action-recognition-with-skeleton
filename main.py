import argparse
import os
import pickle

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from util import *

ORIGINAL_DATA_PATH = 'data/original'


def integrate_features(samples):
    """Convert all samples to features with single time.

    `samples` : sample array, which contains N samples. shape=(N,)
    `return` : feature array, which contains N features. shape=(N,)
    """
    sample_num, frame_num, joint_num, _ = samples.shape
    n_dim = 200  # PCA降维后的维数
    features = np.empty((sample_num, frame_num * n_dim), dtype=np.float32)

    for i in range(sample_num):  # 对每个训练（测试）样本提取特征
        example = samples[i]

        f_cc, f_cp, f_ci = extract_feature(example)

        f_cc = f_cc.reshape(f_cc.shape[0], -1)
        f_cp = f_cp.reshape(f_cp.shape[0], -1)
        f_ci = f_ci.reshape(f_ci.shape[0], -1)
        f_cc, f_cp, f_ci = normalization(f_cc, f_cp, f_ci)

        f_norm = np.hstack((f_cc, f_cp, f_ci))
        f_norm = PCA(n_components=n_dim).fit_transform(f_norm)
        f_norm = f_norm.reshape(-1)
        features[i] = f_norm

    return features


def train_and_save_once(features, labels, path):
    """Train and save the model.

    N is the number of sample.
    `features` : Array of features. `shape`=`(N,)`
    `labels` : Array of labels. `shape`=`(N,)`
    `path` : Path to save the model. `type`=`str`
    `return` : `model` has been trained.
    """
    weights = compute_sample_weight('balanced', labels)
    # Create a Gaussian Classifier
    model = GaussianNB()
    # Train the model using the training sets
    model.fit(features, labels, sample_weight=weights)
    pickle.dump(model, open(path, 'wb'))

    print('The Training set accuracy:%f' % accuracy_score(
        y_true=y_train, y_pred=y_pred, normalize=True))
    return model


def train_and_save_multiple(batches_iter, classes, path, sample_weight=None):
    # Create a Gaussian Classifier
    model = GaussianNB()
    for i, (x_train, y_train, weights) in enumerate(batches_iter):
        # Train the model using the training sets
        # for _ in range(50):
        model.partial_fit(X=x_train, y=y_train,
                          classes=classes, sample_weight=weights)
        y_pred = model.predict(x_train)

        print('batch %d has finished, accuracy:%f' %
              (i, accuracy_score(y_true=y_train, y_pred=y_pred, normalize=True)))

    pickle.dump(model, open(path, 'wb'))
    return model


def feat_batches_iterator(samples, labels, batch_size=100):
    sample_num, frame_num, joint_num, _ = samples.shape
    n_dim = 200  # PCA降维后的维数
    partial_features = np.empty(
        (sample_num, frame_num * n_dim), dtype=np.float32)
    partial_labels = np.empty(sample_num, dtype=np.int32)
    partial_weights = np.empty(sample_num, dtype=np.float32)
    weights = compute_sample_weight('balanced', labels)

    for i in range(sample_num):  # 对每个训练（测试）样本提取特征
        example = samples[i]

        f_cc, f_cp, f_ci = extract_feature(example)

        f_cc = f_cc.reshape(f_cc.shape[0], -1)
        f_cp = f_cp.reshape(f_cp.shape[0], -1)
        f_ci = f_ci.reshape(f_ci.shape[0], -1)
        f_cc, f_cp, f_ci = normalization(f_cc, f_cp, f_ci)

        f_norm = np.hstack((f_cc, f_cp, f_ci))
        f_norm = PCA(n_components=n_dim).fit_transform(f_norm)

        partial_features[i % batch_size] = f_norm.reshape(-1)
        partial_labels[i % batch_size] = labels[i]
        partial_weights[i % batch_size] = weights[i]

        if i % batch_size == 0 and i != 0:
            yield partial_features, partial_labels, partial_weights

    yield (partial_features[0:i % batch_size],
           partial_labels[0:i % batch_size],
           partial_weights[0:i % batch_size])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Action recognition.')
    parser.add_argument('--load-model', action='store_true',
                        help='Load a pretrained model')
    parser.add_argument('--partial-fit', action='store_true',
                        help='Trained model using online learning')
    parser.add_argument('--input-dir', type=str, default='data/input',
                        help='Directory of the processed input')
    args = parser.parse_args()

    dataset = {}

    for fn in ACTION_FILE_NAMES.keys():
        dataset[fn] = np.load(os.path.join(args.input_dir, fn + '.npy'))

    indices, x = scatter_samples(dataset, 500, 10)  # 大样本分散为小样本
    y = mark_labels(indices)  # 把编号（文件名）换成标签
    y = LabelEncoder().fit_transform(y)  # 字符串标签转为数字标签

    y_test, x_test, y_train, x_train = split_dataset(y, x, 0.2)  # 交叉验证并乱序
    print('Total classes number:%d' % (np.max(y_train) + 1))

    if args.load_model and args.partial_fit:
        batches_num, batch_size = x_train.shape[0], 1000
        print('Total batches number:%d' % (batches_num // batch_size + 1))

        iterator = feat_batches_iterator(x_train, y_train, batch_size)
        model = train_and_save_multiple(iterator, np.unique(
            y_train), 'model/naive_bayes.pkl')
    elif args.load_model:
        features = integrate_features(x_train)
        model = pickle.load(open('model/naive_bayes.pkl', 'rb'))
        # Predict Output
        y_pred = model.predict(features)
    else:
        model = train_and_save_once(features, y_train, 'model/naive_bayes.pkl')

    features = integrate_features(x_test)

    y_pred = model.predict(features)
    print('The testing set accuracy:%f' % accuracy_score(
        y_true=y_test, y_pred=y_pred, normalize=True))
