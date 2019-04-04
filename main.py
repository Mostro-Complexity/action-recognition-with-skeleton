import argparse
import os
import pickle
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from matplotlib.ticker import MultipleLocator
from sklearn.utils import shuffle

from util import *

ORIGINAL_DATA_PATH = 'data/original'


@jit(nogil=True)
def integrate_features(samples, n_dim=200):
    """Convert all samples to features with single time.

    `samples` : sample array, which contains N samples. shape=(N,)
    `return` : feature array, which contains N features. shape=(N,)
    """
    sample_num, frame_num, joint_num, _ = samples.shape
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


@jit(nogil=True)
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


@jit(nogil=True)
def train_and_save_multiple(batches_iter, classes, path, sample_weight=None):
    # Create a Gaussian Classifier
    model = GaussianNB()
    for i, (x_train, y_train, weights) in enumerate(batches_iter):
        # Train the model using the training sets
        model.partial_fit(X=x_train, y=y_train,
                          classes=classes, sample_weight=weights)
        y_pred = model.predict(x_train)

        print('Batch %d has finished, accuracy:%f' %
              (i, accuracy_score(y_true=y_train, y_pred=y_pred, normalize=True)))

    pickle.dump(model, open(path, 'wb'))
    return model


@jit(nogil=True)
def feat_batches_iterator(samples, labels, batch_size=1000, n_dim=200):
    sample_num, frame_num, joint_num, _ = samples.shape

    labels, samples = shuffle(labels, samples)  # 打乱顺序
    weights = compute_sample_weight('balanced', labels)

    partial_features = np.empty(
        (batch_size, frame_num * n_dim), dtype=np.float32)
    partial_labels = np.empty(batch_size, dtype=np.int64)
    partial_weights = np.empty(batch_size, dtype=np.float32)

    for i in range(sample_num):  # 对每个训练（测试）样本提取特征
        f_cc, f_cp, f_ci = extract_feature(samples[i])

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


def show_confusion_matrix(x_test, y_true, y_pred, label_dict=None):  # TODO:check out
    matrix = confusion_matrix(y_true, y_pred)
    # Normalize by row
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    matrix *= 100

    # plot
    plt.switch_backend('agg')
    fig = plt.figure(figsize=[9.6, 7.2])
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    # fig.colorbar(cax)
    plt.imshow(matrix, interpolation='nearest',
               cmap=plt.cm.binary, aspect=0.7)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[j, i] > 5 and matrix[j, i] < 35:
                # fontsize用来设置字体大小
                ax.text(i, j, str('%.1f' % matrix[j, i]),
                        va='center', ha='center', color='black')
            elif matrix[j, i] >= 35:
                ax.text(i, j, str('%.1f' % matrix[j, i]),
                        va='center', ha='center', color='white')

    labels = ['']
    if label_dict is not None:
        labels.extend([label_dict[i] for i in range(len(label_dict))])

    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.title('Confusion Matrix')
    plt.show()
    # save
    plt.savefig('confusion_matrix.jpg')


def read_label_dict(data_dir):
    label_dict = None
    try:
        f = open(os.path.join(args.input_dir, 'label_dict.pkl'), 'rb')
        label_dict = pickle.load(f)
    except FileNotFoundError as e:
        pass
    return label_dict


def read_config(path):
    with open(path, 'r') as fp:
        param = json.loads(fp.read())
        return (param['n_dim'],
                param['batch_size'],
                param['sample_size'],
                param['step'],
                param['train_test_rate'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Action recognition.')
    parser.add_argument('--load-model', action='store_true',
                        help='Load a pretrained model')
    parser.add_argument('--partial-fit', action='store_true',
                        help='Trained model using online learning')
    parser.add_argument('--input-dir', type=str, default='data/input',
                        help='Directory of the processed input')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path of configuration file.')
    args = parser.parse_args()

    dataset = pickle.load(
        open(os.path.join(args.input_dir, 'input.pkl'), 'rb'))
    label_dict = read_label_dict(args.input_dir)

    N_DIM, BATCH_SIZE, SAMPLE_SIZE, STEP, TRAIN_TEST_RATE = read_config(
        args.config)

    y, x = scatter_samples(dataset, SAMPLE_SIZE, STEP)  # 大样本分散为小样本

    y_test, x_test, y_train, x_train = split_dataset(
        y, x, TRAIN_TEST_RATE)  # 交叉验证并乱序
    print('Total classes number:%d' % np.unique(y_train).size)

    if args.load_model and args.partial_fit:
        batches_num = x_train.shape[0]
        print('Total batches number:%d' % (batches_num // BATCH_SIZE + 1))

        iterator = feat_batches_iterator(
            x_train, y_train, BATCH_SIZE, N_DIM)
        model = train_and_save_multiple(iterator, np.unique(
            y_train), 'model/naive_bayes.pkl')
    elif args.load_model:
        model = pickle.load(open('model/naive_bayes.pkl', 'rb'))
        print('Model loaded.')
    else:
        features = integrate_features(x_train)
        model = train_and_save_once(features, y_train, 'model/naive_bayes.pkl')
        # Predict Output
        y_pred = model.predict(features)

    features = integrate_features(x_test, N_DIM)

    y_pred = model.predict(features)

    show_confusion_matrix(x_test, y_test, y_pred, label_dict)
    # TODO:输出图像以时间和方法作为文件名保存在visualization文件夹下
    print('The testing set accuracy:%f' % accuracy_score(
        y_true=y_test, y_pred=y_pred, normalize=True))
