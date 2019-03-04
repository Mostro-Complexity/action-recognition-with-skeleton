import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

ACTION_FILE_NAMES = {
    '0510160858': 'still',
    '0510161326': 'talking on the phone',
    '0510165136': 'writing on whiteboard',
    '0510161658': 'drinking water',
    '0510171120': 'rinsing mouth with water',
    '0510170707': 'brushing teeth',
    '0510171427': 'wearing contact lenses',
    '0510171507': 'wearing contact lenses',
    '0510162529': 'talking on couch',
    '0510162821': 'relaxing on couch',
    '0510164129': 'cooking (chopping)',
    '0510163840': 'cooking (stirring)',
    '0510163444': 'opening pill container',
    '0510163513': 'opening pill container',
    '0510163542': 'opening pill container',
    '0510164621': 'working on computer',
    '0511121410': 'still',
    '0511121542': 'talking on the phone',
    '0511124850': 'writing on whiteboard',
    '0511121954': 'drinking water',
    '0511130523': 'rinsing mouth with water',
    '0511130138': 'brushing teeth',
    '0511130920': 'wearing contact lenses',
    '0511131018': 'wearing contact lenses',
    '0511122214': 'talking on couch',
    '0511122813': 'relaxing on couch',
    '0511124349': 'cooking (chopping)',
    '0511124101': 'cooking (stirring)',
    '0511123142': 'opening pill container',
    '0511123218': 'opening pill container',
    '0511123238': 'opening pill container',
    '0511123806': 'working on computer',
    '0512172825': 'still',
    '0512171649': 'talking on the phone',
    '0512175502': 'writing on whiteboard',
    '0512173312': 'drinking water',
    '0512164800': 'rinsing mouth with water',
    '0512164529': 'brushing teeth',
    '0512165243': 'wearing contact lenses',
    '0512165327': 'wearing contact lenses',
    '0512174513': 'talking on couch',
    '0512174643': 'relaxing on couch',
    '0512171207': 'cooking (chopping)',
    '0512171444': 'cooking (stirring)',
    '0512173520': 'opening pill container',
    '0512173548': 'opening pill container',
    '0512173623': 'opening pill container',
    '0512170134': 'working on computer',
    '0512150222': 'still',
    '0512150451': 'talking on the phone',
    '0512154505': 'writing on whiteboard',
    '0512150912': 'drinking water',
    '0512155606': 'rinsing mouth with water',
    '0512155226': 'brushing teeth',
    '0512160143': 'wearing contact lenses',
    '0512160254': 'wearing contact lenses',
    '0512151230': 'talking on couch',
    '0512151444': 'relaxing on couch',
    '0512152943': 'cooking (chopping)',
    '0512152416': 'cooking (stirring)',
    '0512151857': 'opening pill container',
    '0512151934': 'opening pill container',
    '0512152013': 'opening pill container',
    '0512153758': 'working on computer'
}

ORIGINAL_DATA_PATH = 'data/original'


def normalization(f_cc, f_cp, f_ci):
    scaler = MinMaxScaler(feature_range=(-1, 1))

    f_cc_std = scaler.fit_transform(f_cc)
    f_cp_std = scaler.fit_transform(f_cp)
    f_ci_std = scaler.fit_transform(f_ci)
    return f_cc_std, f_cp_std, f_ci_std


def extract_feature(joints):
    frame_num, joint_num, _ = joints.shape

    f_cc = np.empty((frame_num, int(comb(joint_num, 2)), 3), dtype=np.float32)
    f_cp = np.empty((frame_num, joint_num * joint_num, 3), dtype=np.float32)
    f_ci = np.empty((frame_num, joint_num * joint_num, 3), dtype=np.float32)

    k = 0
    for i in range(joint_num):
        for j in range(i + 1, joint_num):
            f_cc[:, k] = joints[:, i, :] - joints[:, j, :]
            k += 1

    for k in range(1, frame_num):
        for i in range(joint_num):
            for j in range(joint_num):
                f_cp[:, i * 15 + j] = joints[k, i, :] - \
                    joints[k - 1, j, :]

    for k in range(1, frame_num):
        for i in range(joint_num):
            for j in range(joint_num):
                f_ci[:, i * 15 + j] = joints[k, i, :] - \
                    joints[0, j, :]

    return f_cc, f_cp, f_ci


def descend_dimension(f):

    pca = PCA(n_components=12)
    pca.fit(f_norm)
    return pca.transform(f_norm)


def scatter_samples(x, sample_size, step):
    labels, samples = [], []
    for k in x:
        sample = x[k]
        frame_num = sample.shape[0]
        for i in range(0, frame_num, step):
            if i + sample_size > frame_num:
                break
            labels.append(k)
            samples.append(sample[i:i + sample_size, :])
    return np.array(labels), np.array(samples)


def split_dataset(y, x, ratio):
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

    return np.array(y_test), np.array(x_test), np.array(y_train), np.array(x_train)


def train_and_save(features, labels, path):
    # Create a Gaussian Classifier
    weights = compute_sample_weight('balanced', labels)
    model = GaussianNB()
    # Train the model using the training sets
    model.fit(features, labels, sample_weight=weights)
    pickle.dump(model, open(path, 'wb'))

    return model


def mark_labels(indices):
    mapped = [ACTION_FILE_NAMES[indices[i]] for i in range(indices.size)]

    return np.array(mapped)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Action recognition.')
    parser.add_argument('--input-dir', type=str, default='data/input',
                        help='Directory of the processed input')
    args = parser.parse_args()

    dataset = {}

    for fn in ACTION_FILE_NAMES.keys():
        dataset[fn] = np.load(os.path.join(args.input_dir, fn + '.npy'))

    indices, x = scatter_samples(dataset, 300, 150)  # 大样本分散为小样本
    y = mark_labels(indices)  # 把编号（文件名）换成标签
    y = LabelEncoder().fit_transform(y)  # 字符串标签转为数字标签

    y_test, x_test, y_train, x_train = split_dataset(y, x, 0.2)
    print('Total classes number:%d' % (np.max(y_train) + 1))

    plt.ion()
    fig = plt.figure()
    ax = Axes3D(fig)
    s = ax.scatter(x_train[0, 0, :, 0],
                x_train[0, 0, :, 1], x_train[0, 0, :, 2])

    for i in range(x_train[0].shape[0]):
        s.remove()
        s = ax.scatter(x_train[0, i, :, 0],
                       x_train[0, i, :, 1], x_train[0, i, :, 2])
        plt.draw()
        plt.pause(0.1)
