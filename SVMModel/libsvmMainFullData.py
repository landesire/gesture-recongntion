print(__doc__)

# License: BSD 3 clause

# Standard scientific Python imports
#import matplotlib.pyplot as plt
from time import time
from sklearn.datasets import load_svmlight_file
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import numpy as np


def gridsearchSVM():
    f = open("feature_all.txt")  # skip the header
    data = []
    line = f.readline()
    current_list = line.split(",")
    data.append(current_list)
    print(type(line))
    i = 0
    while line:
        line = f.readline()
        current_list = line.split(",")
        data.append(current_list)
        i = i + 1
        # print (i)
    data = data[:-1]
    f.close()
    all_data = np.array(data)
    del data
    label_list = []
    print(len(data))
    print(data[255][1])

    shuffled_indices = np.random.permutation(len(all_data))
    test_set_size = int(len(all_data) * 0.2)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    train_data = all_data[train_indices]
    test_data = all_data[test_indices]
    print("train data lenth" + str(len(train_data)))
    print("test data lenth" + str(len(test_data)))

    train_feature = train_data[:, 1:len(train_data[0])]
    train_label = train_data[:, 0].astype(np.int)

    test_feature = test_data[:, 1:len(test_data[0])]
    test_label = test_data[:, 0].astype(np.int)

    print(test_feature.shape)
    # print train_feature
    print("this is label")
    # print test_label
    # The digits dataset
    # digits = datasets.load_digits()
    # (x_train,y_train) = load_svmlight_file("heart_scale",dtype=np.float64,multilabel=True)
    # print x_train
    # print y_train
    t0 = time()
    param_grid = {'C': [1e3],
                  'gamma': [0.001],
                  }
    clf = GridSearchCV(SVC(kernel='linear', class_weight='balanced'), param_grid, verbose=10, n_jobs=1)
    clf = clf.fit(train_feature, train_label)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)
    print(clf.best_score_)
    print(clf.best_params_)

    y_pred = clf.predict(test_feature)
    print(y_pred)
    print(test_label)

    print(accuracy_score(test_label, y_pred))
    print(accuracy_score(test_label, y_pred, normalize=False))
if __name__ == '__main__':
    gridsearchSVM()
# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 4 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# matplotlib.pyplot.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.

