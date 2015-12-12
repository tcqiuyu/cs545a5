import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import time, datetime
import math
from sklearn.datasets import load_svmlight_file
from sklearn.svm import SVC, LinearSVC
from sklearn import cross_validation, pipeline, grid_search
from sklearn.feature_selection import *


def golub_score(X, y):
    if sparse.issparse(X):
        X = X.todense()

    scores = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        feature_i = X[:, i]

        miu_plus = np.mean(feature_i[np.where(y == 1)])
        miu_minus = np.mean(feature_i[np.where(y == -1)])
        sigma_plus = np.std(feature_i[np.where(y == 1)])
        sigma_minus = np.std(feature_i[np.where(y == -1)])

        if sigma_minus == 0.0:
            sigma_minus = 1.0
        if sigma_plus == 0.0:
            sigma_plus = 1.0

        scores[i] = np.abs(miu_plus - miu_minus) / (sigma_plus + sigma_minus)

    return scores, scores


def nonzero_coef_count(X, y):
    classifier = LinearSVC(penalty='l1', dual=False)
    counts = np.zeros(1000)
    for i in range(len(counts)):
        if (i % 10 == 0):
            print ">>> Iteration %d" % i
        classifier.fit(X, y)
        tmp = 0
        for coef in classifier.coef_[0]:
            if coef != 0.0:
                tmp += 1
        counts[i] = tmp
    count = int(round(np.mean(counts)))
    print "non-zero coefficients count: %d\n" % count
    return count


def cal_embedded_methods_accuarcy(X, y, feature_num):
    # L1 SVM
    L1_SVM = LinearSVC(penalty="l1", dual=False)
    cv = cross_validation.StratifiedKFold(y, 5, shuffle=True, random_state=0)
    score = cross_validation.cross_val_score(L1_SVM, X, y, cv=cv)
    print "---------- L1 SVM ----------"
    print "%0.4f\n" % np.mean(score)

    # L2 SVM trained on the features selected by the L1 SVM
    L1_SVM = LinearSVC(penalty="l1", dual=False)
    selector = RFE(L1_SVM, step=0.1, n_features_to_select=feature_num)
    X_t = selector.fit_transform(X, y)
    L2_SVM = LinearSVC(penalty="l2")
    cv = cross_validation.StratifiedKFold(y, 5, shuffle=True, random_state=0)
    score = cross_validation.cross_val_score(L2_SVM, X_t, y, cv=cv)
    print "---------- L2 SVM trained on the features selected by the L1 SVM ----------"
    print "%0.4f\n" % np.mean(score)

    # L2 SVM trained on all the features
    L2_SVM = LinearSVC(penalty="l2")
    cv = cross_validation.StratifiedKFold(y, 5, shuffle=True, random_state=0)
    score = cross_validation.cross_val_score(L2_SVM, X, y, cv=cv)
    print "---------- L2 SVM trained on all the features ----------"
    print "%0.4f\n" % np.mean(score)

    # L2 SVM that uses RFE (with an L2-SVM) to select relevant features
    L2_SVM = LinearSVC(penalty="l2")
    selector = RFE(L2_SVM, step=0.1, n_features_to_select=feature_num)
    rfe_SVM = pipeline.make_pipeline(selector, L2_SVM)
    cv = cross_validation.StratifiedKFold(y, 5, shuffle=True, random_state=0)
    score = cross_validation.cross_val_score(rfe_SVM, X, y, cv=cv)
    print "---------- L2 SVM that uses RFE (with an L2-SVM) to select relevant features ----------"
    print "%0.4f\n" % np.mean(score)

    # Use the class RFECV which automatically selects the number of features.
    L2_SVM = LinearSVC(penalty="l2", dual=False)
    selector = RFECV(L2_SVM, step=0.1, cv=5)
    rfecv_SVM = pipeline.make_pipeline(selector, L2_SVM)
    cv = cross_validation.StratifiedKFold(y, 5, shuffle=True, random_state=0)
    score = cross_validation.cross_val_score(rfecv_SVM, X, y, cv=cv)
    print "---------- Use the class RFECV which automatically selects the number of features. ----------"
    print "%0.4f\n" % np.mean(score)


def k_subsamples(X, y):
    k = 50
    numbers = np.zeros((k, X.shape[1]))
    for i in range(k):
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            X, y, train_size=0.8, random_state=42
        )
        idx = np.random.choice(
            X_train.shape[0], size=int(X_train.shape[0] * 0.8), replace=False
        )
        X_sub = X_train[idx, :]
        y_sub = y_train[idx]
        classifier = LinearSVC(penalty="l1", dual=False)
        classifier.fit(X_sub, y_sub)
        numbers[i] = (classifier.coef_[0] != 0.0)
    scores = np.sum(numbers, 0)
    return scores, scores


def plot_accuracy(X, y, optimal_C=False):
    # number of features increases 10% every iterations
    features = [1]
    while features[-1] < X.shape[1]:
        features.append(features[-1] + int(math.ceil(features[-1] * 0.1)))
    features[-1] = X.shape[1]

    c = np.logspace(start=-10, stop=10, num=21)

    # Golub score
    accuracy = []
    for feature_num in features:
        print ">>> %s Iteration %d" % (time.strftime("%X"), feature_num)
        l1_svm = LinearSVC(penalty="l1", dual=False)
        selector = SelectKBest(score_func=golub_score, k=feature_num)
        svm = pipeline.make_pipeline(selector, l1_svm)
        if optimal_C:
            gridsearch = grid_search.GridSearchCV(
                svm, c, cv=5, n_jobs=-1
            )
            gridsearch.fit(X, y)
            accuracy.append(gridsearch.best_score_)
        else:
            cv = cross_validation.StratifiedKFold(y, 5, shuffle=True, random_state=0)
            score = cross_validation.cross_val_score(svm, X, y, cv=cv)
            accuracy.append(np.mean(score))
    plt.plot(features, accuracy, "r-")
    plt.xlabel("Selected features number")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1])
    if optimal_C:
        plt.title("Golub with optimal C")
    else:
        plt.title("Golub")
    plt.grid("on")
    plt.show()

    # k-subsamples
    accuracy = []
    for feature_num in features:
        print ">>> %s Iteration %d" % (time.strftime("%X"), feature_num)
        l1_svm = LinearSVC(penalty="l2", dual=False)
        selector = SelectKBest(score_func=k_subsamples, k=feature_num)
        svm = pipeline.make_pipeline(selector, l1_svm)
        if optimal_C:
            gridsearch = grid_search.GridSearchCV(
                svm, c, cv=5, n_jobs=-1
            )
            gridsearch.fit(X, y)
            accuracy.append(gridsearch.best_score_)
        else:
            cv = cross_validation.StratifiedKFold(y, 5, shuffle=True, random_state=0)
            score = cross_validation.cross_val_score(svm, X, y, cv=cv)
            accuracy.append(np.mean(score))
    plt.plot(features, accuracy, "g-")
    plt.xlabel("Selected features number")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1])
    if optimal_C:
        plt.title("K-Subsamples with optimal C")
    else:
        plt.title("K-Subsamples")
    plt.grid("on")
    plt.show()

    # RFE SVM
    accuracy = []
    for feature_num in features:
        print ">>> %s Iteration %d" % (time.strftime("%X"), feature_num)
        l1_svm = LinearSVC(penalty="l1", dual=False)
        selector = RFE(l1_svm, step=0.1, n_features_to_select=feature_num)
        l2_svm = LinearSVC()
        rfe_svm = pipeline.make_pipeline(selector, l2_svm)
        if optimal_C:
            gridsearch = grid_search.GridSearchCV(
                rfe_svm, c, cv=5, n_jobs=-1
            )
            gridsearch.fit(X, y)
            accuracy.append(gridsearch.best_score_)
        else:
            cv = cross_validation.StratifiedKFold(y, 5, shuffle=True, random_state=0)
            score = cross_validation.cross_val_score(rfe_svm, X, y, cv=cv)
            accuracy.append(np.mean(score))
    plt.plot(features, accuracy, "b-")
    plt.xlabel("Selected features number")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1])
    if optimal_C:
        plt.title("RFE SVM with optimal C")
    else:
        plt.title("RFE SVM")
    plt.grid("on")
    plt.show()


if __name__ == '__main__':
    # load data
    print ">>> Loading data..."
    _arcene_train_X = np.genfromtxt("./data/arcene_train.data")
    _arcene_train_y = np.genfromtxt("./data/arcene_train.labels")
    _arcene_valid_X = np.genfromtxt("./data/arcene_valid.data")
    _arcene_valid_y = np.genfromtxt("./data/arcene_valid.labels")
    leu_X, leu_y = load_svmlight_file("./data/leu")

    # merge two Arcene datasets into one
    print ">>> Merging two Arcene datasets into one..."
    arcene_X = np.concatenate([_arcene_train_X, _arcene_valid_X])
    arcene_y = np.concatenate([_arcene_train_y, _arcene_valid_y])

    # find non-zero weight vector coefficients number
    print ">>> Finding non-zero weight vector coefficients number..."
    arcene_coef_count = nonzero_coef_count(arcene_X, arcene_y)
    leu_coef_count = nonzero_coef_count(leu_X, leu_y)

    # compare embedded methods accuracy
    print ">>> Comparing embedded methods accuracy..."
    cal_embedded_methods_accuarcy(arcene_X, arcene_y, arcene_coef_count)
    cal_embedded_methods_accuarcy(leu_X, leu_y, leu_coef_count)

    # plotting
    plot_accuracy(arcene_X, arcene_y)
    plot_accuracy(leu_X, leu_y)

    # plotting with margin C
    plot_accuracy(arcene_X, arcene_y, optimal_C=True)
    plot_accuracy(leu_X, leu_y, optimal_C=True)
