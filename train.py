import numpy as np
import csv
import math
import pickle
import joblib


class Node:
    def __init__(self, predicted_class, depth):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.depth = depth
        self.left = None
        self.right = None


def import_package():
    X = np.genfromtxt("train_X_de.csv", delimiter=",",skip_header=1, dtype=np.float64)
    Y = np.genfromtxt("train_Y_de.csv", delimiter=",", dtype=np.float64)

    return X, Y


def calculate_entropy(Y):

    class_value = {}
    for i in Y:
        if i in class_value:
            class_value[i] += 1
        else:
            class_value[i] = 1

    entropy = 0
    for i in class_value.items():
        entropy += -(i[1]/len(Y)*math.log(i[1]/len(Y), 2))

    return entropy


def calculate_information_gain(Y_subsets):

    Y = np.concatenate(Y_subsets)
    entropy = calculate_entropy(Y)
    instance = sum(len(i) for i in Y_subsets)
    ig = entropy
    for Y in Y_subsets:
        p = len(Y)/instance
        ig -= p*calculate_entropy(Y)

    return ig


def calculate_split_entropy(Y_subsets):

    sum_Y = sum(len(i) for i in Y_subsets)
    sif = 0
    for i in Y_subsets:
        probability = len(i)/sum_Y
        sif -= probability*math.log(probability, 2)

    return sif


def calculate_gain_ratio(Y_subsets):

    ig = calculate_information_gain(Y_subsets)
    sif = calculate_split_entropy(Y_subsets)
    gain_ratio = ig/sif

    return gain_ratio


def calculate_gini_index(Y_subsets):

    sum_Y = sum(len(i) for i in Y_subsets)
    gini_index = 0
    Y = np.concatenate(Y_subsets)
    classes = sorted(set(Y))
    for i in Y_subsets:
        m = len(i)
        if m == 0:
            continue
        count = [i.count(j) for j in classes]
        gini = 1-sum((n/m)**2 for n in count)
        gini_index += (m/sum_Y)*gini

    return gini_index


def split_data_set(data_X, data_Y, feature_index, threshold):

    left_X = []
    right_X = []
    left_Y = []
    right_Y = []
    for i in range(len(data_X)):
        if data_X[i][feature_index] < threshold:
            left_X.append(data_X[i])
            left_Y.append(data_Y[i])
        else:
            right_X.append(data_X[i])
            right_Y.append(data_Y[i])

    return left_X, left_Y, right_X, right_Y


def get_best_split(X, Y):

    X = np.array(X)
    best_feature = 0
    best_gini_index = 10
    best_threshold = 0
    for i in range(len(X[0])):
        thresholds = set(sorted(X[:, i]))
        for t in thresholds:
            left_X, left_Y, right_X, right_Y = split_data_set(X, Y, i, t)
            if len(left_X) == 0 and len(right_X) == 0:
                continue
            gini_index = calculate_gini_index([left_Y, right_Y])
            if gini_index < best_gini_index:
                best_gini_index, best_feature, best_threshold = gini_index, i, t

    return best_feature, best_threshold


def construct_tree(X, Y, max_depth, min_size, depth):

    feature_index, threshold = get_best_split(X, Y)
    
    classes = list(set(Y))
    predicted_class = classes[np.argmax([np.sum(Y == c) for c in classes])]
    node = Node(predicted_class, depth)

    if len(set(Y)) == 1:
        return node

    if depth >= max_depth:
        return node

    if len(Y) <= min_size:
        return node

    if feature_index is None or threshold is None:
        return node

    node.feature_index = feature_index
    node.threshold = threshold

    left_X, left_Y, right_X, right_Y = split_data_set(X, Y, feature_index, threshold)

    node.left = construct_tree(np.array(left_X), np.array(left_Y), max_depth, min_size, depth+1)
    node.right = construct_tree(np.array(right_X), np.array(right_Y), max_depth, min_size, depth+1)

    return node


def feature_scaling(X):
    # mean normalisation

    for i in range(X.shape[1]):
        col = X[:, i]
        mean = np.mean(col)
        min = np.min(col)
        max = np.max(col)
        X[:, i] = (col-mean)/(max-min)

    return X


def model(X, Y):
    validation_set=X[710:]
    max_depth= 10
    min_size=1
    depth=1
    clf = construct_tree(X,Y,max_depth,min_size,depth)
    
    joblib.dump(clf, 'MODEL_FILE.sav')
    

if __name__ == "__main__":
    X, Y = import_package()
    X=feature_scaling(X)
    model(X, Y)
