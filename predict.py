import numpy as np
import csv
import sys
import pickle
import joblib
from validate import validate

"""
Predicts the target values for data in the file at 'test_X_file_path', using the model learned during training.
Writes the predicted values to the file named "predicted_test_Y_de.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""


class Node:
    def __init__(self, predicted_class, depth):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.depth = depth
        self.left = None
        self.right = None


def prediction(root, X):
    node = root

    while node.left:
        if X[node.feature_index] < node.threshold:
            node = node.left
        else:
            node = node.right

    return node.predicted_class


def feature_scaling(X):
    # mean normalisation

    for i in range(X.shape[1]):
        col = X[:, i]
        mean = np.mean(col)
        min = np.min(col)
        max = np.max(col)
        X[:, i] = (col-mean)/(max-min)

    return X


def import_data_and_model(test_X_file_path, model_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    model = joblib.load(model_file_path)
    return test_X, model


def predict_target_values(test_X, model):
    # Write your code to Predict Target Variables
    # HINT: You can use other functions which you've already implemented in coding assignments.
    test_X = feature_scaling(test_X)
    b=[]
    for i in range(test_X.shape[0]):
        a=prediction(model,test_X[i])
        b.append([a])

    b=np.array(b)
    return b.reshape(test_X.shape[0],1)


def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    test_X, model = import_data_and_model(test_X_file_path, 'MODEL_FILE.sav')
    pred_Y = predict_target_values(test_X, model)
    write_to_csv_file(pred_Y, "predicted_test_Y_de.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    validate(test_X_file_path, actual_test_Y_file_path="train_Y_de.csv") 
