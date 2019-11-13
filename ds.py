import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier



# Function importing Dataset
def importdata():
    balance_data = pd.read_csv(
        'ModLense.csv',
        sep=',', header=None)

    bh = pd.read_csv("ModLense1.csv",sep=',',header = None)
    # Printing the dataswet shape
    print ("Dataset Length: ", len(balance_data))
    print ("Dataset Shape: ", balance_data.shape)

    # Printing the dataset obseravtions
    print ("Dataset: ", bh.head())
    X = bh.values[:, 0:4]
    Y = bh.values[:, 4]
    return balance_data


# Function to split the dataset
def splitdataset(balance_data):
    # Seperating the target variable
    X = balance_data.values[:, 0:4]
    Y = balance_data.values[:, 4]

    # Spliting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=100)

    return X, Y, X_train, X_test, y_train, y_test

# Function to perform training with entropy.
def train_using_entropy(X_train, X_test, y_train):
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(
        criterion="entropy", random_state=100,
        max_depth=3, min_samples_leaf=5)

    # Performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy


# Function to make predictions
def prediction(X_test, clf_object):
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred


# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: ",
          confusion_matrix(y_test, y_pred))

    print ("Accuracy : ",
           accuracy_score(y_test, y_pred) * 100)

    print("decision tree classification")
    print("Report : ",
          classification_report(y_test, y_pred))

def rf(X_train,y_train,X_test,y_test):
    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(X_train, y_train)
    rfc_pred = rfc.predict(X_test)
    print("Random forests classification")
    print(confusion_matrix(y_test, rfc_pred))
    print(classification_report(y_test,rfc_pred))
    print("Accuracy:",accuracy_score(y_test,rfc_pred)*100)

# Driver code
def main():
    # Building Phase
    data = importdata()
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
    clf_entropy = train_using_entropy(X_train, X_test, y_train)
    print("Results Using Entropy:")
    # Prediction using entropy
    y_pred_entropy = prediction(X_test, clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)
    # Prediction using Random Forests.
    rf(X_train,y_train,X_test,y_test)

# Calling main function
if __name__ == "__main__":
    main()