import pandas
import numpy as np

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
"""
    Options:
        Naive Bayes
        Decision Tree
        FDA
        SVM
"""

type = "neural_network"

fill_str = "EMPTY"

read_train_set = pandas.read_csv("data/train.csv")
numerical = read_train_set[["Age", "SibSp", "Parch"]]
categorical = read_train_set[["Sex", "Embarked"]]
y = read_train_set["Survived"]

categorical = categorical.fillna(fill_str)

not_normalized = type == "decision_tree" or type == "random_forest" or type == "FDA"

if not_normalized:
    enc = OrdinalEncoder()
else:
    enc = OneHotEncoder(sparse=False)


enc.fit(categorical)
categorical = enc.transform(categorical)

train_set = np.concatenate((numerical, categorical), axis=1)

if not not_normalized:
    scaler = StandardScaler()
    scaler.fit(train_set)
    train_set = scaler.transform(train_set)

if not_normalized:
    imputer = SimpleImputer("most_frequent")
else:
    imputer = KNNImputer(n_neighbors=5)

imputer.fit(train_set)

train_set = imputer.transform(train_set)

X_train, X_test, y_train, y_test = train_test_split(train_set, y, test_size=0.33)

read_final_set = pandas.read_csv("data/test.csv")
final_set_num = read_final_set[["Age", "SibSp", "Parch"]]
final_set_cat = read_final_set[["Sex", "Embarked"]]

final_set_cat.fillna(fill_str)
final_set_cat = enc.transform(final_set_cat)

final_set = np.column_stack((final_set_num, final_set_cat))

if not not_normalized:
    final_set = scaler.transform(final_set)

final_set = imputer.transform(final_set)


#   Function here
if type == "decision_tree":
    classifier = DecisionTreeClassifier(max_depth=4)
if type == "random_forest":
    classifier = RandomForestClassifier(35, max_depth=3, max_features=3)
if type == "neural_network":
    classifier = MLPClassifier(solver="lbfgs", max_iter=50000, alpha=1e-5, hidden_layer_sizes=(10,3))
if type == "KNN":
    classifier = KNeighborsClassifier(n_neighbors=10, weights="uniform")
if type == "logistic_regression":
    classifier = LogisticRegression()
if type == "FDA":
    classifier = QuadraticDiscriminantAnalysis()
if type == "SVM":
    classifier = SVC(kernel="poly", degree=2)

classifier.fit(X_train, y_train)
train_pred = classifier.predict(X_train)
test_pred = classifier.predict(X_test)
final_pred = classifier.predict(final_set)

train_perc = sum(train_pred == y_train) / len(y_train) * 100
test_perc = sum(test_pred == y_test) / len(y_test) * 100

print("Train {:.3f}%, Test {:.3f}%".format(train_perc, test_perc))

df_pred = pandas.DataFrame(index=read_final_set["PassengerId"], columns=["Survived"], data=final_pred)
df_pred.to_csv("predictions/{}_result.csv".format(type))