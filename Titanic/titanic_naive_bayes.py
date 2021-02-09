from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
import pandas
import numpy as np

train_set = pandas.read_csv("data/train.csv")

numerical = train_set[["Age", "SibSp", "Parch"]]
categorical = train_set[["Sex", "Embarked"]]
y = train_set["Survived"]

numerical_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
numerical_imputer.fit(numerical)
numerical = numerical_imputer.transform(numerical)

categorical_imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
categorical_imputer.fit(categorical)
categorical = categorical_imputer.transform(categorical)

enc = OrdinalEncoder()
enc.fit(categorical)
categorical = enc.transform(categorical)

gnb = GaussianNB()
gnb.fit(numerical, y)

cnb = CategoricalNB()
cnb.fit(categorical, y)

#  Training set accuracy
num_train_pred = gnb.predict_log_proba(numerical)
print(num_train_pred)
cat_train_pred = cnb.predict_log_proba(categorical)
total_train_pred = num_train_pred + cat_train_pred

train_pred = np.argmax(total_train_pred, axis=1)
wrong_result = train_pred ^ y
train_perc = sum(wrong_result) / len(y) * 100
print(train_perc)

test_set = pandas.read_csv("data/test.csv")
test_set_num = test_set[["Age", "SibSp", "Parch"]]
test_set_cat = test_set[["Sex", "Embarked"]]

test_set_num = numerical_imputer.transform(test_set_num)

test_set_cat = categorical_imputer.transform(test_set_cat)
test_set_cat = enc.transform(test_set_cat)

num_test_pred = gnb.predict_log_proba(test_set_num)
cat_test_pred = cnb.predict_log_proba(test_set_cat)
total_test_pred = num_test_pred + cat_test_pred

test_pred = np.argmax(total_test_pred, axis=1)

df_pred = pandas.DataFrame(index=test_set["PassengerId"], columns=["Survived"], data=test_pred)
df_pred.to_csv("naive_bayes_result.csv")
