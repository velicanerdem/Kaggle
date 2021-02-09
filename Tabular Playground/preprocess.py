import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Po
from torch.utils.data import TensorDataset

kernel = True

train_data = pd.read_csv("data/train.csv")
eval_data = pd.read_csv("data/test.csv")

eval_id = eval_data.index

X_train = train_data.drop("target", axis=1)
y_train = train_data["target"]
y_train = y_train.to_numpy().reshape(-1, 1)

feature_scaler = StandardScaler()
feature_scaler.fit(X_train)

X_train = feature_scaler.transform(X_train)
eval_data = feature_scaler.transform(eval_data)

if kernel:


X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3)

X_train = torch.tensor(X_train, dtype=torch.float)
X_test = torch.tensor(X_test, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.float)
y_test = torch.tensor(y_test, dtype=torch.float)
print(len(eval_data))
train_tensor = TensorDataset(X_train, y_train)
test_tensor = TensorDataset(X_test, y_test)
X_predict = torch.tensor(eval_data, dtype=torch.float)
predict_tensor = TensorDataset(X_predict)

if kernel == False:
    torch.save(train_tensor, "preprocess/train.pt")
    torch.save(test_tensor, "preprocess/test.pt")
    torch.save(predict_tensor, "preprocess/predict.pt")
else:
    torch.save(train_tensor, "preprocess/kernel_train.pt")
    torch.save(test_tensor, "preprocess/kernel_test.pt")
    torch.save(predict_tensor, "preprocess/kernel_predict.pt")