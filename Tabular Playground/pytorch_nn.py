import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


def main():
    status_test = True
    status_predict = False

    batch_size = 4
    num_workers = 4

    train_tensor = torch.load("preprocess/train.pt")
    train_loader = DataLoader(dataset=train_tensor, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    num_features = len(train_tensor[0][0])
    if status_test:
        test_tensor = torch.load("preprocess/test.pt")
        test_loader = DataLoader(dataset=test_tensor, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                 pin_memory=True)
    if status_predict:
        predict_file = pd.read_csv("data/test.csv")
        predict_id = predict_file["id"]
        predict_tensor = torch.load("preprocess/predict.pt")
        predict_loader = DataLoader(dataset=predict_tensor, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                    pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    class Net(nn.Module):
        def __init__(self, num_features, size_hidden_layer, n_hidden_layer):
            super(Net, self).__init__()
            hidden_layers = list()
            hidden_layers.append(nn.Linear(num_features, size_hidden_layer))
            for _ in range(n_hidden_layer-1):
                hidden_layers.append(nn.Linear(size_hidden_layer, size_hidden_layer))
            self.hidden_layers = nn.Sequential(*hidden_layers)
            self.last_layer = nn.Linear(size_hidden_layer, 1)

        def forward(self, x):
            for i in range(len(self.hidden_layers)):
                x = torch.relu(self.hidden_layers[i](x))
            return self.last_layer(x)

    net = Net(num_features, 20, 3)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    net.to(device)

    num_for_print = 2500
    train_amount = 30000
    for epoch in range(1):
        running_loss = 0
        for i, data in enumerate(train_loader, 0):
            inputs, values = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, values)

            running_loss += loss.item()

            l2_lambda = 0.1
            l2_reg = 0.0
            for param in net.parameters():
                l2_reg += torch.norm(param)

            loss += l2_reg * l2_lambda

            loss.backward()
            optimizer.step()


            if i  == train_amount / batch_size:
                break

            if i % (num_for_print/batch_size) == (num_for_print/batch_size-1):
                print("[{}, {}], loss: {:.3f}".format(epoch+1, (i+1)*batch_size, running_loss / num_for_print))
                running_loss = 0
    print("Finished training")

    test_subset = True
    test_amount = 10000
    if status_test:
        running_loss = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):
                inputs, values = data[0].to(device), data[1].to(device)
                outputs = net(inputs)
                loss = criterion(outputs, values)
                running_loss += loss.item()
                if test_subset:
                    if i >= test_amount / batch_size:
                        break

        print("Test loss: " + str(running_loss / len(test_tensor)))

    if status_predict:
        predictions = torch.zeros(len(predict_id)).to(device)
        with torch.no_grad():
            for i, data in enumerate(predict_loader, 0):
                inputs = data[0].to(device)
                outputs = net(inputs)
                for j in range(batch_size):
                    predictions[batch_size*i+j] = outputs[j]

        predictions = predictions.cpu().numpy()
        predictions = pd.DataFrame(index=predict_id, columns=["target"], data=predictions)
        predictions.index.name = "id"
        predictions.to_csv("predictions/neural_network.csv")


if __name__ == "__main__":
    main()