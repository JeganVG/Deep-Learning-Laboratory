import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn

iris = pd.read_csv("D:\\MEPCO\\SEMESTER 6\\Deep Learning\\Datasets\\Iris.csv")
iris.head() 

X = torch.tensor(iris.drop("Species", axis=1).values, dtype=torch.float)
y = torch.tensor(
    [0 if vty == "Setosa" else 1 if vty == "Versicolor" else 2 for vty in iris["Species"]], 
    dtype=torch.long
)

print(X.shape, y.shape) 

X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=0.8, random_state=42)

X_train, X_test, Y_train, Y_test = torch.tensor(X_train, dtype=torch.float32),torch.tensor(X_test, dtype=torch.float32),torch.tensor(Y_train, dtype=torch.float32),torch.tensor(Y_test, dtype=torch.float32)

samples, features = X_train.shape

class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.first_layer = nn.Linear(features, 5)
        self.second_layer = nn.Linear(5, 10)
        self.third_layer = nn.Linear(10, 15)
        self.final_layer = nn.Linear(15,1)
        self.relu = nn.ReLU()

    def forward(self, X_batch):
        layer_out = self.relu(self.first_layer(X_batch))
        layer_out = self.relu(self.second_layer(layer_out))
        layer_out = self.relu(self.third_layer(layer_out))

        return self.final_layer(layer_out)

regressor = Regressor()

preds = regressor(X_train[:5])

def TrainModel(model, loss_func, optimizer, X, Y, epochs=500):
    for i in range(epochs):
        preds = model(X) ## Make Predictions by forward pass through network

        loss = loss_func(preds.ravel(), Y) ## Calculate Loss

        optimizer.zero_grad() ## Zero weights before calculating gradients
        loss.backward() ## Calculate Gradients
        optimizer.step() ## Update Weights

        if i % 100 == 0: ## Print MSE every 100 epochs
            print("MSE : {:.2f}".format(loss))

from torch.optim import Adam

torch.manual_seed(42) ##For reproducibility.This will make sure that same random weights are initialized each time.

epochs = 1000
learning_rate = torch.tensor(1/1e3) # 0.001

regressor = Regressor()
mse_loss = nn.MSELoss()
optimizer = Adam(params=regressor.parameters(), lr=learning_rate)

TrainModel(regressor, mse_loss, optimizer, X_train, Y_train, epochs=epochs)

test_preds = regressor(X_test) ## Make Predictions on test dataset

train_preds = regressor(X_train) ## Make Predictions on train dataset

from sklearn.metrics import r2_score

print("Train R^2 Score : {:.2f}".format(r2_score(train_preds.detach().numpy().squeeze(), Y_train.detach().numpy())))
print("Test  R^2 Score : {:.2f}".format(r2_score(test_preds.detach().numpy().squeeze(), Y_test.detach().numpy())))