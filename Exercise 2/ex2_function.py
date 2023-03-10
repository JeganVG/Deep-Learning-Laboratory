import torch
from torch import Tensor
from torch.nn import Linear, MSELoss, functional as F
from torch.optim import SGD, Adam, RMSprop
from torch.autograd import Variable
import numpy as np

def data_generator(data_size=50):
    inputs = []
    labels = []
    for ix in range(data_size):        
        x = np.random.randint(1000) / 1000
        y = 7*(x*x*x) + 8*x+ 2
        inputs.append([x])
        labels.append([y])
        
    return inputs, labels

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = Linear(1, 6)
        self.fc3 = Linear(6, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x   
model = Net()

critereon = MSELoss()
optimizer = SGD(model.parameters(), lr=0.01)

nb_epochs = 70
data_size = 50

for epoch in range(nb_epochs):
    X, y = data_generator(data_size)
    
    epoch_loss = 0
    
    for ix in range(data_size):
        y_pred = model(Variable(Tensor(X[ix])))
        loss = critereon(y_pred, Variable(Tensor(y[ix]), requires_grad=False)) 
        # print(loss.data)    
        epoch_loss = loss.data
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print("Epoch: {} Loss: {}".format(epoch, epoch_loss))
#test the model

model.eval()
test_data = data_generator(1)
prediction = model(Variable(Tensor(test_data[0][0])))
print("Prediction: {}".format(prediction.data[0]))
print("Expected: {}".format(test_data[1][0]))