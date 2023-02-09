import numpy as np
import torch
def activation(out,threshold):
    if out>threshold:
        return 1
    else:
        return -1
def perceptron(and_input):
    a = torch.tensor([-1,-1,1,1], dtype = torch.float64)
    b= torch.tensor([-1,1,-1,1], dtype = torch.float64)
    y= torch.tensor([-1,-1,-1,1], dtype = torch.float64)
    w= torch.tensor([0.1,0.1], dtype = torch.float64)
    threshold=1
    learning_rate = 0.5
    i=0
    print("Perceptron Training:")
    for i in range(4):
        summation=a[i]*w[0]+b[i]*w[1]
        o=activation(summation,threshold)
        print("Input:"+str(a[i])+","+str(b[i]))
        print("Weights:"+str(w[0])+","+str(w[1]))
        print("summation:"+str(summation)+"\n"+"threshold"+" "+str(threshold))
        print("Actual Output:"+str(y[i])+"Predicted Output"+str(o))
        if(o!=y[i]):
            print("__________\n Updating Weights")
            w[0]=w[0]+learning_rate*(y[i]-0)*a[i]
            w[1]=w[1]+learning_rate*(y[i]-0)*b[i]
            print("Updated Weights:"+str(w[0])+","+str(w[1]))
        i=i+1
    summation=and_input[0]*w[0]+and_input[1]*w[1]   


and_input=  torch.tensor([-1,1], dtype = torch.float64)
print("AND GATE OUTPUT FOR"+str(and_input)+":"+str(perceptron(and_input)))