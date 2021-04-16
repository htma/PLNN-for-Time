import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from data_loader import MyCustomDataset
from plnn_based import PLNN

# 1. load an example of coffee dataset 

coffee_test = pd.read_csv('data/coffee_test.csv', sep=',', header=None).astype(float)
coffee_test_y = coffee_test.loc[:, 0]
coffee_test_x = coffee_test.loc[:, 1:]

# 2. Let's plot the two time series datasets using different colors for each class.
idx = 0
if coffee_test_y[idx] == 0:
    plt.plot(coffee_test_x.loc[idx, :], 'b')
else:
    plt.plot(coffee_test_x.loc[idx, :], 'r')
plt.title('An Example of Coffee Dataset')
plt.show()

def main():
    D_in, D_out = 286, 2
    H1, H2, H3 = 4, 16, 2
    model = PLNN(D_in, H1, H2, H3, D_out)
    model.load_state_dict(torch.load('./coffee_plnn.pt'))
    #print(model)
    test_loader = torch.utils.data.DataLoader(
        MyCustomDataset('./data/coffee_test.csv',
                        transform=transforms.Compose([
                            transforms.ToTensor()])),
        batch_size=64, shuffle=True)

    model.eval()
    data,target = test_loader.dataset[0][0], test_loader.dataset[0][1]
    data = data.view(-1, 286)
    output= model(data)
    pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
    print(pred.data, target)

if __name__ == '__main__':
    main()
