import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from data_loader import MyCustomDataset

def active_state(x):
    '''Calculate hidden neurons' active states.
       - x: hidden neurons outputs
       - Return a list of zeros/ones of hidden neuron states
    '''
    x = x.detach().numpy()
    x = x.reshape(x.shape[1])
    states = x.copy()
    states[x>0] = 1
    return list(states.astype(int))

class PLNN(nn.Module):
    def __init__(self,D_in, H1, H2, H3, D_out):
        super(PLNN, self).__init__()
        self.fc1 = nn.Linear(D_in, H1)
        self.fc2 = nn.Linear(H1, H2)
        self.fc3 = nn.Linear(H2, H3)
        self.fc4 = nn.Linear(H3, D_out)

    def forward(self, x):
        act_states = []
        h1 = F.relu(self.fc1(x))
        h1_states = active_state(h1)
        print('Hidden Layer 1 output : ', h1)
        print('Hidden Layer 1 active states: ', h1_states)
        h2 = F.relu(self.fc2(h1))
        h2_states = active_state(h2)
        h3 = F.relu(self.fc3(h2))
        h3_states = active_state(h3)
        act_states = list(np.hstack([h1_states, h2_states, h3_states]))
                                    
        out = self.fc4(h3)
        out = F.log_softmax(out, dim=1)
        return act_states, out

if __name__ == '__main__':
    #main()
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
    data,target = test_loader.dataset[15][0], test_loader.dataset[15][1]
    data = data.view(-1, 286)
    states, output= model(data)
    pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
    print(pred.data, target)
    print("sates : ", states)
        
    
                        
    
    
    
