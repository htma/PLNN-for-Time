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
        states = {}
        h1 = F.relu(self.fc1(x))
        states['h1'] = active_state(h1)
        print('Hidden Layer 1 active states: ', states['h1'])
        h2 = F.relu(self.fc2(h1))
        states['h2'] = active_state(h2)
        h3 = F.relu(self.fc3(h2))
        states['h3'] = active_state(h3)
#                                    
        out = self.fc4(h3)
        out = F.log_softmax(out, dim=1)
        return states, out

def calculate_inequality_cofficients(model, data, filename):
    states, output = model(data)
#    _, prediction = torch.max(output.data, 1)
 #   prediction = np.array(prediction)
  #  prediction = prediction.reshape(prediction.shape[0],1)
  #  print("prediction is ", prediction)
    w1, b1 = model.state_dict()['fc1.weight'], model.state_dict()['fc1.bias']
    w2, b2 = model.state_dict()['fc2.weight'], model.state_dict()['fc2.bias']
    w3, b3 = model.state_dict()['fc3.weight'], model.state_dict()['fc3.bias']
    w4, b4 = model.state_dict()['fc4.weight'], model.state_dict()['fc4.bias']
    
    diag_s1 = torch.diag(torch.tensor((states['h1']),
                                      dtype=torch.float32))
    w2_hat = torch.matmul(w2, torch.matmul(diag_s1, w1))
    b2_hat = torch.matmul(w2, torch.matmul(diag_s1, b1)) + b2

    diag_s2 = torch.diag(torch.tensor((states['h2']),
                                      dtype=torch.float32))

    w3_hat = torch.matmul(w3, torch.matmul(diag_s2, w2_hat))
    b3_hat = torch.matmul(w3, torch.matmul(diag_s2, b2_hat)) + b3
#    print(w3_hat.size(), b3_hat.size())

    weights = torch.cat((w1, w2_hat, w3_hat)).numpy()
    bias = torch.cat((b1, b2_hat, b3_hat)).numpy()
    bias = bias.reshape(22, 1)
    active_states = np.hstack((states['h1'], states['h2'],
                               states['h3'])).astype(int)
    active_states = active_states.reshape(22, 1)

    weight_bias = np.append(weights, bias, axis=1)

    weight_bias_states = np.append(weight_bias, active_states, axis=1)

    print(len(weight_bias_states))
    output_file = open(filename, 'wb')
    np.savetxt(output_file, weight_bias_states, delimiter=',')
    output_file.close()
    return filename

def calculate_feasible_range(file_name):
    '''
       First devide  inequality cofficients into two classes, le_zeros and g_zeros. le_zeros means that of ax + by + c <= 0 and g_zeros means that of ax+by+c > 0).
       '''
    weight_bias_states = np.loadtxt(file_name, delimiter=',')
   
    le_zeros = weight_bias_states[weight_bias_states[(slice(None),-1)]<=0]
    g_zeros = weight_bias_states[weight_bias_states[(slice(None),-1)] > 0]

    # change the le_zeros to greate zeros by multiply -1 to the both sides of the inequality. That means make ax+by+c <= 0 to -ax-by-c > 0.
    nle_zeros = -1 * le_zeros

    # make the right sides of all equalities to be 0.
    nle_zeros[:,-1] = 0
    g_zeros[:,-1] = 0

    feasible_range = np.concatenate((nle_zeros, g_zeros), axis=0)
    print(len(feasible_range[0]))
    return feasible_range
    

def process_negative_ys(negative_y_states):

    states = -1*negative_y_states
    print(states[:,-1])
    zero_states = states[states[:,-1] == 0]
    print(zero_states[:,-1])
    print('one states size is ', len(zero_states))
    one_states = states[states[:,-1] == -1]
    zero_to_one_states = np.column_stack((zero_states[:,-1], np.ones(zero_states.shape[0])))
    one_to_zero_states = np.column_stack((one_states[:,-1], np.zeros(one_states.shape[0])))
    return np.concatenate((zero_to_one_states, one_to_zero_states), axis=0)

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
  #  print(pred.data, target)
 #   print("states : ", states.items())
    filename = './inequality_coeff.txt'
    calculate_inequality_cofficients(model, data, filename)
    calculate_feasible_range(filename)
    
  
        
    
                        
    
    
    
