import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from data_loader import MyCustomDataset

class PLNN(nn.Module):
    def __init__(self,D_in, H1, H2, H3, D_out):
        super(PLNN, self).__init__()
        self.fc1 = nn.Linear(D_in, H1)
        self.fc2 = nn.Linear(H1, H2)
        self.fc3 = nn.Linear(H2, H3)
        self.fc4 = nn.Linear(H3, D_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PLNN Coffee Example')
    parser.add_argument('--batch-size', type=int, default=64,
                        metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=500,
                        metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        metavar='LR', help='learning rate default(: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5,
                        metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=True, help='desables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50,
                        metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True, help='For saving the current model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_avaiable()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if use_cuda else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # loading training data
    train_loader = torch.utils.data.DataLoader(
        MyCustomDataset('./data/coffee_train.csv',
                        transform=transforms.Compose([
                            transforms.ToTensor()])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        MyCustomDataset('./data/coffee_test.csv', 
                        transform=transforms.Compose([
                            transforms.ToTensor()])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

#    print('length of test loader', len(test_loader))
 #   print('length of test loader dataset', len(test_loader.dataset))

    # define model
    D_in, D_out = 286, 2
    H1, H2, H3 = 4, 16, 2
    # training the model
    model = PLNN(D_in, H1, H2, H3, D_out).to(device)
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum)
    for epoch in range(1, args.epochs+1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            #        data = data.view(-1, 2) # maybe should use this line for multiple dimension data
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

        if epoch % 50 == 0:
       #     print('########', batch_idx, len(data), len(train_loader.dataset), len(train_loader))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                epoch, (batch_idx+1)*len(data), len(train_loader.dataset), 100.*(batch_idx+1)/len(train_loader), loss.item()))

#       testing
        model.eval()
        test_loss, correct = 0.0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)

        if epoch % 50 == 0:
            print('\nTest set Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))

    if (args.save_model):
        torch.save(model.state_dict(), 'coffee_plnn.pt')

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
    data,target = test_loader.dataset[0][0], test_loader.dataset[0][1]
    data = data.view(-1, 286)
    output= model(data)
    pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
    print(pred.data, target)
        
    
                        
    
    
    
