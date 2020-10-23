"""
This script is modified from https://github.com/pytorch/examples.git
"""
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from apex import amp
import time
# import profile
import cProfile, pstats, io
from pstats import SortKey


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # self.dropout1 = nn.Dropout2d(0.25)
        # self.dropout2 = nn.Dropout2d(0.5)
        # self.fc1 = nn.Linear(9216, 128)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
      
def train(args, model, device, train_loader, optimizer, epoch):

    to_dev_t =0
    forw_t   =0
    loss_t   =0
    back_t   =0
    step_t   =0


    start_time = time.time() ##
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        bef_to_dev = time.time()###
        data, target = data.to(device), target.to(device)
        aft_to_dev = time.time()###

        optimizer.zero_grad()

        bef_forward = time.time()###
        output = model(data)
        aft_forward = time.time() ###

        loss = F.nll_loss(output, target)
        aft_loss = time.time() ###

        # loss.backward()

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        aft_bak = time.time() ###

        optimizer.step()
        aft_step = time.time() ###

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

        # print("to dev %.5f, forw %.5f, loss %.5f, back %.5f, step %.5f" % (aft_to_dev-bef_to_dev, aft_forward-bef_forward, aft_loss-aft_forward, aft_bak-aft_loss, aft_step-aft_bak))

        to_dev_t+=aft_to_dev-bef_to_dev
        forw_t+=aft_forward-bef_forward
        loss_t+=aft_loss-aft_forward
        back_t+=aft_bak-aft_loss
        step_t+=aft_step-aft_bak


        # print("do one batch and break");break

    # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #     epoch, batch_idx * len(data), len(train_loader.dataset),
    #     100. * batch_idx / len(train_loader), loss.item()))
    end_time=time.time()
    print("training used time %.5f sec" %(end_time-start_time))

    print("to dev %.5f, forw %.5f, loss %.5f, back %.5f, step %.5f" % ( to_dev_t, forw_t, loss_t, back_t, step_t))



def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--opt-level', type=str, default="O3")
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}

    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                     )

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

    print("train shape", train_loader.dataset.data.shape)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):

        # use cprofile

        # pr = cProfile.Profile()
        # pr.enable()

        train(args, model, device, train_loader, optimizer, epoch)

        # pr.disable()
        # s = io.StringIO()
        # sortby = 'tottime'
        # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        # ps.print_stats()
        # print(s.getvalue())

        # test(model, device, test_loader)
        scheduler.step()

    

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    # profile.run('main()')
    main()