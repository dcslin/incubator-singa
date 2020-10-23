import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from apex import amp
import time
import argparse


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(8, 8, 3, 1)
        self.fc1 = nn.Linear(26 * 26 * 3, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        # print(x.shape)
        x = torch.flatten(x, 1)
        print(x.shape)
        # x = self.fc1(x)
        # output = F.log_softmax(x, dim=1)
        # return output
        return x


def main():
    parser = argparse.ArgumentParser(description='One Layer')
    parser.add_argument('--opt-level', type=str)
    parser.add_argument('--eval',
                        default='True',
                        action='store_false',
                        dest='train')
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    bs = 64
    data = torch.rand(bs, 8, 28, 28).to(device)
    # target = torch.randint(0, 9, (bs,)).to(device)
    target = torch.rand(bs, 5408).to(device)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    model, optimizer = amp.initialize(model,
                                      optimizer,
                                      opt_level=args.opt_level)

    start_time = time.time()
    if args.train:
        model.train()
        for b in range(0, 1):
            optimizer.zero_grad()
            output = model(data)
            # loss = F.nll_loss(output, target)
            loss = F.mse_loss(output, target)
            print(loss)
            # loss.backward()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
    else:
        model.eval()
        output = model(data)

    end_time = time.time()
    print("training used time %.5f sec" % (end_time - start_time))


if __name__ == '__main__':
    main()