import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import torchvision
import math
import random

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='same', batchnorm=True, nonlinearity='leaky_relu'):
        super(ConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        if padding == 'same':
            self.cut_last_element = self.kernel_size % 2 == 0 and self.stride == 1
            self.padding = math.ceil((1 - self.stride + (self.kernel_size - 1)) / 2)
        elif isinstance(padding, str):
            raise ValueError(f"Padding type {padding} not supported.")
        else:
            self.cut_last_element = False
            self.padding = padding

        self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, stride=self.stride, padding=self.padding, bias=not batchnorm)

        if batchnorm:
            self.batchnorm = nn.BatchNorm2d(self.out_channels)
        else:
            self.batchnorm = None

        if nonlinearity == 'relu':
            self.nonlinearity == nn.ReLU()
        elif nonlinearity == 'leaky_relu':
            self.nonlinearity = nn.LeakyReLU(negative_slope=0.01)
        elif nonlinearity == 'elu':
            self.nonlinearity = nn.ELU()
        elif nonlinearity == 'tanh':
            self.nonlinearity = nn.Tanh()
        elif nonlinearity == 'sigmoid':
            self.nonlinearity = nn.Sigmoid()
        elif nonlinearity is None:
            self.nonlinearity = None
        else:
            raise NameError(f"Non-linearity {nonlinearity} is not currently supported.")

        if nonlinearity in ['relu', 'leaky_relu']:
            torch.nn.init.kaiming_normal_(self.conv.weight, nonlinearity=nonlinearity)

    def forward(self, x):
        x = self.conv(x)
        
        if self.cut_last_element:
            x = x[:,:,:-1,:-1]
            
        if self.batchnorm is not None:
            x = self.batchnorm(x)
        
        if self.nonlinearity is not None:
            x = self.nonlinearity(x)

        return x

class ResidualBlock(nn.Module):
    def __init__(self, size):
        super(ResidualBlock, self).__init__()

        self.conv1 = ConvLayer(size, size, 3)
        self.conv2 = ConvLayer(size, size, 3, nonlinearity=None)
        self.nonlinearity = nn.ReLU()

    def forward(self, x):
        return self.nonlinearity(self.conv2(self.conv1(x)) + x)

class GameNetwork(nn.Module):
    def __init__(self):
        super(GameNetwork, self).__init__()

    def forward(self, x):
        raise NotImplementedError()

    def clone(self):
        raise NotImplementedError()

    def predict(self, board):
        """
        predict the value and policy for a given board state in canonical form
        """
        return self.forward(board.unsqueeze(0).permute(0, 3, 1, 2).to(torch.float))

    def fit(self, x, y, batch_size=32, epochs=10, shuffle=True):
        print(f"BATCH: {x.shape[0]}")

        policy, value = y
        if shuffle:
            order = list(range(x.shape[0]))
            np.random.shuffle(order)
            x, policy, value = x[order], policy[order], value[order]
        
        avg_loss = 0
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(x.shape[0] // batch_size):
                pred_policy, pred_value = self.forward(x[batch_size * i : batch_size * (i+1)].permute(0, 3, 1, 2).to(torch.float), softmax=False)
                true_policy, true_value = policy[batch_size * i : batch_size * (i+1)], value[batch_size * i : batch_size * (i+1)]

                log_policy = F.log_softmax(pred_policy, dim=1)
                value_loss = F.mse_loss(pred_value, true_value.unsqueeze(1))
                # policy_loss = - (torch.log(pred_policy) * true_policy).sum(1).mean(0)

                if (log_policy != log_policy).any(): # check for nan
                    breakpoint()

                policy_loss = 5 * F.kl_div(log_policy, true_policy, reduction='batchmean')

                loss = value_loss + policy_loss

                if (loss != loss).any(): # check for nan
                    breakpoint()
                    
                self.optimizer.zero_grad()
                loss.backward()

                if (list(self.parameters())[0].grad != list(self.parameters())[0].grad).any(): # check for nan
                    breakpoint()

                self.optimizer.step()

                epoch_loss += loss.item()

            epoch_loss = epoch_loss * (batch_size / x.shape[0])
            avg_loss += epoch_loss
            print(epoch_loss)
        
        avg_loss /= epochs
        print("Average loss:", avg_loss)

        print(f"pred_policy: {pred_policy[0]}, pred_value: {pred_value[0]}, true_policy: {true_policy[0]}, true_value: {true_value[0]}")

        return self


class TicTacToeNetwork(GameNetwork):
    def __init__(self, size):
        super(TicTacToeNetwork, self).__init__()
        self.size = size

        self.conv1 = ConvLayer(2, 3) # (2 prior for each color)
        self.res1 = ResidualBlock(3)
        self.res2 = ResidualBlock(3)
        self.res3 = ResidualBlock(3)

        self.policy_conv = ConvLayer(3, 128, 1)
        self.policy_fc = nn.Linear((size ** 2) * 128, (size ** 2))

        self.value_conv = ConvLayer(3, 128, 1)
        self.value_fc = nn.Linear((size ** 2) * 128, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def clone(self):
        return TicTacToeNetwork(self.size)      

    def forward(self, x, softmax=True):
        x = self.res3(self.res2(self.res1(self.conv1(x))))
        value = torch.tanh(self.value_fc(self.value_conv(x).reshape(x.shape[0], -1)))
        policy = self.policy_fc(self.policy_conv(x).reshape(x.shape[0], -1))

        if softmax:
            policy = F.softmax(policy, dim=1)
            
        return policy, value

class OthelloNetwork(GameNetwork):
    def __init__(self, size):
        super(OthelloNetwork, self).__init__()
        self.size = size

        self.conv1 = ConvLayer(2, 128) # (2 prior for each color)
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)

        self.policy_conv = ConvLayer(128, 256, 1)
        self.policy_fc = nn.Linear((size ** 2) * 256, size ** 2 + 1) # moves plus pass

        self.value_conv = ConvLayer(128, 256, 1)
        self.value_fc = nn.Linear((size ** 2) * 256, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def clone(self):
        return OthelloNetwork(self.size)

    def forward(self, x, softmax=True):
        x = self.res3(self.res2(self.res1(self.conv1(x))))
        value = torch.tanh(self.value_fc(self.value_conv(x).reshape(x.shape[0], -1)))
        policy = self.policy_fc(self.policy_conv(x).reshape(x.shape[0], -1))

        if softmax:
            policy = F.softmax(policy, dim=1)

        return policy, value

if __name__ == "__main__":
    board = torch.randn((1, 2, 4, 4))
    network = OthelloNetwork(4)

    policy, value = network(board)
    print(policy, value)