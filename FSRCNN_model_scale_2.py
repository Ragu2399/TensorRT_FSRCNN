#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# This file contains functions for training a PyTorch MNIST Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
import os

from random import randint

# Network



   
class Net(nn.Module):
    def __init__(self, scale_factor, num_channels=1, d=56, s=12, m=4):
        super(Net, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2),
            nn.PReLU(d)
        )
        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            self.mid_part.extend([nn.Conv2d(s, s, kernel_size=3, padding=3//2), nn.PReLU(s)])
        self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=scale_factor, padding=9//2, output_padding=scale_factor-1)


    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x



class MnistModel(object):
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 100
        self.learning_rate = 0.0025
        self.sgd_momentum = 0.9
        self.log_interval = 100
        # Fetch MNIST data set.
#         self.train_loader = torch.utils.data.DataLoader(
#             datasets.MNIST('/tmp/mnist/data', train=True, download=True, transform=transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.1307,), (0.3081,))
#                 ])),
#             batch_size=self.batch_size,
#             shuffle=True,
#             num_workers=1,
#             timeout=600)
#         self.test_loader = torch.utils.data.DataLoader(
#             datasets.MNIST('/tmp/mnist/data', train=False, transform=transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.1307,), (0.3081,))
#                 ])),
#             batch_size=self.test_batch_size,
#             shuffle=True,
#             num_workers=1,
#             timeout=600)
        self.network = Net(2)
        print(self.network.eval())                                            

    # Train the network for one or more epochs, validating after each epoch.
    def learn(self, num_epochs=2):
        # Train the network for a single epoch
        def train(epoch):
            self.network.train()
            optimizer = optim.SGD(self.network.parameters(), lr=self.learning_rate, momentum=self.sgd_momentum)
            for batch, (data, target) in enumerate(self.train_loader):
                data, target = Variable(data), Variable(target)
                optimizer.zero_grad()
                output = self.network(data)
                #loss = F.(output, target)
                loss = F.mse_loss(output, data)
                loss.backward()
                optimizer.step()
                if batch % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch * len(data), len(self.train_loader.dataset), 100. * batch / len(self.train_loader), loss.data.item()))

                #break

        # Test the network
        def test(epoch):
            self.network.eval()
            test_loss = 0
            correct = 0
            for data, target in self.test_loader:
                with torch.no_grad():
                    data, target = Variable(data), Variable(target)
                output = self.network(data)
                #test_loss += F.nll_loss(output, target).data.item()
                test_loss += F.mse_loss(output, data).data.item()
                #pred = output.data.max(1)[1]
                #correct += pred.eq(target.data).cpu().sum()
            #test_loss /= len(self.test_loader)
            #print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(self.test_loader.dataset), 100. * correct / len(self.test_loader.dataset)))

        for e in range(num_epochs):
            train(e + 1)
            test(e + 1)

    def get_weights(self):
        # loaddict
        self.network.load_state_dict(torch.load('/home/raguhtic/Codes/FSRCNN scale 2/fsrcnn_x2.pth'))
        return self.network.state_dict()

    def get_random_testcase(self):
        data, target = next(iter(self.test_loader))
        case_num = randint(0, len(data) - 1)
        test_case = data.numpy()[case_num].ravel().astype(np.float32)
        test_name = target.numpy()[case_num]
        return test_case, test_name

