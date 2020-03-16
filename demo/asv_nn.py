#!/usr/bin/env python
# coding=utf-8
"""
@author: Jiawei Wu
@create time: 2020-01-16 09:40
@edit time: 2020-03-09 11:09
@FilePath: /asv_nn.py
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
CUDA = torch.cuda.is_available()

print('import nn')
class ASVActorNet(nn.Module):
    """
    定义Actor的网络结构：
    三个隐藏层，每层都是全连接，100个神经元
    每层50%的dropout
    隐藏层之间用ReLU激活，输出层使用tanh激活
    """

    def __init__(self, n_states, n_actions, n_neurons=30, a_bound=1):
        """
        定义隐藏层和输出层参数
        @param n_obs: number of observations
        @param n_actions: number of actions
        @param n_neurons: 隐藏层神经元数目
        @param a_bound: action的倍率
        """
        super().__init__()
        self.bound = a_bound
        self.fc1 = nn.Linear(n_states, n_neurons)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(n_neurons, n_actions)
        self.out.weight.data.normal_(0, 0.1)
        if CUDA:
            self.bound = torch.FloatTensor([self.bound]).cuda()
        else:
            self.bound = torch.FloatTensor([self.bound])

    def forward(self, x):
        """
        定义网络结构: 第一层网络->ReLU激活->输出层->tanh激活->softmax->输出
        """
        x = x.cuda() if CUDA else x
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        action_value = torch.tanh(x)
        action_value = action_value * self.bound
        return action_value


class ASVCriticNet(nn.Module):
    """
    定义Critic的网络结构：
    三个隐藏层，每层都是全连接，100个神经元
    每层50%的dropout
    隐藏层之间用ReLU激活

    Critic和Actor的区别在于输入维度和输出维度
    Critic接收state+action作为输入，输出则总是1维
    """

    def __init__(self, n_states, n_actions, n_neurons=64, a_bound=1):
        """
        @param n_states: number of observations
        @param n_actions: number of actions
        @param n_neurons: 隐藏层神经元数目，按照论文规定是100
        @param a_bound: action的倍率
        """
        super().__init__()

        self.fc1 = nn.Linear(n_states+n_actions, n_neurons)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(n_neurons, 1)
        self.out.weight.data.normal_(0, 0.1)


    def forward(self, s, a):
        x = torch.cat((s, a), dim=-1)
        x = x.cuda() if CUDA else x
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        q_value = torch.tanh(x)
        return q_value