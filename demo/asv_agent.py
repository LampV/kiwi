#!/usr/bin/env python
# coding=utf-8
"""
@author: Jiawei Wu
@create time: 2019-12-06 23:01
@edit time: 2020-03-30 17:38
@desc: ASV 智能体
"""

import torch
import numpy as np
import time
from vvlab.agents import DDPGBase
from asv_nn import ASVActorNet, ASVCriticNet
CUDA = torch.cuda.is_available()
import os

print('import agent')


class DDPG(DDPGBase):
    """DDPG类"""
    def _param_override(self):
        self.summary = True

    def _build_net(self):
        n_states, n_actions = self.n_states, self.n_actions
        self.actor_eval = ASVActorNet(n_states, n_actions, a_bound=self.bound)
        self.actor_target = ASVActorNet(n_states, n_actions, a_bound=self.bound)
        self.critic_eval = ASVCriticNet(n_states, n_actions)
        self.critic_target = ASVCriticNet(n_states, n_actions)

    def _build_noise(self):
        self.noise = np.random.random(2) - 0.5

    def get_action_noise(self, state, rate=1):
        action = self.get_action(state)
        action_noise = (np.random.random(2) - 0.5) * rate * self.bound
        action += action_noise
        return action[0]
