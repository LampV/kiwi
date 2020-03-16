#!/usr/bin/env python
# coding=utf-8
"""
@author: Jiawei Wu
@create time: 2019-12-06 23:01
@edit time: 2020-03-09 16:36
@desc: ASV 智能体
"""

import torch
import numpy as np
import time
from wjwgym.agents import DDPGBase
from asv_nn import ASVActorNet, ASVCriticNet
CUDA = torch.cuda.is_available()
import os

print('import agent')


class DDPG(DDPGBase):
    """DDPG类"""

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

    def learn_batch(self):
        if 'learn_step' not in self.__dict__:
            self.learn_step = 0
        c_loss, a_loss = self.learn()
        if c_loss is not None:
            self.summary_writer.add_scalar('c_loss', c_loss, self.learn_step)
            self.summary_writer.add_scalar('a_loss', a_loss, self.learn_step)
            self.learn_step += 1

    # TODO 将方法固定到基类
    def save(self, episode):
        state = {
            'actor_eval_net': self.actor_eval.state_dict(),
            'actor_target_net': self.actor_target.state_dict(),
            'critic_eval_net': self.critic_eval.state_dict(),
            'critic_target_net': self.critic_target.state_dict(),
            'episode': episode,
            'learn_step': self.learn_step
        }
        torch.save(state, './drlte.pth')
        
    def save_best(self, episode):
        state = {
            'actor_eval_net': self.actor_eval.state_dict(),
            'actor_target_net': self.actor_target.state_dict(),
            'critic_eval_net': self.critic_eval.state_dict(),
            'critic_target_net': self.critic_target.state_dict(),
            'episode': episode,
            'learn_step': self.learn_step
        }
        torch.save(state, './best_asv.pth')

    # TODO 将方法固定到基类
    def load(self, filename):
        print('\033[1;31;40m{}\033[0m'.format('加载模型参数...'))
        if not os.path.exists(filename):
            print('\033[1;31;40m{}\033[0m'.format('没找到保存文件'))
            return 0
        saved_state = torch.load(filename, map_location=torch.device('cpu'))
        self.actor_eval.load_state_dict(saved_state['actor_eval_net'])
        self.actor_target.load_state_dict(saved_state['actor_target_net'])
        self.critic_eval.load_state_dict(saved_state['critic_eval_net'])
        self.critic_target.load_state_dict(saved_state['critic_target_net'])
        self.learn_step = saved_state['learn_step']
        return saved_state['episode'] + 1
