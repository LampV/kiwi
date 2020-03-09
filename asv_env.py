#!/usr/bin/env python
# coding=utf-8
"""
@author: Jiawei Wu
@create time: 2020-01-16 10:00
@edit time: 2020-03-09 16:39
@FilePath: /asv/asv_env.py
"""
from asv_dynamic import ASV
from move_point import MovePoint
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces


class ASVEnv(gym.Env):
    """
    ASV 的环境
    使用一个asv去追逐一个点
    """

    def __init__(self, action_type='velocity', interval=0.1):
        self.action_type = action_type
        self.asv = ASV(interval)
        self.interval = 0.1
        self.playground_shape = (600, 600)
        self.aim = MovePoint("func_sin")

        plt.ion()
        self.aim_his = [self.aim.position]
        self.asv_his = [self.asv.position.data]

        self.observation_space = spaces.Box(low=0, high=600, shape=(2,))
        self.action_space = spaces.Box(low=0, high=120, shape=(2,))

    def reset(self):
        self.aim.reset()
        self.asv.reset_state()
        aim_pos = self.aim.position
        asv_pos = self.asv.position.data
        self.aim_his = [list(aim_pos)]
        self.asv_his = [list(asv_pos)]
        plt.ioff()
        plt.clf()
        plt.ion()
        return self.get_state()

    def get_state(self):
        asv_pos = self.asv.position.data
        aim_pos = self.aim.position
        return aim_pos - asv_pos

    def get_reward(self):
        asv_pos = self.asv.position.data
        aim_pos = self.aim.position
        return -np.sum(np.power((asv_pos - aim_pos), 2))

    def get_done(self):
        return False

    def step(self, action):
        if self.action_type == 'velocity':
            self.asv.velocity = action
        elif self.action_type == 'acceleration':
            self.asv.acceleration = action
        else:
            raise TypeError("不在列表中的动作类型")

        # 记录
        cur_aim = self.aim.next_point(self.interval)
        cur_asv = self.asv.move()
        self.aim_his.append(list(cur_aim))
        self.asv_his.append(list(cur_asv.data))

        return self.get_state(), self.get_reward(), self.get_done(), ''

    def render(self):
        plt.clf()
        # 绘制aim
        plt.plot(*zip(*self.aim_his), 'y')

        # 绘制asv
        plt.plot(*zip(*self.asv_his), 'b')

        plt.pause(0.1)
