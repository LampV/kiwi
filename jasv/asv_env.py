#!/usr/bin/env python
# coding=utf-8
"""
@author: Jiawei Wu
@create time: 2020-01-16 10:00
@edit time: 2020-03-12 16:28
@desc: 包含asv与目标动点的环境
"""
from .asv_dynamic import ASV
from .move_point import MovePoint
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces


class ASVEnv(gym.Env):
    """
    ASV 的环境
    使用一个asv去追逐一个点
    """

    def __init__(self, aim_type='linear', action_type='velocity', interval=0.1):
        self.action_type = action_type
        self.asv = ASV(interval)
        self.interval = 0.1
        self.playground_shape = (600, 600)
        self.aim = MovePoint(aim_type)

        plt.ion()
        self.aim_his = [self.aim.position]
        self.asv_his = [self.asv.position]

        self.observation_space = spaces.Box(low=0, high=600, shape=(2,))
        self.action_space = spaces.Box(low=0, high=120, shape=(2,))

    def reset(self):
        """重设环境状态
        将目标点重置到(0, 0)之后，获取下一个目标点
        将船只重置为(0, 0)
        reset返回的state是当前点与目标点的差值
        """
        self.aim.reset()
        self.aim.next_point(self.interval)
        self.asv.reset_state()
        aim_pos = self.aim.position
        asv_pos = self.asv.position
        self.aim_his = [list(aim_pos)]
        self.asv_his = [list(asv_pos)]
        plt.ioff()
        plt.clf()
        plt.ion()
        return self.get_state()

    def get_state(self):
        """获取当前环境状态，即目标点坐标与船只坐标的差值"""
        asv_pos = self.asv.position
        aim_pos = self.aim.position
        return aim_pos - asv_pos

    def get_reward(self):
        """获取当前奖励，即目标点坐标与船只坐标的距离的负值
        注意距离越大奖励应该越小，所以取负值
        """
        asv_pos = self.asv.position
        aim_pos = self.aim.position
        return -np.sum(np.power((asv_pos - aim_pos), 2))

    def get_done(self):
        return False

    def step(self, action):
        """执行动作，获取奖励和新的状态
        在获得action之后，让asv根据asv移动
        奖励应该是对于当前aim（即执行action的aim），以及移动以后的asv计算
        状态则应该是对应下一个aim（即实际aim），以及移动后的asv计算
        """
        if self.action_type == 'velocity':
            self.asv.velocity = action
        elif self.action_type == 'acceleration':
            self.asv.acceleration = action
        else:
            raise TypeError("不在列表中的动作类型")

        # 让asv移动，则当前asv坐标更新为移动后的坐标
        cur_asv = self.asv.move()
        # 注意奖励永远是根据当前aim坐标和当前asv坐标计算，当前aim尚未移动
        reward = self.get_reward()
        # 计算完奖励之后，可以移动aim坐标
        cur_aim = self.aim.next_point(self.interval)
        # 此时aim已经是下一个要追逐的点，可以计算state
        state = self.get_state()

        # 记录坐标点，便于绘图
        self.aim_his.append(list(cur_aim))
        self.asv_his.append(list(cur_asv))

        return state, reward, self.get_done(), ''

    def render(self):
        plt.clf()
        # 绘制aim
        plt.plot(*zip(*self.aim_his), 'y')

        # 绘制asv
        plt.plot(*zip(*self.asv_his), 'b')

        plt.pause(0.1)
