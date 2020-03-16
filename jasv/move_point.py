#!/usr/bin/env python
# coding=utf-8
"""
@author: Jiawei Wu
@create time: 2020-03-08 19:32
@edit time: 2020-03-12 16:03
@desc: 动点模型
"""
import numpy as np

class MovePoint(object):
    """动点模型
    通过 mave_path 选择动点轨迹
    """
    def __init__(self, move_path='linear'):
        self.point = np.array([0.0, 0.0])
        self.impl = getattr(self, move_path)

    def reset(self):
        self.point = np.array([0.0, 0.0])
        
    @property
    def position(self):
        return self.point
        
    def next_point(self, interval):
        return self.impl(interval)
    
    def linear(self, interval):
        self.point += np.array([interval, interval]) * 10
        return self.point

    def func_sin(self, interval):
        x, y = self.point
        x += interval * 10
        y = np.sin(x / 10)
        self.point = np.array([x, y])
        return self.point

    def random(self, interval):
        self.point = np.random.randint(0, 600, 2)
        return self.point