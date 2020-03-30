#!/usr/bin/env python
# coding=utf-8
"""
@author: Jiawei Wu
@create time: 2020-01-16 10:23
@edit time: 2020-03-09 16:42
@FilePath: /asv/asv_main.py
"""

from jasv import ASVEnv
import numpy as np
import matplotlib.pyplot as plt
import time
from asv_agent import DDPG
import os
import json
import platform
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class RandAgent():
    def get_action_noise(self, s):
        return np.array([10, 10]) + (np.random.random(2) - 0.5) * 5


MAX_EPISODE = 10000000
MAX_DECAYEP = 500
MAX_STEP = 100


def rl_loop(need_load=True):
    RENDER = False

    env = ASVEnv()
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high[0]

    agent = DDPG(s_dim, a_dim, a_bound, MAX_MEM=10000, MIN_MEM=64, BATCH_SIZE=32)
    # agent = RandAgent()
    if need_load:
        START_EPISODE = agent.load('best_asv.pth')
    else:
        START_EPISODE = 0

    summary_writer = agent.get_summary_writer()
    best_cum_reward = -9999
    for e in range(START_EPISODE, MAX_EPISODE):
        cur_state = env.reset()
        cum_reward = 0
        for step in range(MAX_STEP):

            noise_decay_rate = max((MAX_DECAYEP - e) / MAX_DECAYEP, 0.1)
            
            # action = agent.get_action_noise(cur_state, rate=noise_decay_rate)
            action = agent.get_action(cur_state)[0]
            
            next_state, reward, done, info = env.step(action)
            
            reward = float(reward / 10000)
            
            agent.add_step(cur_state, action, reward, done, next_state)
            agent.learn()
            
            info = {
                    "cur_state": list(cur_state), "action": list(action),
                    "next_state": list(next_state), "reward": reward, "done": done
                }
            print(info, flush=True)

            cur_state = next_state
            cum_reward += reward
            if RENDER:
                env.render()
                time.sleep(0.1)

            done = done or step == MAX_STEP-1
            if done:
                print(f'episode: {e}, cum_reward: {cum_reward}', flush=True)
                # 在Linux平台上始终不开启RENDER
                if cum_reward > 0.1 and platform.system() != 'Linux':
                    RENDER = True
                break
        summary_writer.add_scalar('cum_reward', cum_reward, e)
        agent.save(e)   # 保存网络参数  
        
        # 保存最佳
        if cum_reward > best_cum_reward:
            agent.save(e, 'best_asv.pth')

if __name__ == '__main__':
    rl_loop()
