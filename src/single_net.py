import sys
import os
from collections import deque

import numpy as np

import time
import gym
from networks import dqns
from train_schedule import LinearExploration, LinearSchedule
from utils.general import get_logger, Progbar, export_plot
from utils.preprocess import greyscale
from utils.wrappers import PreproWrapper, MaxAndSkipEnv
from  utils import actions

from networks.dqns import AdvantageQN

def main():
    import config
    g_config = config.config()

    # make env
    env = gym.make("Pong-v0")
    env = MaxAndSkipEnv(env, skip=g_config.skip_frame)
    env = PreproWrapper(env, prepro=greyscale, shape=(80, 80, 1), 
                        overwrite_render=g_config.overwrite_render)

    # exploration strategy
    # you may want to modify this schedule
    exp_schedule = LinearExploration(env, g_config.eps_begin, 
            g_config.eps_end, g_config.eps_nsteps)

    # you may want to modify this schedule
    # learning rate schedule
    lr_schedule  = LinearSchedule(g_config.lr_begin, g_config.lr_end,
            g_config.lr_nsteps)

    # train model
    model = AdvantageQN(env, config.config(), name="SingleADV")
    model.run(exp_schedule, lr_schedule)

if __name__ == '__main__':
    main()
