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

from multi_q import SelfPlayTrainer 
from networks.dqns import AdvantageQN
import cProfile

from single_net import main
from multi_q import main


def multi():
    import config
    g_config = config.config()
    g_config.env_name = "Pong2p-v0"
    g_config.nsteps_train = 5000
    env = gym.make(g_config.env_name)
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
    model_0 = dqns.AdvantageQN(env, config.config(), name="Adv_A_Profile")
    model_1 = dqns.AdvantageQN(env, config.config(), name="Adv_B")
    trainer = SelfPlayTrainer(model_0, model_1, env, g_config)
    trainer.run_parallel_models(exp_schedule, lr_schedule, True, True)


def single():
    import config
    g_config = config.config()
    g_config.nsteps_train = 5000

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
    model = AdvantageQN(env, g_config, name="SingleADV")
    model.run(exp_schedule, lr_schedule)

def prof_multi():
    cProfile.run("multi()", "multi_stats")
    import pstats

    p = pstats.Stats('multi_stats')
    p.sort_stats("cumulative").print_stats()

def prof_single():
    cProfile.run("single()", "single_stats")
    import pstats

    p = pstats.Stats('single_stats')
    p.sort_stats("cumulative").print_stats()

if __name__ == '__main__':
    prof_single()