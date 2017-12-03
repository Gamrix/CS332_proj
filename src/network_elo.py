from __future__ import division

import itertools
import time
import os
import random

import gym


from networks import dqns
from train_schedule import LinearExploration, LinearSchedule
from utils.general import get_logger, Progbar, export_plot
from utils.preprocess import greyscale
from utils.wrappers import PreproWrapper, MaxAndSkipEnv
from  utils import actions
import config
import csv


def update_elo(m0, m1, k, p0_wins, p0_losses):
    q_0 = 10 ** (m0.elo/ 400)
    q_1 = 10 ** (m1.elo / 400)
    expected_win_rate = q_0 / (q_0 + q_1)
    actual_rate = p0_wins / (p0_wins + p0_losses)
    change = k * (actual_rate - expected_win_rate)
    m0.elo += change
    m1.elo -= change

    

# games_per_match = 2

# Networks can't play against past versions of themselves, or networks they played against

class Evaluator(object):

    def __init__(self, env, config):
        name =  time.strftime("%m%d_%H%M")

        config.output_path = "./elo_scores/{}/".format(name)
        config.log_path = config.output_path + "log.log"

        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)

        self.logger = get_logger(config.log_path.format(time.strftime("_%m%d_%H%M")))

    def evaluate(self,model_0, model_1, env=None, exp_schedule=None):
        """
        Evaluation with same procedure as the training
        """
        # log our activity only if default call
        if env is None:
            env = self.env

        score_0 = 0
        score_1 = 0

        model_0.train_init()
        model_1.train_init()

        state = env.reset()
        while True:
            action_0 = model_0.train_step_pre(state, exp_schedule)
            action_1 = model_1.train_step_pre(state[:, ::-1], exp_schedule)
            cur_action = actions.trans(action_0, action_1)

            # perform action in env
            new_state, reward, done, info =env.step(cur_action)


            # need to start another game
            model_0.train_step_post(reward, done, 0, None, False)
            model_1.train_step_post(-reward, done, 0, None, False)

            if done:
                break
            if reward != 0:
                if reward == 1:
                    score_0 += 1
                elif reward == -1:
                    score_1 += 1
                else:
                    raise Exception("Invalid reward, {}".format(reward))

            # store last state in buffer
            state = new_state
        return score_0, score_1

def model_info(model):
    return model.__class__.__name__, model.model_dir, model.num

model_info_names = ["model", "model_dir", "num"]

def main():
    g_config = config.config()
    g_config.env_name = "Pong2p-v0"
    env = gym.make(g_config.env_name)
    env = MaxAndSkipEnv(env, skip=g_config.skip_frame)
    env = PreproWrapper(env, prepro=greyscale, shape=(80, 80, 1),
                        overwrite_render=g_config.overwrite_render)
    exp_schedule = LinearExploration(env, g_config.eps_begin,
                                        g_config.eps_end, g_config.eps_nsteps)
    lr_schedule  = LinearSchedule(g_config.lr_begin, g_config.lr_end,
                                    g_config.lr_nsteps)


    evaluator = Evaluator(env, config)
    csv_file = open(evaluator.config.output_path + "results.csv" , mode='w', newline="")
    csv_res = csv.writer(csv_file)
    csv_res.writerow(["model_0", "model_dir_0", "num_0", "model_1", "model_dir_1", "num_1", "win_0", "win_1"]])


    def enumerate_models(model_dir, model_nums, name):
        models = []
        for m in model_nums:
            cur_model = dqns.AdvantageQN(env, g_config(), name=name)
            cur_model.num = m
            cur_model.elo = 1000
            cur_model.model_dir = model_dir
            cur_model.load(model_dir.format(m))
            models.append(cur_model)
        return models

    pairs = []

    def compatable_with(model_set_a, model_sets_b):
        msb = itertools.chain.from_iterable(model_sets_b)
        pairs.extend(itertools.product(model_set_a, msb))
        
    # Now to specify the models that are available

    # scoring 1 game takes
    # 1.5 min * (15/ 100) ~= 15 sec
    # one hour =  240 games -> 480 game results 
    rounds = 0
    models = []

    def first_run():
        # ok, first goal is to get scores (25 games each) for 

        # 1 single play , 2 self-play @ 250k
        # 1 single play , 2 self-play @ 1M
        # 1 single play , 2 self-play @ 2.5M
        # 2 single play , 4 self-play ends
        # total 15 models
        single_play = enumerate_models(FILL_IN)        
        self_play0 = enumerate_models(FILL_IN)        
        self_play1 = enumerate_models(FILL_IN)        

        compatable_with(single_play, [self_play0, self_play1])
        compatable_with(self_play0, [single_play, self_play1])
        compatable_with(self_play1, [single_play, self_play0])

        nonlocal models = single_play + self_play0 + self_play1
        nonlocal rounds = (15 * 25 / 2) / len(pairs)  + 1

    # now to actually score the games 
    results = []

    for m0, m1 in pairs:
        score_0, score_1 = evaluator.evaluate(m0, m1)
        update_elo(m0, m1, 30, score_0, score_1)
        csv_res.writerow([*model_info(m0), *model_info(m1), score_0, score_1])
        results.append([m0, m1, score_0, score_1])
    
    for i in range(10):
        for m0, m1, score_0, score_1 in random.shuffle(results)
            update_elo(m0, m1, 30, score_0, score_1)
    
    elo_res_f = open(evaluator.config.output_path + "elos.csv" , mode='w', newline="")
    csv_file = csv.writer(elo_res_f)
    csv_file.writerow([*model_info_names, "elo"])

    # record the elo results
    elos = [[*model_info(m), m.elo] for m in models]
    elos.sort(key=lambda x: x[3], x[2], reverse=True)
    csv_res = csv.writer(csv_file)
    csv_res.writerows(elos)

            
