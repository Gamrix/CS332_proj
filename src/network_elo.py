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
        self.config = config
        self.env = env

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
class ThinModel:
    def __init__(self, model_class, model_dir, num, name):
        self.model_class = model_class
        self.model_dir = model_dir
        self.num = int(num)
        self.name = name
        self.elo = 1000

def model_info(model):
    try :
        m_class = model.model_class
    except Exception:
        m_class = model.__class__.__name__
    return m_class, model.model_dir, model.num, model.name

model_info_names = ["model", "model_dir", "num", "name"]

def load_games_res(game_results):
    loaded_results = []
    model_dict= {}
    def find_model(args):
        model_tuple = tuple(args)
        print(model_tuple)
        if model_tuple not in model_dict:
            model_dict[model_tuple] = ThinModel(*model_tuple)
        return model_dict[model_tuple]

    for res in game_results:
        m0 = find_model(res[:4])
        m1 = find_model(res[4:8])
        loaded_results.append([m0, m1, int(res[8]), int(res[9])])
    return list(model_dict.values()), loaded_results

    # load up the results


def run_games():
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
    csv_res.writerow([*[m + "_0" for m in model_info_names],
                      *[m + "_1" for m in model_info_names],
                      "win_0", "win_1"])


    def enumerate_models(model_dir, model_nums, name):
        models = []
        for m in model_nums:
            cur_model = dqns.AdvantageQN(env, g_config, name=name)
            cur_model.num = m
            cur_model.elo = 1000
            cur_model.model_dir = model_dir
            cur_model.load(model_dir + "-" + str(m))
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

        model_dir = "trained_models/{}/model.weights/model"

        single_play = enumerate_models(model_dir.format("02_2204/SingleADV"), [4011594, 4764484], "Single")        
        self_play0A = enumerate_models(model_dir.format("02_2205/Adv_A"), [4006694, 4757864], "Adv0A")        
        self_play0B = enumerate_models(model_dir.format("02_2205/Adv_B"), [4006694, 4757864], "Adv0B")        
        self_play1A = enumerate_models(model_dir.format("02_2209/Adv_A"), [4005221, 4756947] , "Adv1A")        
        self_play1B = enumerate_models(model_dir.format("02_2209/Adv_B"), [4005221, 4756947] , "Adv1B")        
        self_play0 = self_play0A + self_play0B
        self_play1 = self_play1A + self_play1B

        compatable_with(single_play, [self_play0, self_play1])
        compatable_with(self_play0, [single_play, self_play1])
        compatable_with(self_play1, [single_play, self_play0])

        nonlocal models
        nonlocal rounds
        models = single_play + self_play0 + self_play1
        rounds = 5

    # Which environment to run
    first_run()

    # now to actually score the games 
    results = []

    for i in range(rounds):
        for m0, m1 in pairs:
            score_0, score_1 = evaluator.evaluate(m0, m1)
            update_elo(m0, m1, 30, score_0, score_1)
            info = [*model_info(m0), *model_info(m1), score_0, score_1]
            print(info)
            print(m0.elo, m1.elo)
            csv_res.writerow(info)
            results.append([m0, m1, score_0, score_1])
    return results
    
def calculate_elo(models, results, res_dir):
    k_schedule = [30, 30, 30, 10, 10, 10, 3, 3, 1, 1, .1, .1] # decay the K over time to smooth scores 
    for k in k_schedule:
        for m0, m1, score_0, score_1 in random.sample(results, len(results)):
            update_elo(m0, m1, k, score_0, score_1)
    
    elo_res_f = open(res_dir + "elos.csv" , mode='w', newline="")
    csv_file = csv.writer(elo_res_f)
    csv_file.writerow([*model_info_names, "elo"])

    # record the elo results
    elos = [[*model_info(m), m.elo] for m in models]
    elos.sort(key=(lambda x: (x[4], x[2])), reverse=True)
    csv_file.writerows(elos)


if __name__ == '__main__':
    r_dir = "elo_scores/1203_1606/"
    scores_f = open(r_dir + "results.csv", newline='')
    c_reader = csv.reader(scores_f)
    model_data = list(c_reader)[1:]
    models, results = load_games_res(model_data)
    calculate_elo(models, results, r_dir)
