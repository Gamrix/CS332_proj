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

from PIL import Image
from io import BytesIO


def display_img(data):
    p_img = Image.fromarray(np.squeeze(data), 'L') # Because images are grayscale
    p_img.show()


class SelfPlayTrainer(object):
    def __init__(self, model_0, model_1, env, config):
        self.model_0 = model_0
        self.model_1 = model_1
        self.env = env
        self.config = config
        name =  time.strftime("_%m%d_%H%M")

        config.output_path = config.output_path.format(name)
        config.model_output = config.model_output.format(name)
        config.log_path = config.log_path.format(name)
        config.plot_output = config.plot_output.format(name)
        config.record_path = config.record_path.format(name)

        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)

        self.logger = get_logger(config.log_path.format(time.strftime("_%m%d_%H%M")))

    def initialize(self):
        self.model_0.initialize()
        self.model_1.initialize()

    def record(self, exp_schedule=None):
        self.logger.info("Recording training episode")
        # evaluate with no exp
        env = gym.make(self.config.env_name)
        env = gym.wrappers.Monitor(env, self.config.record_path, video_callable=lambda x: True, resume=True)
        env = MaxAndSkipEnv(env, skip=self.config.skip_frame)
        env = PreproWrapper(env, prepro=greyscale, shape=(80, 80, 1),
                            overwrite_render=self.config.overwrite_render)
        self.evaluate(env, 1)

        # evaluate with Exp
        env = gym.make(self.config.env_name)
        env = gym.wrappers.Monitor(env, self.config.record_path, video_callable=lambda x: True, resume=True)
        env = MaxAndSkipEnv(env, skip=self.config.skip_frame)
        env = PreproWrapper(env, prepro=greyscale, shape=(80, 80, 1),
                            overwrite_render=self.config.overwrite_render)
        self.evaluate(env, 1, exp_schedule=exp_schedule)

    def run_parallel_models(self, exp_schedule, lr_schedule, train_0, train_1):
        self.model_0.initialize()
        self.model_1.initialize()

        if True:
            self.record(exp_schedule)  # record one at beginning

        self.train(exp_schedule, lr_schedule, train_0, train_1)

        if True:
            self.record(exp_schedule)  # record one at end
    
    def get_new_ball(self, env):
        while True:
            # we need to account for stochasticity of the env, 
            # and we need to account for pong not letting the user 
            # immediately launch the ball again
            new_state, reward, done, info = env.step(actions.FIRE)  # launch a new ball
            if done:
                return None 
            # center = new_state[60:190, 20:140]
            center = new_state[10:70, 10:70]  # because it is reshaped to 80:80:3
            # print(center.max())
            # print("call new state")
            #print(new_state.shape)
            if center.max() > 200:
                # The ball is now back in the field
                return new_state
            
            
    def train(self, exp_schedule, lr_schedule, train_0, train_1, env=None):
        """
        Performs training of Q

        Args:
            exp_schedule: Exploration instance s.t.
                exp_schedule.get_action(best_action) returns an action
            lr_schedule: Schedule for learning rate
        """
        if env is None:
            env = self.env

        # initialize replay buffer and variables
        rewards = deque(maxlen=self.config.num_episodes_test)
        rewardsB = deque(maxlen=self.config.num_episodes_test)
        self.model_0.rewards = rewards
        self.model_1.rewards = rewardsB
        # self.init_averages()

        t = last_eval = last_record = 0 # time control of nb of steps
        scores_eval = [] # list of scores computed at iteration time
        scores_eval += [self.evaluate()]

        prog = Progbar(target=self.config.nsteps_train)
        self.model_0.train_init()
        self.model_1.train_init()

        # next_fire_B = False

        # interact with environment
        while t < self.config.nsteps_train:
            total_reward = 0
            state = self.env.reset()
            # need_new_ball = False
            while True:
                t += 1
                last_eval += 1
                last_record += 1
                if self.config.render_train: env.render()

                action_0 = self.model_0.train_step_pre(state, exp_schedule)
                action_1 = self.model_1.train_step_pre(state[:, ::-1], exp_schedule)
                cur_action = actions.trans(action_0, action_1)

                # now calculate if we still need a ball
                """
                if need_new_ball:
                    center = new_state[10:70, 10:70]  # because it is reshaped to 80:80:3
                    if center.max() > 200:
                        need_new_ball = False
                    else:
                        if next_fire_B
                """


                # display_img(state)

                # perform action in env
                new_state, reward, done, info =env.step(cur_action)

                # print("Reward", reward)

                # Problem 
                loss_e0, grad_e0 = self.model_0.train_step_post(reward, done, t, lr_schedule, train_0)
                self.model_1.train_step_post(-reward, done, t, lr_schedule, train_1)
                state = new_state

                # logging stuff
                if ((t > self.config.learning_start) and (t % self.config.log_freq == 0) and
                        (t % self.config.learning_freq == 0)):
                    # self.update_averages(rewards, max_q_values, q_values, scores_eval)
                    exp_schedule.update(t)
                    lr_schedule.update(t)
                    if len(rewards) > 0:
                        prog.update(t + 1, exact=[("Loss", loss_e0),
                                                  ("Avg R", np.mean(rewards)),
                                                  ("Max R", np.max(rewards)),
                                                  ("Min R", np.min(rewards)),
                                                  ("eps", exp_schedule.epsilon),
                                                  ("Grads", grad_e0),
                                                  ("Max Q", np.mean(self.model_0.max_q_values)),
                                                  ("lr", lr_schedule.epsilon)])

                elif (t < self.config.learning_start) and (t % self.config.log_freq == 0):
                    sys.stdout.write("\rPopulating the memory {}/{}...".format(t,
                                                                               self.config.learning_start))
                    sys.stdout.flush()


                # count reward
                total_reward += reward
                if done or t >= self.config.nsteps_train:
                    break


            # updates to perform at the end of an episode
            rewards.append(total_reward)
            rewardsB.append(-total_reward)

            if (t > self.config.learning_start) and (last_eval > self.config.eval_freq):
                # evaluate our policy
                last_eval = 0
                print("")
                scores_eval += [self.evaluate()]

            if (t > self.config.learning_start) and self.config.record and (last_record > self.config.record_freq):
                self.logger.info("Recording...")
                last_record =0
                self.record(exp_schedule)
                self.model_0.save(t) # save the models
                self.model_1.save(t) # save the models

        # last words
        self.logger.info("- Training done.")
        self.model_0.save() # save the models
        self.model_1.save() # save the models
        scores_eval += [self.evaluate()]
        export_plot(scores_eval, "Scores", self.config.plot_output)

    def evaluate(self, env=None, num_episodes=None, exp_schedule=None):
        """
        Evaluation with same procedure as the training
        """
        # log our activity only if default call
        if num_episodes is None:
            self.logger.info("Evaluating...")

        # arguments defaults
        if num_episodes is None:
            num_episodes = self.config.num_episodes_test

        if env is None:
            env = self.env

        # replay memory to play
        rewards = []
        
        # keep the replay buffer alive
        try:
            r0 = self.model_0.replay_buffer
            r1 = self.model_1.replay_buffer
            has_replay = True
        except Exception:
            has_replay = False
        

        self.model_0.train_init()
        self.model_1.train_init()

        for i in range(num_episodes):
            total_reward = 0
            state = env.reset()
            while True:
                if self.config.render_test: env.render()

                action_0 = self.model_0.train_step_pre(state, exp_schedule)
                action_1 = self.model_1.train_step_pre(state[:, ::-1], exp_schedule)
                cur_action = actions.trans(action_0, action_1)

                # perform action in env
                new_state, reward, done, info =env.step(cur_action)

                total_reward += reward

                # need to start another game
                self.model_0.train_step_post(reward, done, 0, None, False)
                self.model_1.train_step_post(-reward, done, 0, None, False)

                if done:
                    break

                if reward !=0:
                    state = self.get_new_ball(env)
                    if state is None:
                        break # alternative done state
                # store last state in buffer
                state = new_state


            # updates to perform at the end of an episode
            rewards.append(total_reward)

        avg_reward = np.mean(rewards)
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))

        if num_episodes > 1:
            msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
            self.logger.info(msg)
        
        if has_replay:
            self.model_0.replay_buffer = r0
            self.model_1.replay_buffer = r1

        return avg_reward


def main():
    import config
    # make env
    g_config = config.config()
    g_config.env_name = "Pong2p-v0"
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
    model_0 = dqns.AdvantageQN(env, config.config(), name="Adv_A")
    model_1 = dqns.AdvantageQN(env, config.config(), name="Adv_B")
    trainer = SelfPlayTrainer(model_0, model_1, env, g_config)
    trainer.run_parallel_models(exp_schedule, lr_schedule, True, True)


if __name__ == '__main__':
    main()