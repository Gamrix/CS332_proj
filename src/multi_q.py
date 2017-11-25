import gym
import sys
import numpy as np
from collections import deque

from utils.general import get_logger, Progbar, export_plot
from train_schedule import LinearExploration, LinearSchedule
from utils.preprocess import greyscale
from utils.wrappers import PreproWrapper, MaxAndSkipEnv

import config
import dqns


class SelfPlayTrainer(object):
    def __init__(self, model_0, model_1, env, config):
        self.model_0 = model_0
        self.model_1 = model_1
        self.env = env
        self.config = config

        self.logger = get_logger(config.log_path)

    def record(self):
        env = gym.make(self.config.env_name)
        env = gym.wrappers.Monitor(env, self.config.record_path, video_callable=lambda x: True, resume=True)
        env = MaxAndSkipEnv(env, skip=self.config.skip_frame)
        env = PreproWrapper(env, prepro=greyscale, shape=(80, 80, 1),
                            overwrite_render=self.config.overwrite_render)
        self.evaluate(env, 1)

    def run_parallel_models(self, exp_schedule, lr_schedule, train_0, train_1):
        self.model_0.initialize()
        self.model_1.initialize()

        if config.record():
            self.record()  # record one at beginning

        self.train(exp_schedule, lr_schedule, train_0, train_1)

        if self.config.record:
            self.record()  # record one at end

    def evaluate(self, env=None, num_episodes=None):
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
        self.model_0.train_init()
        self.model_1.train_init()

        for i in range(num_episodes):
            total_reward = 0
            state = env.reset()
            while True:
                if self.config.render_test: env.render()

                action_0 = self.model_0.train_step_pre(state, exp_schedule)
                action_1 = self.model_1.train_step_pre(state[:, ::-1], exp_schedule)

                # perform action in env
                new_state, reward, done, info = self.env.step((action_0, action_1))

                self.model_0.train_step_post(reward, done, 0, lr_schedule, False)
                self.model_1.train_step_post(-reward, done, 0, lr_schedule, False)

                # store last state in buffer
                state = new_state

                # count reward
                total_reward += reward
                if done:
                    break

            # updates to perform at the end of an episode
            rewards.append(total_reward)

        avg_reward = np.mean(rewards)
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))

        if num_episodes > 1:
            msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
            self.logger.info(msg)

        return avg_reward


    def train(self, exp_schedule, lr_schedule, train_0, train_1):
        """
        Performs training of Q

        Args:
            exp_schedule: Exploration instance s.t.
                exp_schedule.get_action(best_action) returns an action
            lr_schedule: Schedule for learning rate
        """

        # initialize replay buffer and variables
        rewards = deque(maxlen=self.config.num_episodes_test)
        # self.init_averages()

        t = last_eval = last_record = 0 # time control of nb of steps
        scores_eval = [] # list of scores computed at iteration time
        scores_eval += [self.evaluate()]

        prog = Progbar(target=self.config.nsteps_train)
        self.model_0.train_init()
        self.model_1.train_init()

        # interact with environment
        while t < self.config.nsteps_train:
            total_reward = 0
            state = self.env.reset()
            while True:
                t += 1
                last_eval += 1
                last_record += 1
                if self.config.render_train: self.env.render()

                action_0 = self.model_0.train_step_pre(state, exp_schedule)
                action_1 = self.model_1.train_step_pre(state[:, ::-1], exp_schedule)

                # perform action in env
                new_state, reward, done, info = self.env.step((action_0, action_1))

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

            if (t > self.config.learning_start) and (last_eval > self.config.eval_freq):
                # evaluate our policy
                last_eval = 0
                print("")
                scores_eval += [self.evaluate()]

            if (t > self.config.learning_start) and self.config.record and (last_record > self.config.record_freq):
                self.logger.info("Recording...")
                last_record =0
                self.record()

        # last words
        self.logger.info("- Training done.")
        self.model_0.save() # save the models
        self.model_1.save() # save the models
        scores_eval += [self.evaluate()]
        export_plot(scores_eval, "Scores", self.config.plot_output)

if __name__ == '__main__':
    # make env
    env = gym.make(config.env_name)
    env = MaxAndSkipEnv(env, skip=config.skip_frame)
    env = PreproWrapper(env, prepro=greyscale, shape=(80, 80, 1),
                        overwrite_render=config.overwrite_render)

    # exploration strategy
    # you may want to modify this schedule
    exp_schedule = LinearExploration(env, config.eps_begin,
                                     config.eps_end, config.eps_nsteps)

    # you may want to modify this schedule
    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
                                  config.lr_nsteps)

    # train model
    model_0 = dqns.AdvantageQN(env, config)
    model_1 = dqns.AdvantageQN(env, config)
    trainer = SelfPlayTrainer(model_0, model_1, env, config)
    trainer.train(exp_schedule, lr_schedule, train_0=True, train_1=True)

