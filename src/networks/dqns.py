import gym
from utils.preprocess import greyscale
from utils.wrappers import PreproWrapper, MaxAndSkipEnv

import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.general import get_logger
from utils.test_env import EnvTest
from train_schedule import LinearExploration, LinearSchedule
from linear_network import Linear



class AdvantageQN(Linear):
    """
    Going beyond - implement your own Deep Q Network to find the perfect
    balance between depth, complexity, number of parameters, etc.
    You can change the way the q-values are computed, the exploration
    strategy, or the learning rate schedule. You can also create your own
    wrapper of environment and transform your input to something that you
    think we'll help to solve the task. Ideally, your network would run faster
    than DeepMind's and achieve similar performance!

    You can also change the optimizer (by overriding the functions defined
    in TFLinear), or even change the sampling strategy from the replay buffer.

    If you prefer not to build on the current architecture, you're welcome to
    write your own code.

    You may also try more recent approaches, like double Q learning
    (see https://arxiv.org/pdf/1509.06461.pdf) or dueling networks 
    (see https://arxiv.org/abs/1511.06581), but this would be for extra
    extra bonus points.
    """
    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, img height, img width, nchannels)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        # this information might be useful
        with self.tf_graph.as_default():
            out = state
            with tf.variable_scope(scope, reuse):
                hid1 = layers.conv2d(state, num_outputs=32, kernel_size=8, stride=4, padding="SAME", scope="c1")
                hid2 = layers.conv2d(hid1, num_outputs=64, kernel_size=4, stride=2, padding="SAME", scope="c2")
                hid3 = layers.conv2d(hid2, num_outputs=64, kernel_size=3, stride=1, padding="SAME", scope="c3")
                hid3_flat = layers.flatten(hid3)

                fc_advantage = layers.fully_connected(hid3_flat, num_outputs=256, scope="FC1")
                advantage = layers.fully_connected(fc_advantage, self.action_space, scope="FC2", activation_fn=None)

                fc_value = layers.fully_connected(hid3_flat, num_outputs=64, scope="FC_V1")
                value = layers.fully_connected(fc_value, num_outputs=1, scope="FC_V2", activation_fn=None)

                norm_advantage = advantage - tf.reshape(tf.reduce_mean(advantage, axis=1), (-1, 1))
                out = tf.reshape(value, (-1,1)) + norm_advantage
            return out


"""
Use a different architecture for the Atari game. Please report the final result.
Feel free to change the configuration. If so, please report your hyperparameters.
"""
if __name__ == '__main__':
    import config
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
    model = AdvantageQN(env, config)
    model.run(exp_schedule, lr_schedule)
