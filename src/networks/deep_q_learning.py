import os
import numpy as np
import tensorflow as tf
import time

from .q_learning import QN


class DQN(QN):
    """
    Abstract class for Deep Q Learning
    """
    def __init__(self, *args, **kwargs):
        self.tf_graph = tf.Graph()

        # initialize this to prevent errors
        self.avg_reward = -50
        self.max_reward = -50
        self.std_reward = -50
        self.avg_q = -50
        self.max_q = -50
        self.std_q = -50
        self.eval_reward = -50


        super(DQN, self).__init__(*args, **kwargs)

    def add_placeholders_op(self):
        raise NotImplementedError


    def get_q_values_op(self, scope, reuse=False):
        """
        set Q values, of shape = (batch_size, num_actions)
        """
        raise NotImplementedError


    def add_update_target_op(self, q_scope, target_q_scope):
        """
        Update_target_op will be called periodically 
        to copy Q network to target Q network
    
        Args:
            q_scope: name of the scope of variables for q
            target_q_scope: name of the scope of variables for the target
                network
        """
        raise NotImplementedError


    def add_loss_op(self, q, target_q):
        """
        Set (Q_target - Q)^2
        """
        raise NotImplementedError


    def add_optimizer_op(self, scope):
        """
        Set training op wrt to loss for variable in scope
        """
        raise NotImplementedError


    def process_state(self, state):
        """
        Processing of state

        State placeholders are tf.uint8 for fast transfer to GPU
        Need to cast it to float32 for the rest of the tf graph.

        Args:
            state: node of tf graph of shape = (batch_size, height, width, nchannels)
                    of type tf.uint8.
                    if , values are between 0 and 255 -> 0 and 1
        """
        with self.tf_graph.as_default():
            state = tf.cast(state, tf.float32)
            state /= self.config.high

            return state

    def build(self):
        """
        Build model by adding all necessary variables
        """
        # add placeholders
        with self.tf_graph.as_default():
            self.add_placeholders_op()

            # compute Q values of state
            s = self.process_state(self.s)
            self.q = self.get_q_values_op(s, scope="q", reuse=False)

            # compute Q values of next state
            sp = self.process_state(self.sp)
            self.target_q = self.get_q_values_op(sp, scope="target_q", reuse=False)

            # add update operator for target network
            self.add_update_target_op("q", "target_q")

            # add square loss
            self.add_loss_op(self.q, self.target_q)

            # add optmizer for the main networks
            self.add_optimizer_op("q")


    def initialize(self):
        """
        Assumes the graph has been constructed
        Creates a tf Session and run initializer of variables
        """
        # create tf session
        with self.tf_graph.as_default():
            
            # limit GPU memory usage.
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            # config.gpu_options.per_process_gpu_memory_fraction = 0.2

            
            self.sess = tf.Session(config=config)

            # tensorboard stuff
            self.add_summary()

            # initiliaze all variables
            init = tf.global_variables_initializer()
            self.sess.run(init)

            # synchronise q and target_q networks
            self.sess.run(self.update_target_op)

            # for saving networks weights
            self.saver = tf.train.Saver(max_to_keep=0)

    def load(self, src):
        """
        load an existing network. Takes the place of the initialize step.
        """
        # create tf session
        with self.tf_graph.as_default():
            
            # limit GPU memory usage.
            config = tf.ConfigProto()
            # config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.8

            
            self.sess = tf.Session(config=config)

            # tensorboard stuff
            self.add_summary()

            # initiliaze all variables
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, src)

            # synchronise q and target_q networks
            self.sess.run(self.update_target_op)

            # for saving networks weights


       
    def add_summary(self):
        """
        Tensorboard stuff
        """
        # extra placeholders to log stuff from python
        with self.tf_graph.as_default():
            self.avg_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_reward")
            self.max_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="max_reward")
            self.std_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="std_reward")

            self.avg_q_placeholder  = tf.placeholder(tf.float32, shape=(), name="avg_q")
            self.max_q_placeholder  = tf.placeholder(tf.float32, shape=(), name="max_q")
            self.std_q_placeholder  = tf.placeholder(tf.float32, shape=(), name="std_q")

            self.eval_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="eval_reward")

            # add placeholders from the graph
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("grads norm", self.grad_norm)

            # extra summaries from python -> placeholders
            tf.summary.scalar("Avg Reward", self.avg_reward_placeholder)
            tf.summary.scalar("Max Reward", self.max_reward_placeholder)
            tf.summary.scalar("Std Reward", self.std_reward_placeholder)

            tf.summary.scalar("Avg Q", self.avg_q_placeholder)
            tf.summary.scalar("Max Q", self.max_q_placeholder)
            tf.summary.scalar("Std Q", self.std_q_placeholder)

            tf.summary.scalar("Eval Reward", self.eval_reward_placeholder)

            # logging
            self.merged = tf.summary.merge_all()
            self.file_writer = tf.summary.FileWriter(self.config.output_path,
                                                    self.sess.graph)



    def save(self, step=None):
        """
        Saves session
        """
        with self.tf_graph.as_default():
            if not os.path.exists(self.config.model_output):
                os.makedirs(self.config.model_output)

            self.saver.save(self.sess, self.config.model_output + "model", global_step=step)


    def get_best_action(self, state):
        """
        Return best action

        Args:
            state: 4 consecutive observations from gym
        Returns:
            action: (int)
            action_values: (np array) q values for all actions
        """
        with self.tf_graph.as_default():
            feed_dict = {self.s: [state]}
            action_values = self.sess.run(self.q, feed_dict=feed_dict)[0]
            return np.argmax(action_values), action_values


    def update_step(self, t, replay_buffer, lr):
        """
        Performs an update of parameters by sampling from replay_buffer

        Args:
            t: number of iteration (episode and move)
            replay_buffer: ReplayBuffer instance .sample() gives batches
            lr: (float) learning rate
        Returns:
            loss: (Q - Q_target)^2
        """

        with self.tf_graph.as_default():
            s_batch, a_batch, r_batch, sp_batch, done_mask_batch = replay_buffer.sample(
                self.config.batch_size)


            fd = {
                # inputs
                self.s: s_batch,
                self.a: a_batch,
                self.r: r_batch,
                self.sp: sp_batch,
                self.done_mask: done_mask_batch,
                self.lr: lr,
                # extra info
                self.avg_reward_placeholder: self.avg_reward,
                self.max_reward_placeholder: self.max_reward,
                self.std_reward_placeholder: self.std_reward,
                self.avg_q_placeholder: self.avg_q,
                self.max_q_placeholder: self.max_q,
                self.std_q_placeholder: self.std_q,
                self.eval_reward_placeholder: self.eval_reward,
            }

            loss_eval, grad_norm_eval, summary, _ = self.sess.run([self.loss, self.grad_norm,
                                                     self.merged, self.train_op], feed_dict=fd)


            # tensorboard stuff
            self.file_writer.add_summary(summary, t)

            return loss_eval, grad_norm_eval


    def update_target_params(self):
        """
        Update parametes of Q' with parameters of Q
        """

        with self.tf_graph.as_default():
            self.sess.run(self.update_target_op)

