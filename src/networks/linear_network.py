
import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.general import get_logger
from utils.test_env import EnvTest
from networks.deep_q_learning import DQN


class Linear(DQN):
    """
    Implement Fully Connected with Tensorflow
    """
    def __init__(self, *args, **kwargs):
        super(Linear, self).__init__(*args, **kwargs)


    def add_placeholders_op(self):
        """
        Adds placeholders to the graph

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model
        """
        # this information might be useful
        # here, typically, a state shape is (80, 80, 1)
        state_shape = list(self.env.observation_space.shape)

        ##############################################################
        """
        TODO: add placeholders:
              Remember that we stack 4 consecutive frames together, ending up with an input of shape
              (80, 80, 4).
               - self.s: batch of states, type = uint8
                         shape = (batch_size, img height, img width, nchannels x config.state_history)
               - self.a: batch of actions, type = int32
                         shape = (batch_size)
               - self.r: batch of rewards, type = float32
                         shape = (batch_size)
               - self.sp: batch of next states, type = uint8
                         shape = (batch_size, img height, img width, nchannels x config.state_history)
               - self.done_mask: batch of done, type = bool
                         shape = (batch_size)
                         note that this placeholder contains bool = True only if we are done in 
                         the relevant transition
               - self.lr: learning rate, type = float32
        
        (Don't change the variable names!)
        
        HINT: variables from config are accessible with self.config.variable_name
              Also, you may want to use a dynamic dimension for the batch dimension.
              Check the use of None for tensorflow placeholders.

              you can also use the state_shape computed above.
        """
        ##############################################################
        ################YOUR CODE HERE (6-15 lines) ##################
        input_shape = (None, state_shape[0], state_shape[1], state_shape[2] * self.config.state_history)
        self.state_shape = input_shape
        self.s = tf.placeholder(tf.uint8, input_shape)
        self.sp = tf.placeholder(tf.uint8, input_shape)
        self.a = tf.placeholder(tf.int32, (None,))
        self.r = tf.placeholder(tf.float32, (None,))
        self.done_mask = tf.placeholder(tf.bool, (None,))
        self.lr = tf.placeholder(tf.float32, shape=tuple())

        ##############################################################
        ######################## END YOUR CODE #######################


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

        out = state

        ##############################################################
        """
        TODO: implement a fully connected with no hidden layer (linear
            approximation) using tensorflow. In other words, if your state s
            has a flattened shape of n, and you have m actions, the result of 
            your computation sould be equal to
                W s where W is a matrix of shape m x n

        HINT: you may find tensorflow.contrib.layers useful (imported)
              make sure to understand the use of the scope param

              you can use any other methods from tensorflow
              you are not allowed to import extra packages (like keras,
              lasagne, cafe, etc.)
        """
        ##############################################################
        ################ YOUR CODE HERE - 2-3 lines ##################
        shape= self.state_shape
        # s_norm = tf.cast(self.s, dtype=tf.float32) / 255
        s_norm = state
        # new_shape = (-1, shape[1] * shape[2] * shape[3])
        # s_reshape = tf.reshape(s_norm, shape=new_shape)
        s_reshape = layers.flatten(s_norm)
        out = layers.fully_connected(s_reshape, num_actions, scope=scope, reuse=reuse, activation_fn=None)
        ##############################################################
        ######################## END YOUR CODE #######################

        return out


    def add_update_target_op(self, q_scope, target_q_scope):
        """
        update_target_op will be called periodically 
        to copy Q network weights to target Q network

        Remember that in DQN, we maintain two identical Q networks with
        2 different set of weights. In tensorflow, we distinguish them
        with two different scopes. One for the target network, one for the
        regular network. If you're not familiar with the scope mechanism
        in tensorflow, read the docs
        https://www.tensorflow.org/programmers_guide/variable_scope

        Periodically, we need to update all the weights of the Q network 
        and assign them with the values from the regular network. Thus,
        what we need to do is to build a tf op, that, when called, will 
        assign all variables in the target network scope with the values of 
        the corresponding variables of the regular network scope.
    
        Args:
            q_scope: (string) name of the scope of variables for q
            target_q_scope: (string) name of the scope of variables
                        for the target network
        """
        ##############################################################
        """
        TODO: add an operator self.update_target_op that assigns variables
            from target_q_scope with the values of the corresponding var 
            in q_scope

        HINT: you may find the following functions useful:
            - tf.get_collection
            - tf.assign
            - tf.group

        (be sure that you set self.update_target_op)
        """
        ##############################################################
        ################### YOUR CODE HERE - 5-10 lines #############
        update_ops = []
        new_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, q_scope)
        target_vars= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, target_q_scope)
        for src, dst in zip(new_vars, target_vars):
            update_ops.append(tf.assign(dst, src))
            # tf.summary.histogram(src.name, src)
            # tf.summary.histogram(dst.name, dst)
            # This is a place where we can get the weight matrix...

        self.update_target_op = tf.group(*update_ops)
        ##############################################################
        ######################## END YOUR CODE #######################


    def add_loss_op(self, q, target_q):
        """
        Sets the loss of a batch, self.loss is a scalar

        Args:
            q: (tf tensor) shape = (batch_size, num_actions)
            target_q: (tf tensor) shape = (batch_size, num_actions)
        """
        # you may need this variable
        ##############################################################
        """
        TODO: The loss for an example is defined as:
                Q_samp(s) = r if done
                          = r + gamma * max_a' Q_target(s', a')
                loss = (Q_samp(s) - Q(s, a))^2 

              You need to compute the average of the loss over the minibatch
              and store the resulting scalar into self.loss

        HINT: - config variables are accessible through self.config
              - you can access placeholders like self.a (for actions)
                self.r (rewards) or self.done_mask for instance
              - you may find the following functions useful
                    - tf.cast
                    - tf.reduce_max / reduce_sum
                    - tf.one_hot
                    - ...

        (be sure that you set self.loss)
        """
        ##############################################################
        ##################### YOUR CODE HERE - 4-5 lines #############
        done_float = tf.cast(self.done_mask, dtype=tf.float32)
        q_samp = self.r + (1 - done_float) * self.config.gamma * tf.reduce_max(target_q, axis=1)

        # diff = (tf.reshape(q_samp, (-1,1)) - q) * tf.one_hot(self.a, depth=num_actions)   # maybe this is wrong
        diff = q_samp - tf.reduce_sum(q * tf.one_hot(self.a, depth=self.action_space), axis=1)
        self.loss = tf.reduce_sum(diff * diff)
        # self.loss = tf.Print(self.loss,[self.loss], name="Loss_calc", summarize=20)
        # for elem in [ diff, q_samp, q, target_q, self.r, self.a]:
            # self.loss = tf.Print(self.loss, [elem], summarize=20)
        ##############################################################
        ######################## END YOUR CODE #######################


    def add_optimizer_op(self, scope):
        """
        Set self.train_op and self.grad_norm
        """

        ##############################################################
        """
        TODO: 1. get Adam Optimizer (remember that we defined self.lr in the placeholders
                section)
              2. compute grads wrt to variables in scope for self.loss
              3. clip the grads by norm with self.config.clip_val if self.config.grad_clip
                is True
              4. apply the gradients and store the train op in self.train_op
               (sess.run(train_op) must update the variables)
              5. compute the global norm of the gradients and store this scalar
                in self.grad_norm

        HINT: you may find the following functions useful
            - tf.get_collection
            - optimizer.compute_gradients
            - tf.clip_by_norm
            - optimizer.apply_gradients
            - tf.global_norm
             
             you can access config variable by writing self.config.variable_name

        (be sure that you set self.train_op and self.grad_norm)
        """
        ##############################################################
        #################### YOUR CODE HERE - 8-12 lines #############
        # global variables
        # tf.get_collection()
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

        # note self.config.clip_grad
        grads_val = optimizer.compute_gradients(self.loss, var_list=vars)
        grads, val = zip(*grads_val)
        self.grad_norm = tf.global_norm(grads)
        normed_grads = [tf.clip_by_norm(g, self.config.clip_val) for g in grads]
        self.train_op = optimizer.apply_gradients(zip(normed_grads, val))

        # self.summ = tf.summary.merge_all()
        # self.writer = tf.summary.FileWriter("log/tensorboard")

        ##############################################################
        ######################## END YOUR CODE #######################
    


if __name__ == '__main__':
    from configs.q2_linear import config
    from train_schedule import LinearExploration, LinearSchedule
    env = EnvTest((5, 5, 1))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = Linear(env, config)
    model.run(exp_schedule, lr_schedule)
