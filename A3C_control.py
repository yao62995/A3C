#!/usr/bin/python
#  -*- coding: utf-8 -*-
# author: yao62995@gmail.com

import re
import gym
import signal
import threading
import scipy.signal
from collections import deque
from tensorflow.python.ops.rnn_cell import BasicLSTMCell

from common import *

tf.app.flags.DEFINE_string("game", "Breakout-v0", "gym environment name")
tf.app.flags.DEFINE_string("train_dir", "./models/experiment0/", "gym environment name")
tf.app.flags.DEFINE_integer("gpu", 0, "gpu id")
tf.app.flags.DEFINE_bool("use_lstm", False, "use LSTM layer")

tf.app.flags.DEFINE_integer("t_max", 32, "episode max time step")
tf.app.flags.DEFINE_integer("t_train", 1e4, "train max time step")
tf.app.flags.DEFINE_integer("t_test", 1e4, "test max time step")
tf.app.flags.DEFINE_integer("jobs", 8, "parallel running thread number")

tf.app.flags.DEFINE_integer("frame_skip", 1, "number of frame skip")
tf.app.flags.DEFINE_integer("frame_seq", 3, "number of frame sequence")

tf.app.flags.DEFINE_float("learn_rate", 5e-4, "param of smooth")
tf.app.flags.DEFINE_float("eps", 1e-8, "param of smooth")
tf.app.flags.DEFINE_float("entropy_beta", 1e-4, "param of policy entropy weight")
tf.app.flags.DEFINE_float("gamma", 0.95, "discounted ratio")

tf.app.flags.DEFINE_float("train_step", 0, "train step. unchanged")

flags = tf.app.flags.FLAGS


class ControlEnv(object):
    def __init__(self, env):
        self.env = env
        self.frame_skip = flags.frame_skip
        self.frame_seq = flags.frame_seq
        # local variables
        self.state_dim = self.env.observation_space.shape[0]
        self.state = np.zeros(self.state_dim, dtype=np.float32)

    @property
    def state_shape(self):
        return self.env.observation_space.shape[0] * self.frame_seq

    @property
    def action_dim(self):
        return self.env.action_space.n

    def reset_env(self):
        obs = self.env.reset()
        self.state[:] = 0
        self.state[-self.state_dim] = obs
        return self.state

    def forward_action(self, action):
        obs, reward, done = None, None, None
        for _ in xrange(self.frame_skip):
            obs, reward, done, _ = self.env.step(action)
            if done:
                break
        self.state = np.append(self.state[self.state_dim:], obs)
        return self.state, reward, done


class A3CNet(object):
    """
        1. In continuous control, policy network and value network do not share any parameters.
        2. In continuous control, output of actor network is normal distribution.
    """

    def __init__(self, state_dim, action_dim, scope):
        with tf.device("/gpu:%d" % flags.gpu):
            # placeholder
            self.state = tf.placeholder(tf.float32, shape=[None, state_dim], name="state")  # (None, 84, 84, 4)
            self.action = tf.placeholder(tf.float32, shape=[None, action_dim], name="action")  # (None, actions)
            self.target_q = tf.placeholder(tf.float32, shape=[None])
            # policy parts
            with tf.variable_scope("%s_policy" % scope):
                pi_fc_1, self.pi_w1, self.pi_b1 = full_connect(self.state, (512, 256), "pi_fc1", with_param=True)
                pi_fc_2, self.pi_w2, self.pi_b2 = full_connect(pi_fc_1, (256, 256), "pi_fc2", with_param=True)
                pi_fc_3, self.pi_w3, self.pi_b3 = full_connect(pi_fc_2, (256, action_dim), "pi_fc3", activate=None,
                                                               with_param=True)
                self.policy_out = NetTools.batch_normalized(pi_fc_3, name="pi_out")

            # value parts
            with tf.variable_scope("%s_value" % scope):
                v_fc_1, self.v_w1, self.v_b1 = full_connect(self.state, (512, 256), "v_fc1", with_param=True)
                v_fc_2, self.v_w2, self.v_b2 = full_connect(v_fc_1, (256, 256), "v_fc2", with_param=True)
                v_fc_3, self.v_w3, self.v_b3 = full_connect(v_fc_2, (256, 1), "v_fc3", activate=None, with_param=True)
                self.value_out = tf.reshape(v_fc_3, [-1], name="v_out")
            # loss values
            with tf.op_scope([self.policy_out, self.value_out], "%s_loss" % scope):
                self.entropy = - (tf.log(2 * pi_fc_3 * self.policy_out + flags.eps) + 1) / 2
                time_diff = self.target_q - self.value_out
                self.value_loss = tf.reduce_sum(tf.square(time_diff))
                self.total_loss = self.value_loss + self.entropy * flags.entropy_beta

    def get_policy(self, sess, state):
        return sess.run(self.policy_out, feed_dict={self.state: [state]})[0]

    def get_value(self, sess, state):
        return sess.run(self.value_out, feed_dict={self.state: [state]})[0]

    def get_vars(self):
        return [self.pi_w1, self.pi_b1, self.pi_w2, self.pi_b2, self.pi_w3, self.pi_b3,
                self.v_w1, self.v_b1, self.v_w2, self.v_b2, self.v_w3, self.v_b3]


class A3CLSTMNet(object):
    def __init__(self, state_shape, action_dim, scope):

        class InnerLSTMCell(BasicLSTMCell):
            def __init__(self, num_units, forget_bias=1.0, input_size=None):
                BasicLSTMCell.__init__(self, num_units, forget_bias=forget_bias, input_size=input_size)
                self.matrix, self.bias = None, None

            def __call__(self, inputs, state, scope=None):
                """
                    Long short-term memory cell (LSTM).
                    implement from BasicLSTMCell.__call__
                """
                with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
                    # Parameters of gates are concatenated into one multiply for efficiency.
                    c, h = tf.split(1, 2, state)
                    concat = self.linear([inputs, h], 4 * self._num_units, True)

                    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
                    i, j, f, o = tf.split(1, 4, concat)

                    new_c = c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * tf.tanh(j)
                    new_h = tf.tanh(new_c) * tf.sigmoid(o)

                    return new_h, tf.concat(1, [new_c, new_h])

            def linear(self, args, output_size, bias, bias_start=0.0, scope=None):
                """
                    Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
                    implement from function of tensorflow.python.ops.rnn_cell.linear()
                """
                if args is None or (isinstance(args, (list, tuple)) and not args):
                    raise ValueError("`args` must be specified")
                if not isinstance(args, (list, tuple)):
                    args = [args]

                    # Calculate the total size of arguments on dimension 1.
                total_arg_size = 0
                shapes = [a.get_shape().as_list() for a in args]
                for shape in shapes:
                    if len(shape) != 2:
                        raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
                    if not shape[1]:
                        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
                    else:
                        total_arg_size += shape[1]

                # Now the computation.
                with tf.variable_scope(scope or "Linear"):
                    matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
                    if len(args) == 1:
                        res = tf.matmul(args[0], matrix)
                    else:
                        res = tf.matmul(tf.concat(1, args), matrix)
                    if not bias:
                        return res
                    bias_term = tf.get_variable(
                        "Bias", [output_size],
                        initializer=tf.constant_initializer(bias_start))
                    self.matrix = matrix
                    self.bias = bias_term
                return res + bias_term

        with tf.device("/gpu:%d" % flags.gpu):
            # placeholder
            self.state = tf.placeholder(tf.float32, shape=[None] + list(state_shape), name="state")  # (None, 84, 84, 4)
            self.action = tf.placeholder(tf.float32, shape=[None, action_dim], name="action")  # (None, actions)
            self.target_q = tf.placeholder(tf.float32, shape=[None])
            # policy parts
            with tf.variable_scope("%s_policy" % scope):
                pi_fc_1, self.pi_w1, self.pi_b1 = full_connect(self.state, (512, 256), "pi_fc1", with_param=True)
                pi_fc_2, self.pi_w2, self.pi_b2 = full_connect(pi_fc_1, (256, 256), "pi_fc2", with_param=True)
            # policy rnn parts
            with tf.variable_scope("%s_policy_rnn" % scope) as scope:
                h_flat1 = tf.reshape(pi_fc_2, (1, -1, 256))
                self.pi_lstm = InnerLSTMCell(256)
                self.pi_initial_lstm_state = tf.placeholder(tf.float32, shape=[1, self.pi_lstm.state_size])
                self.pi_sequence_length = tf.placeholder(tf.float32, [1])
                lstm_outputs, self.pi_lstm_state = tf.nn.dynamic_rnn(self.pi_lstm, h_flat1,
                                                                     initial_state=self.pi_initial_lstm_state,
                                                                     sequence_length=self.pi_sequence_length,
                                                                     time_major=False,
                                                                     scope=scope)
                lstm_outputs = tf.reshape(lstm_outputs, [-1, 256])
                pi_fc_3, self.pi_w3, self.pi_b3 = full_connect(lstm_outputs, (256, action_dim), "pi_fc3", activate=None,
                                                               with_param=True)
                self.policy_out = NetTools.batch_normalized(pi_fc_3, name="pi_out")

            # value parts
            with tf.variable_scope("%s_value" % scope):
                v_fc_1, self.v_w1, self.v_b1 = full_connect(self.state, (512, 256), "v_fc1", with_param=True)
                v_fc_2, self.v_w2, self.v_b2 = full_connect(v_fc_1, (256, 256), "v_fc2", with_param=True)
            # value rnn parts
            with tf.variable_scope("%s_value_rnn" % scope) as scope:
                h_flat2 = tf.reshape(v_fc_2, (1, -1, 256))
                self.v_lstm = InnerLSTMCell(256)
                self.v_initial_lstm_state = tf.placeholder(tf.float32, shape=[1, self.v_lstm.state_size])
                self.v_sequence_length = tf.placeholder(tf.float32, [1])
                lstm_outputs, self.v_lstm_state = tf.nn.dynamic_rnn(self.v_lstm, h_flat2,
                                                                    initial_state=self.v_initial_lstm_state,
                                                                    sequence_length=self.v_sequence_length,
                                                                    time_major=False,
                                                                    scope=scope)
                lstm_outputs = tf.reshape(lstm_outputs, [-1, 256])
                v_fc_3, self.v_w3, self.v_b3 = full_connect(lstm_outputs, (256, 1), "v_fc3", activate=None,
                                                            with_param=True)
                self.value_out = tf.reshape(v_fc_3, [-1], name="v_out")
            # loss values
            with tf.op_scope([self.policy_out, self.value_out], "%s_loss" % scope):
                self.entropy = - tf.reduce_mean(self.policy_out * tf.log(self.policy_out + flags.eps))
                time_diff = self.target_q - self.value_out
                # policy_prob = tf.reduce_sum(tf.log(tf.mul(self.policy_out, self.action)), reduction_indices=1)
                # self.policy_loss = - tf.reduce_sum(policy_prob * time_diff, reduction_indices=1)
                self.value_loss = tf.square(time_diff)
                self.total_loss = self.value_loss + self.entropy * flags.entropy_beta
        # lstm state
        self.pi_lstm_state_out = np.zeros((1, self.pi_lstm.state_size), dtype=np.float32)
        self.v_lstm_state_out = np.zeros((1, self.v_lstm.state_size), dtype=np.float32)

    def reset_lstm_state(self):
        self.pi_lstm_state_out = np.zeros((1, self.pi_lstm.state_size), dtype=np.float32)
        self.v_lstm_state_out = np.zeros((1, self.v_lstm.state_size), dtype=np.float32)

    def get_policy(self, sess, state):
        policy_out, self.pi_lstm_state_out = sess.run([self.policy_out, self.pi_lstm_state],
                                                      feed_dict={self.state: [state],
                                                                 self.pi_initial_lstm_state: self.pi_lstm_state,
                                                                 self.pi_sequence_length: [1]})
        return policy_out[0]

    def get_value(self, sess, state):
        value_out, self.v_lstm_state_out = sess.run([self.value_out, self.v_lstm_state],
                                                    feed_dict={self.state: [state],
                                                               self.v_initial_lstm_state: self.v_lstm_state,
                                                               self.v_sequence_length: [1]})[0]
        return value_out[0]

    def get_vars(self):
        return [self.pi_w1, self.pi_b1, self.pi_w2, self.pi_b2,
                self.pi_lstm.matrix, self.pi_lstm.bias, self.pi_w3, self.pi_b3,
                self.v_w1, self.v_b1, self.v_w2, self.v_b2,
                self.v_lstm.matrix, self.v_lstm.bias, self.v_w3, self.v_b3]


class A3CSingleThread(object):
    def __init__(self, thread_id, master):
        self.thread_id = thread_id
        self.env = ControlEnv(gym.make(flags.game))
        self.master = master
        # local network
        if flags.use_lstm:
            self.local_net = A3CLSTMNet(self.env.state_shape, self.env.action_dim, scope="local_net_%d" % thread_id)
        else:
            self.local_net = A3CNet(self.env.state_shape, self.env.action_dim, scope="local_net_%d" % thread_id)
        # sync network
        self.sync = self.sync_network(master.shared_net)
        # accumulate gradients
        self.accum_grads = self.create_accumulate_gradients()
        self.do_accum_grads_ops = self.do_accumulate_gradients()
        self.reset_accum_grads_ops = self.reset_accumulate_gradients()
        # collect summaries for debugging
        summaries = list()
        summaries.append(tf.scalar_summary("entropy_%d" % self.thread_id, self.local_net.entropy))
        summaries.append(tf.scalar_summary("value_loss_%d" % self.thread_id, self.local_net.value_loss))
        summaries.append(tf.scalar_summary("total_loss_%d" % self.thread_id, self.local_net.total_loss))
        # apply accumulated gradients
        with tf.device("/gpu:%d" % flags.gpu):
            self.apply_gradients = master.shared_opt.apply_gradients(
                zip(self.accum_grads, master.shared_net.get_vars()))
            self.summary_op = tf.merge_summary(summaries)

    def sync_network(self, source_net):
        sync_ops = []
        with tf.device("/gpu:%d" % flags.gpu):
            with tf.op_scope([], name="sync_ops_%d" % self.thread_id):
                for (target_var, source_var) in zip(source_net.get_vars(), self.local_net.get_vars()):
                    ops = tf.assign(target_var, source_var)
                    sync_ops.append(ops)
                return tf.group(*sync_ops, name="sync_group_%d" % self.thread_id)

    def create_accumulate_gradients(self):
        accum_grads = []
        with tf.device("/gpu:%d" % flags.gpu):
            with tf.op_scope([self.local_net], name="create_accum_%d" % self.thread_id):
                for var in self.local_net.get_vars():
                    zero = tf.zeros(var.get_shape().as_list(), dtype=var.dtype)
                    name = var.name.replace(":", "_") + "_accum_grad"
                    accum_grad = tf.Variable(zero, name=name, trainable=False)
                    accum_grads.append(accum_grad.ref())
                return accum_grads

    def do_accumulate_gradients(self):
        net = self.local_net
        accum_grad_ops = []
        with tf.device("/gpu:%d" % flags.gpu):
            with tf.op_scope([net], name="grad_ops_%d" % self.thread_id):
                var_refs = [v.ref() for v in net.get_vars()]
                grads = tf.gradients(net.total_loss, var_refs, gate_gradients=False,
                                     aggregation_method=None,
                                     colocate_gradients_with_ops=False)
            with tf.op_scope([], name="accum_ops_%d" % self.thread_id):
                for (grad, var, accum_grad) in zip(grads, net.get_vars(), self.accum_grads):
                    name = var.name.replace(":", "_") + "_accum_grad_ops"
                    accum_ops = tf.assign_add(accum_grad, grad, name=name)
                    accum_grad_ops.append(accum_ops)
                return tf.group(*accum_grad_ops, name="accum_group_%d" % self.thread_id)

    def reset_accumulate_gradients(self):
        net = self.local_net
        reset_grad_ops = []
        with tf.device("/gpu:%d" % flags.gpu):
            with tf.op_scope([net], name="reset_grad_ops_%d" % self.thread_id):
                for (var, accum_grad) in zip(net.get_vars(), self.accum_grads):
                    zero = tf.zeros(var.get_shape().as_list(), dtype=var.dtype)
                    name = var.name.replace(":", "_") + "_reset_grad_ops"
                    reset_ops = tf.assign(accum_grad, zero, name=name)
                    reset_grad_ops.append(reset_ops)
                return tf.group(*reset_grad_ops, name="reset_accum_group_%d" % self.thread_id)

    def forward_explore(self, train_step):
        terminal = False
        t_start = train_step
        rollout_path = {"state": [], "action": [], "rewards": [], "done": []}
        while not terminal and (train_step - t_start <= flags.t_max):
            action = self.local_net.get_policy(self.master.sess, self.env.state)
            _, reward, terminal = self.env.forward_action(action)
            train_step += 1
            rollout_path["state"].append(self.env.state)
            one_hot_action = np.zeros(self.env.action_dim)
            one_hot_action[action] = 1
            rollout_path["action"].append(one_hot_action)
            rollout_path["rewards"].append(reward)
            rollout_path["done"].append(terminal)
        return train_step, rollout_path

    def discount(self, x):
        return scipy.signal.lfilter([1], [1, -flags.gamma], x[::-1], axis=0)[::-1]

    def train_phase(self):
        sess = self.master.sess
        self.env.reset_env()
        loop = 0
        while flags.train_step <= flags.t_train:
            train_step = 0
            loop += 1
            # reset gradients
            sess.run(self.reset_accum_grads_ops)
            # sync variables
            sess.run(self.sync)
            # forward explore
            train_step, rollout_path = self.forward_explore(train_step)
            # rollout for discounted R values
            if rollout_path["done"][-1]:
                rollout_path["rewards"][-1] = 0
                self.env.reset_env()
                if flags.use_lstm:
                    self.local_net.reset_lstm_state()
            else:
                rollout_path["rewards"][-1] = self.local_net.get_value(sess, rollout_path["state"][-1])
            rollout_path["returns"] = self.discount(rollout_path["rewards"])
            # accumulate gradients
            lc_net = self.local_net
            fetches = [self.do_accum_grads_ops, self.master.global_step]
            if loop % 5 == 0:
                fetches.append(self.summary_op)
            res = sess.run(fetches, feed_dict={lc_net.state: rollout_path["state"],
                                               lc_net.action: rollout_path["action"],
                                               lc_net.target_q: rollout_path["returns"]})
            if loop % 5 == 0:
                global_step, summary_str = res[1], res[2]
                self.master.summary_writer.add_summary(summary_str, global_step=global_step)
            # async update grads to global network
            sess.run(self.apply_gradients)
            flags.train_step += train_step

    def test_phase(self, max_step=1e3):
        rewards = []
        test_step = 0
        while test_step <= flags.t_test:
            terminal = False
            self.env.reset_env()
            episode_reward = 0
            t_start = test_step
            while not terminal and (test_step - t_start) < max_step:
                pi_probs = self.local_net.get_policy(self.master.sess, self.env.state)
                action = self.weighted_choose_action(pi_probs)
                _, reward, terminal = self.env.forward_action(action)
                test_step += 1
                episode_reward += reward
            rewards.append(episode_reward)
        avg_reward = np.mean(rewards)
        logger.info("episode: %d, avg_reward: %.4f" % (len(rewards), avg_reward))


class A3CAtari(object):
    def __init__(self):
        self.env = ControlEnv(gym.make(flags.game))
        # shared network
        if flags.use_lstm:
            self.shared_net = A3CLSTMNet(self.env.state_shape, self.env.action_dim, scope="global_net")
        else:
            self.shared_net = A3CNet(self.env.state_shape, self.env.action_dim, scope="global_net")
        # shared optimizer
        self.shared_opt, self.global_step, self.summary_writer = self.shared_optimizer()
        # local training threads
        self.jobs = []
        for thread_id in xrange(flags.jobs):
            job = A3CSingleThread(thread_id, self)
            self.jobs.append(job)
        # session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                                     allow_soft_placement=True))
        self.sess.run(tf.initialize_all_variables())
        # saver
        self.saver = tf.train.Saver(var_list=self.shared_net.get_vars(), max_to_keep=3)
        restore_model(self.sess, flags.train_dir, self.saver)
        self.global_time_step = 0
        self.phase_id = 0

    def shared_optimizer(self):
        with tf.device("/gpu:%d" % flags.gpu):
            # optimizer
            optimizer = tf.train.RMSPropOptimizer(flags.learn_rate, name="global_optimizer")
            global_step = tf.get_variable("global_step", [], initializer=tf.constant_initializer(0), trainable=False)
            summary_writer = tf.train.SummaryWriter(flags.train_dir, graph_def=self.graph)
        return optimizer, global_step, summary_writer

    def _train(self, thread_idx):
        while True:
            # train phase
            self.jobs[thread_idx].train_phase()
            # test phase
            if flags.train_step > flags.t_train:
                if thread_idx == 0:
                    self.phase_id += 1
                    self.global_time_step += flags.train_step
                    job = self.jobs[0]
                    job.test_phase()
                    if self.phase_id % 5 == 0:
                        save_model(self.sess, flags.train_dir, self.saver, "a3c_model",
                                   global_step=self.global_time_step)
                    flags.train_step = 0
                else:
                    time.sleep(1)

    def signal_handler(self):
        # print "saving model"
        # save_model(self.sess, flags.train_dir, self.saver, "a3c_model", global_step=self.global_time_step)
        sys.exit(-1)

    def train(self):
        flags.train_step = 0
        threads = [threading.Thread(target=self._train, args=(i,)) for i in xrange(flags.jobs)]
        signal.signal(signal.SIGINT, self.signal_handler)
        for thread in threads:
            thread.start()
            thread.join()


def main(_):
    # mkdir
    if not os.path.isdir(flags.train_dir):
        os.makedirs(flags.train_dir)
    # remove old tfevents files
    for f in os.listdir(flags.train_dir):
        if re.search(".*tfevents.*", f):
            os.remove(os.path.join(flags.train_dir, f))
    # model
    model = A3CAtari()
    model.train()


if __name__ == "__main__":
    tf.app.run()
