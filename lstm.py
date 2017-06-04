import os

import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, LSTMCell
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)



class Model():
    def __init__(self, config):

        num_layers = config['num_layers']
        hidden_size = config['hidden_size']
        max_grad_norm = config['max_grad_norm']
        self.batch_size = config['batch_size']
        seq_len = config['seq_len']
        n_outputs = config['num_classes']
        n_inputs = config['n_inputs']
        """Place holders"""
        self.input = tf.placeholder(
            tf.float32, [None, seq_len, n_inputs], name='input')
        self.labels = tf.placeholder(tf.int64, [None], name='labels')
        self.input_keep_prob = tf.placeholder("float", name='input_keep_prob')
        self.output_keep_prob = tf.placeholder(
            "float", name='output_keep_prob')
        with tf.name_scope("Cell"):
            def single_cell():
                if config['cell_type'] == 'LSTM':
                    return tf.contrib.rnn.DropoutWrapper(LSTMCell(hidden_size), output_keep_prob=self.output_keep_prob)
                else:
                    return tf.contrib.rnn.DropoutWrapper(GRUCell(hidden_size), output_keep_prob=self.output_keep_prob)
            cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(
                [single_cell() for _ in range(num_layers)]), input_keep_prob=self.input_keep_prob)

        with tf.variable_scope('Rnn'):
            outputs, states = tf.nn.dynamic_rnn(
                cell, self.input, dtype=tf.float32)
        if config['cell_type'] == 'LSTM':
            top_layer_h_state = states[-1][1]
        else:
            top_layer_h_state = states[-1]
        logits = tf.layers.dense(top_layer_h_state, n_outputs, name="softmax")
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.labels, logits=logits)
        self.cost = tf.reduce_mean(xentropy, name="loss") / self.batch_size

        with tf.name_scope("Prediction_sequence"):
            self.prediction = tf.map_fn(lambda x: tf.cast(tf.argmax(tf.layers.dense(
                x, n_outputs, name="softmax", reuse=True), 1), "float"), outputs)
            self.prediction_prob = tf.map_fn(lambda x: tf.nn.softmax(
                tf.layers.dense(x, n_outputs, name="softmax", reuse=True)), outputs)

        with tf.name_scope("Evaluating_accuracy"):
            correct_prediction = tf.equal(tf.argmax(logits, 1), self.labels)
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, "float"))
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.scalar('cost', self.cost)

        """Optimizer"""
        with tf.name_scope("Optimizer"):
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(
                tf.gradients(self.cost, tvars), max_grad_norm)
            # We clip the gradients to prevent explosion
            optimizer = config['optimizer']
            gradients = list(zip(grads, tvars))
            self.train_op = optimizer.apply_gradients(gradients)
            # Add histograms for variables, gradients and gradient norms.
            # The for-loop loops over all entries of the gradient and plots
            # a histogram.
            for gradient, variable in gradients:  # plot the gradient of each trainable variable
                if isinstance(gradient, ops.IndexedSlices):
                    grad_values = gradient.values
                else:
                    grad_values = gradient
                tf.summary.histogram(variable.name, variable)
                tf.summary.histogram(variable.name + "/gradients", grad_values)
                tf.summary.histogram(
                    variable.name + "/gradient_norm", clip_ops.global_norm([grad_values]))

        # Final code for the TensorBoard
        self.merged = tf.summary.merge_all()
        self.init_op = tf.global_variables_initializer()
        print('Finished computation graph')
