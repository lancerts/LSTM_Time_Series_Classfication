"""
===============================================================================
LSTM/GRU for time series classification problem.
===============================================================================
"""

import shutil

import numpy as np

import tensorflow as tf  # TF 1.1
from lstm import Model
from utilities import DataIterator, load_data

tf.logging.set_verbosity(tf.logging.ERROR)
tf.reset_default_graph()


# Set directories for datasets and summary logs
direc = '.'
summaries_dir = './tf2'
try:
    shutil.rmtree(summaries_dir)
except:
    print("model_dir does not exist.")

"""Load the data"""
ratio = np.array(
    [0.1, 0.2])  # Ratios where to split the training and validation set
np.random.seed(123)
df_train, df_val, df_test = load_data(
    direc, ratio, dataset='ChlorineConcentration')
N = df_train.shape[0]
seq_len = df_train.shape[1] - 1
num_classes = len(np.unique(df_train[:, 0]))

"""Hyperparamaters"""
batch_size = 200
max_iterations = 10000
input_keep_prob = 1.
output_keep_prob = 1.  # used for dropout
n_inputs = 1
config = {'cell_type':    'GRU',  # LSTM or GRU
          'num_layers':    1,  # number of layers of stacked RNN's
          'hidden_size':   200,  # memory cells in a layer, hidden layer size
          'max_grad_norm': 5,  # maximum gradient norm during training
          'batch_size':    batch_size,
          'seq_len':        seq_len,
          'num_classes':    num_classes,
          'n_inputs':       n_inputs}

config['optimizer'] = tf.train.AdamOptimizer(learning_rate=0.005)
# decay = 0.99 # decay parameter for running average

early_stopping = {'metric':       'Accuracy',
                  'stop_iters':   1000,
                  'eval_iters':   100}


epochs = np.floor(batch_size * max_iterations / N)
print('Train %.0f samples in approximately %d epochs' % (N, epochs))


# Instantiate a model
model = Model(config)
saver = tf.train.Saver()

"""Session time"""
sess = tf.Session()  # Depending on your use, do not forget to close the session
tr_writer = tf.summary.FileWriter(
    summaries_dir + '/train', sess.graph)  # writer for Tensorboard
va_writer = tf.summary.FileWriter(summaries_dir + '/validate')
sess.run(model.init_op)
tf.set_random_seed(1)
np.random.seed(1)

# Moving average training cost
cost_train_rn = -np.log(1 / float(num_classes) + 1e-9)
acc_train_rn = 0.0
tr = DataIterator(df_train, seq_len, n_inputs)
va = DataIterator(df_val, seq_len, n_inputs)
te = DataIterator(df_test, seq_len, n_inputs)
acc_val_max = 0.0


try:
    for i in range(max_iterations):
        X_batch_tr, y_batch_tr = tr.next_batch(batch_size)
        # Next line does the actual training
        cost_train, acc_train, _, summary = sess.run([model.cost, model.accuracy, model.train_op, model.merged],
                                                     feed_dict={model.input: X_batch_tr, model.labels: y_batch_tr,
                                                                model.input_keep_prob: input_keep_prob,
                                                                model.output_keep_prob: output_keep_prob})
        #tr_writer.add_summary(summary, i)

        cost_train_rn = cost_train_rn * i / (i + 1) + cost_train / (i + 1)
        acc_train_rn = acc_train_rn * i / (i + 1) + acc_train / (i + 1)

        if i % early_stopping['eval_iters'] == 1:
            # Evaluate validation performance
            X_batch_va, y_batch_va = va.next_batch(df_val.shape[0])
            cost_val, acc_val, summary = sess.run([model.cost, model.accuracy, model.merged],
                                                  feed_dict={model.input: X_batch_va, model.labels: y_batch_va,
                                                             model.input_keep_prob: 1.0,
                                                             model.output_keep_prob: 1.0})

            epoch = float(i) * batch_size / N
            print(
                'Trained %.1f epochs, progress measures on train/validation/train_rn: ' % epoch)
            print('Cost %5.3f/%5.3f/%5.3f ' %
                  (cost_train, cost_val, cost_train_rn))
            print('Acc %5.3f/%5.3f/%5.3f ' %
                  (acc_train, acc_val, acc_train_rn))
            #saver.save(sess, summaries_dir+'/model.ckpt', global_step = i)
            # Write information to TensorBoard
            va_writer.add_summary(summary, i)
            if max(acc_val, acc_val_max) > acc_val_max:
                acc_val_max = max(acc_val, acc_val_max)
                iters = 0  # early_stopping iteration count
            else:
                iters += early_stopping['eval_iters']
                if iters >= early_stopping['stop_iters']:
                    print('Best validation results achieved at %s iterations, %.3f epochs' % (
                        i - iters, float(i - iters) * batch_size / N))
                    break


except KeyboardInterrupt:
    pass


# Evaluate test performance
X_te, y_te = te.next_batch(df_test.shape[0])
acc_te, pred_te, pred_prob_te = sess.run([model.accuracy, model.prediction, model.prediction_prob], feed_dict={model.input: X_te, model.labels: y_te,
                                                                                                               model.input_keep_prob: 1.0,
                                                                                                               model.output_keep_prob: 1.0})
print('Test accuracy %5.3f ' % (acc_te))
#print('Sample predicted class sequence ', pred_te[0])
#print('Sample predicted probability sequence ', pred_prob_te[0])

sess.close()

tr_writer.close()
va_writer.close()
