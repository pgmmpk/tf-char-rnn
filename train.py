"""
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import reader
import os
import time

import graph
from vocab import Vocab
from summary_graph import summary_graph


flags = tf.flags
logging = tf.logging

flags.DEFINE_string ('input_data',     None,  'filename of the input data')
flags.DEFINE_string ('train_dir',      None,  'training directory (models and summaries are saved there periodically)')
flags.DEFINE_string ('load_model',     None,  '(optional) filename of the model to load')

flags.DEFINE_integer('num_layers',     3,     'number of layers (default 3)')
flags.DEFINE_integer('hidden_size',    256,   'size of one neural layer (default 256)')
flags.DEFINE_integer('vocab_size',     256,   'size of vocabulary (default 256)')

flags.DEFINE_integer('batch_size',     50,    'minibatch size (default 50)')
flags.DEFINE_integer('num_steps',      200,   'number of steps to unroll RNN in time (default 200)')
flags.DEFINE_integer('max_epochs',     5,     'max number of training epochs (default 5)')
flags.DEFINE_integer('eval_val_every', 1000,  'how often to evaluate on the validation set (default every 1000 steps)')
flags.DEFINE_float  ('train_fraction', 0.95,  'fraction of data used for training (default 0.95)')
flags.DEFINE_float  ('valid_fraction', 0.05,  'fraction of data used for validation (default 0.05)')
flags.DEFINE_float  ('init_scale',     0.1,   'scale of initial uniform random initialization of model parameters (default 0.1).')
flags.DEFINE_float  ('dropout_rate',   0.0,   'dropout rate (0 means - no dropout, default 0)')
flags.DEFINE_float  ('grad_clip',      5.0,   'clip gradients (default 5.0)')
flags.DEFINE_float  ('learning_rate',  0.002, 'Adam\'s learning rate (default 0.002)')
flags.DEFINE_float  ('beta1',          0.0,   'Adam\'s learning parameter "beta1" (default 0.0)')
flags.DEFINE_float  ('beta2',          0.95,  'Adam\'s learning parameter "beta2" (default 0.95)')

FLAGS = flags.FLAGS


def run_test(session, m, data, batch_size, num_steps):
    """Runs the model on the given data."""

    costs = 0.0
    iters = 0
    state = session.run(m.initial_state)

    for step, (x, y) in enumerate(reader.dataset_iterator(data, batch_size, num_steps)):
        cost, state = session.run([m.cost, m.final_state], {
            m.input_data: x,
            m.targets: y,
            m.initial_state: state
        })

        costs += cost
        iters += 1

    return costs / iters


def main(unused_args):
    ''' Trains model from data '''

    if not FLAGS.input_data:
        raise ValueError("Must set --input_data to the filename of input dataset")

    if not FLAGS.train_dir:
        raise ValueError("Must set --train_dir to the directory where training files will be saved")

    if not os.path.exists(FLAGS.train_dir):
        os.mkdir(FLAGS.train_dir)

    with tf.Graph().as_default(), tf.Session() as session:

        ''' To make tf.train.Saver write parameters as part of the saved file, add params to the graph as variables (hackish? - MK)'''
        with tf.variable_scope("params", reuse=None):
            num_layers_var = tf.Variable(FLAGS.num_layers, trainable=False, name='num_layers')
            hidden_size_var = tf.Variable(FLAGS.hidden_size, trainable=False, name='hidden_size')

            ''' If pre-trained model loaded from file, use loaded vocabulary and NN geometry. Else, compute vocabulary and use command-line params for num_layers and hidden_size '''
            if FLAGS.load_model:
                vocab_size_var = tf.Variable(0, trainable=False, name='vocab_size')
                tf.train.Saver([num_layers_var, hidden_size_var, vocab_size_var]).restore(session, FLAGS.load_model)
                vocab_var = tf.Variable([0] * vocab_size_var.eval(), trainable=False, name='vocab')
                tf.train.Saver([vocab_var]).restore(session, FLAGS.load_model)

                FLAGS.num_layers = np.asscalar(num_layers_var.eval())  # need np.asscalar to upcast np.int32 to Python int
                FLAGS.hidden_size = np.asscalar(hidden_size_var.eval())

                vocab = Vocab.from_array(vocab_var.eval())
                train_data, valid_data, test_data, vocab = reader.read_datasets(FLAGS.input_data, FLAGS.train_fraction, FLAGS.valid_fraction, vocab=vocab)
            else:
                train_data, valid_data, test_data, vocab = reader.read_datasets(FLAGS.input_data, FLAGS.train_fraction, FLAGS.valid_fraction, vocab_size=FLAGS.vocab_size)
                vocab_size_var = tf.Variable(vocab.size, trainable=False, name='vocab_size')
                vocab_var = tf.Variable(vocab.to_array(), trainable=False, name='vocab')

        ''' build training graph '''
        initializer = tf.random_uniform_initializer(-FLAGS.init_scale, FLAGS.init_scale)
        with tf.variable_scope("model", initializer=initializer):
            m = graph.inference_graph(vocab.size, FLAGS.num_layers, FLAGS.hidden_size, FLAGS.batch_size, FLAGS.num_steps, FLAGS.dropout_rate)
            m.update(graph.cost_graph(m.logits, FLAGS.batch_size, FLAGS.num_steps, vocab.size))
            m.update(graph.training_graph(m.cost, FLAGS.grad_clip))

        # create saver before creating more graph nodes, so that we do not save any vars defined below
        saver = tf.train.Saver(max_to_keep=50)

        ''' build graph for validation and testing (shares parameters with the training graph!) '''
        with tf.variable_scope("model", reuse=True):
            mvalid = graph.inference_graph(vocab.size, FLAGS.num_layers, FLAGS.hidden_size, FLAGS.batch_size, FLAGS.num_steps)
            mvalid.update(graph.cost_graph(mvalid.logits, FLAGS.batch_size, FLAGS.num_steps, vocab.size))

        if FLAGS.load_model:
            saver.restore(session, FLAGS.load_model)
            print('Loaded model from', FLAGS.load_model)
        else:
            print('Created model')

        print('\tnum_layers:', FLAGS.num_layers)
        print('\thidden_size:', FLAGS.hidden_size)
        print('\tvocab_size:', vocab.size)
        print()
        print('Training parameters')
        print('\tbatch_size:', FLAGS.batch_size)
        print('\tnum_steps:', FLAGS.num_steps)
        print('\tlearning_rate:', FLAGS.learning_rate)
        print('\tbeta1:', FLAGS.beta1)
        print('\tbeta2:', FLAGS.beta2)
        print()
        print('Datasets')
        print('\ttraining dataset size:', len(train_data))
        print('\tvalidation dataset size:', len(valid_data))
        print('\ttest dataset size:', len(test_data))
        print()

        ''' create two summaries: training cost and validation cost '''
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph=session.graph)
        summary_train = summary_graph('Training cost', ema_decay=0.95)
        summary_valid = summary_graph('Validation cost')

        session.run([
            m.lr.initializer,
            m.beta1.initializer,
            m.beta2.initializer,
        ])

        tf.initialize_all_variables().run()

        session.run([
            tf.assign(m.lr, FLAGS.learning_rate),
            tf.assign(m.beta1, FLAGS.beta1),
            tf.assign(m.beta2, FLAGS.beta2),
        ])

        state = session.run(m.initial_state)
        iterations = len(train_data) // FLAGS.batch_size // FLAGS.num_steps * FLAGS.max_epochs
        for i, (x, y) in enumerate(reader.next_batch(train_data, FLAGS.batch_size, FLAGS.num_steps)):
            if i >= iterations:
                break

            start_time = time.time()

            cost, state, _ = session.run([m.cost, m.final_state, m.train_op], {
                    m.input_data: x,
                    m.targets: y,
                    m.initial_state: state
            })

            epoch = float(i) / (len(train_data) // FLAGS.batch_size // FLAGS.num_steps)
            time_elapsed = time.time() - start_time
            print('%d/%d (epoch %.3f), train_loss = %6.8f, time/batch = %.4fs' % (i+1, iterations, epoch, cost, time_elapsed))

            session.run([summary_train.update], {summary_train.x: cost})

            if (i+1) % FLAGS.eval_val_every == 0 or i == iterations-1:
                # evaluate loss on validation data
                cost = run_test(session, mvalid, valid_data, FLAGS.batch_size, FLAGS.num_steps)
                print("validation cost = %6.8f" % cost)
                save_as = '%s/epoch%.2f_%.4f.model' % (FLAGS.train_dir, epoch, cost)
                saver.save(session, save_as)

                ''' write out summary events '''
                buffer, = session.run([summary_train.summary])
                summary_writer.add_summary(buffer, i)

                session.run([summary_valid.update], {summary_valid.x: cost})
                buffer, = session.run([summary_valid.summary])
                summary_writer.add_summary(buffer, i)

                summary_writer.flush()

        if len(test_data) > FLAGS.batch_size * FLAGS.num_steps:
            cost = run_test(session, mvalid, test_data, FLAGS.batch_size, FLAGS.num_steps)
            print("Test cost: %.3f" % test_loss)

if __name__ == "__main__":
    tf.app.run()
