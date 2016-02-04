from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf

from tensorflow.models.rnn import rnn
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq


class adict(dict):
    ''' Attribute dictionary - a convenience data structure, similar to SimpleNamespace in python 3.3
        One can use attributes to read/write dictionary content.
    '''
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self


def inference_graph(vocab_size=164, num_layers=3, hidden_size=256, batch_size=1, num_steps=1, dropout_rate=0.0):
    ''' Builds an inference graph. '''

    input_data = tf.placeholder(tf.int32, [batch_size, num_steps])

    lstm_cell = rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0)
    if dropout_rate > 0:
        lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=1-dropout_rate)
    cell = rnn_cell.MultiRNNCell([lstm_cell] * num_layers)

    initial_state = cell.zero_state(batch_size, tf.float32)

    with tf.device("/cpu:0"):
        embedding = tf.get_variable("embedding", [vocab_size, hidden_size])
        inputs = tf.split(1, num_steps, tf.nn.embedding_lookup(embedding, input_data))
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

    if dropout_rate > 0:
        inputs = [tf.nn.dropout(input_, 1-dropout_rate) for input_ in inputs]

    outputs, states = rnn.rnn(cell, inputs, initial_state=initial_state)

    output = tf.reshape(tf.concat(1, outputs), [-1, hidden_size])
    logits = tf.nn.xw_plus_b(output,
                             tf.get_variable("softmax_w", [hidden_size, vocab_size]),
                             tf.get_variable("softmax_b", [vocab_size]))
    final_state = states[-1]
    
    return adict(
        input_data=input_data,
        initial_state=initial_state,
        logits=logits,
        final_state=final_state)


def cost_graph(logits, batch_size, num_steps, vocab_size):
    ''' Builds cost graph. '''
    
    targets = tf.placeholder(tf.int32, [batch_size, num_steps], name='targets')
    
    loss = seq2seq.sequence_loss_by_example([logits],
                                            [tf.reshape(targets, [-1])],
                                            [tf.ones([batch_size * num_steps])],
                                            vocab_size)
    cost = tf.div(tf.reduce_sum(loss), batch_size * num_steps, name='cost')
    
    return adict(
        targets=targets,
        cost=cost)


def training_graph(cost, grad_clip=5.0):
    ''' Builds training graph. '''

    # Adam's learning parameters
    lr = tf.Variable(0.001, trainable=False, name='lr')
    beta1 = tf.Variable(0.8, trainable=False, name='beta1')
    beta2 = tf.Variable(0.95, trainable=False, name='beta2')

    # collect all trainable variables
    tvars = tf.trainable_variables()
    
    grads = [tf.clip_by_value(t, -grad_clip, grad_clip) for t in tf.gradients(cost, tvars)]

    optimizer = tf.train.AdamOptimizer(lr, beta1, beta2)
    train_op = optimizer.apply_gradients(zip(grads, tvars))

    return adict(
        lr=lr,
        beta1=beta1,
        beta2=beta2,

        tvars=tvars,        
        grads=grads,
        
        train_op=train_op)
