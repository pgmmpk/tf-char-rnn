from __future__ import print_function

import reader
import numpy as np

import tensorflow as tf

from vocab import Vocab
import graph

flags = tf.flags
logging = tf.logging

flags.DEFINE_string('load_model', None, 'model file to load')
flags.DEFINE_integer('sample_size', 2000, 'how many characters to sample')
flags.DEFINE_float('temperature', 1, 'sampling tempterature')

FLAGS = flags.FLAGS

FLAGS.num_layers = 2
FLAGS.hidden_size = 128


def main(unused_args):
    ''' Generates data from a trained model (fun!) '''
    
    if not FLAGS.load_model:
        print('--load_model is required')
        return -1
  
    with tf.Graph().as_default(), tf.Session() as session:
        
        ''' load parameters of the model '''    
        with tf.variable_scope("params"):
            num_layers_var = tf.Variable(0, name='num_layers')
            hidden_size_var = tf.Variable(0, name='hidden_size')
            vocab_size_var = tf.Variable(0, name='vocab_size')
            tf.train.Saver([num_layers_var, hidden_size_var, vocab_size_var]).restore(session, FLAGS.load_model)
            vocab_var = tf.Variable([0] * vocab_size_var.eval(), name='vocab')
            tf.train.Saver([vocab_var]).restore(session, FLAGS.load_model)
            
            FLAGS.num_layers = np.asscalar(num_layers_var.eval())
            FLAGS.hidden_size = np.asscalar(hidden_size_var.eval())
            
            vocab = Vocab.from_array(vocab_var.eval())
            
            print('Loaded model from file', FLAGS.load_model)
            print('\tnum_layers:', FLAGS.num_layers)
            print('\thidden_size:', FLAGS.hidden_size)
            print('\tvocab_size', vocab.size)
        
        ''' load inference graph '''
        with tf.variable_scope("model", reuse=None):
            m = graph.inference_graph(vocab.size, FLAGS.num_layers, FLAGS.hidden_size)
          
        tf.train.Saver().restore(session, FLAGS.load_model)
        
        logits = np.ones((vocab.size,))
        state = session.run(m.initial_state)
        for i in range(FLAGS.sample_size):
            logits = logits / FLAGS.temperature
            prob = np.exp(logits)
            prob /= np.sum(prob)
            prob = prob.ravel()
            ix = np.random.choice(range(len(prob)), p=prob)
            
            print(vocab.decode(ix), end='')
        
            logits, state = session.run([m.logits, m.final_state],
                                         {m.input_data: np.array([[ix]]),
                                          m.initial_state: state})


if __name__ == "__main__":
  tf.app.run()
