tf-char-rnn
===========

This is a [TensorFlow](http://tensorflow.org) re-implementation of the [char-rnn Lua code by Karpathy](https://github.com/karpathy/char-rnn).

See [Karpathy blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) for description of what all this is about.

Running
-------

You will need:

1. A text file with some data (you can use `tinyshakespeare.txt` from the `data/` folder)
2. A machine with installed TensorFlow
3. (optional but highly recommended) a GPU on the machine where TensorFlow is installed

Training
--------

```bash
python train.py -i data/tinyshakespeare.txt -o tinyshakespeare
```

Monitoring training and validation losses
-----------------------------------------

Start TensorBoard server
```bash
python -m tensorflow.tensorboard.tensorboard `pwd`
```

Use web browser to connect to the TensorBoard server and see computational graph and plots of training and validation losses.


Sampling
--------

```bash
python sample.py -m tinyshakespeare/lm_10000.model -t "O thy!"
```
