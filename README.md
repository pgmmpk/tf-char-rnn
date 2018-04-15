tf-char-rnn
===========

This is a [TensorFlow](http://tensorflow.org) re-implementation of the [char-rnn Lua code by Karpathy](https://github.com/karpathy/char-rnn).

See [Karpathy blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) for description of what all this is about.

Running
-------

You will need:

1. A text file with some data (you can use `tinyshakespeare.txt` from the `data/` folder)
2. A machine with installed Python 3
3. (optional but highly recommended) a GPU on this


Configure and use virtual environment, install required packages:
```!bash
virtualenv .venv -p python3
. .venv/bin/activate
pip install -r requirements.txt
pip install tensorflow  # or tensorflow-gpu, if you have GPU
```

Training
--------

```!bash
python train.py --input_data data/tinyshakespeare.txt --train_dir tinyshakespeare
```

Monitoring training and validation losses
-----------------------------------------

Start TensorBoard server
```!bash
python -m tensorflow.tensorboard.tensorboard --logdir `pwd`
```

Use web browser to connect to the TensorBoard server and see computational graph and plots of training and validation losses.


Sampling
--------

```!bash
python sample.py --load_model tinyshakespeare/lm_10000.model --prefix 'O thy!'
```
