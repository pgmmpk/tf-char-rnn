from graph import adict
import tensorflow as tf


def summary_graph(tag, ema_decay=0):
    """
    Usage:
        sg = summary_graph('Training cost', ema_decay=0.95)

        while training:
            ...

            session.run([sg.update], {sg.x: cost})  # accumulate data here
            summary_writer.add_summary(
    """

    x = tf.placeholder(tf.float32, [])

    if ema_decay == 0:
        var = tf.Variable(0.0, trainable=False)
        update = tf.assign(var, x)
    else:
        ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        update = ema.apply([x])
        var = ema.average(x)

    summary = tf.summary.scalar(tag, var)

    return adict(
        x=x,
        update=update,
        summary=summary
    )
