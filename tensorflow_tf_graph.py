# http://localhost:6006/# to view the graphs in tensorboard
import tensorflow as tf
from datetime import datetime

print(tf.__version__)

@tf.function
def add(a,b):
    return tf.multiply(a, b)

stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = 'logs/func/%s' % stamp
writer = tf.summary.create_file_writer(logdir)

a = tf.constant(2)
b = tf.constant(3)

tf.summary.trace_on(graph=True, profiler=True)
c = add(a, b)

with writer.as_default():
  tf.summary.trace_export(
      name="my_func_trace",
      step=0,
      profiler_outdir=logdir)