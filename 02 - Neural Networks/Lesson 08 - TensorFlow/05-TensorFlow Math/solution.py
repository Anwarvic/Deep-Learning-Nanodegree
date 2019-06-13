import tensorflow as tf

# TODO: Convert the following to TensorFlow:
x = tf.constant(10, dtype=tf.float32)
y = tf.constant(2, dtype=tf.float32)
z = x/y - tf.constant(1, dtype=tf.float32)

# TODO: Print z from a session
with tf.Session() as sess:
    print(sess.run(z))