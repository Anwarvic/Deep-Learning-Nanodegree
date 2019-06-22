# Solution is available in the other "solution.py" tab
import tensorflow as tf

hidden_layer_weights = [
    [0.1, 0.2, 0.4],
    [0.4, 0.6, 0.6],
    [0.5, 0.9, 0.1],
    [0.8, 0.2, 0.8]]
out_weights = [
    [0.1, 0.6],
    [0.2, 0.1],
    [0.7, 0.9]]

# Weights and biases
weights = [
    tf.Variable(hidden_layer_weights),
    tf.Variable(out_weights)]
biases = [
    tf.Variable(tf.zeros(3)),
    tf.Variable(tf.zeros(2))]

# Input
features = tf.Variable([[0.0, 2.0, 3.0, 4.0],
                        [0.1, 0.2, 0.3, 0.4],
                        [11.0, 12.0, 13.0, 14.0]])

# TODO: Create Model with Dropout
net1 = tf.matmul(features, weights[0]) + biases[0]
net1 = tf.nn.relu(net1)
net1 = tf.nn.dropout(net1, 0.5) #keep_prob of 0.5
net2 = tf.matmul(net1, weights[1]) + biases[1]
logits = net2

# TODO: Print logits from a session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(logits))


