# Linear functions in TensorFlow

The most common operation in neural networks is calculating the linear combination of inputs, weights, and biases. As a reminder, we can write the output of the linear operation as

[![img](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58a4d8b3_linear-equation/linear-equation.gif)](https://classroom.udacity.com/nanodegrees/nd101-ent/parts/7de57e8c-12ed-4d7a-b8ea-15db419f6a58/modules/9f27732b-a272-4d8d-8cf3-28159ebc7200/lessons/4d790b9b-f4a1-48ac-bcff-2993ee97e560/concepts/baf36422-c1b4-4005-960f-63a550e635d4#)

Here, \mathbf{W}W is a matrix of the weights connecting two layers. The output \mathbf{y}y, the input \mathbf{x}x, and the biases \mathbf{b}bare all vectors.

## Weights and Bias in TensorFlow

The goal of training a neural network is to modify weights and biases to best predict the labels. In order to use weights and bias, you'll need a Tensor that can be modified. This leaves out [`tf.placeholder()`](https://www.tensorflow.org/api_docs/python/tf/placeholder) and [`tf.constant()`](https://www.tensorflow.org/api_docs/python/tf/constant), since those Tensors can't be modified. This is where [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable) class comes in.

### tf.Variable()

```
x = tf.Variable(5)

```

The [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable) class creates a tensor with an initial value that can be modified, much like a normal Python variable. This tensor stores its state in the session, so you must initialize the state of the tensor manually. You'll use the [`tf.global_variables_initializer()`](https://www.tensorflow.org/api_docs/python/tf/global_variables_initializer) function to initialize the state of all the Variable tensors.

##### Initialization

```
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

```

The [`tf.global_variables_initializer()`](https://www.tensorflow.org/api_docs/python/tf/global_variables_initializer) call returns an operation that will initialize all TensorFlow variables from the graph. You call the operation using a session to initialize all the variables as shown above. Using the [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable) class allows us to change the weights and bias, but an initial value needs to be chosen.

Initializing the weights with random numbers from a normal distribution is good practice. Randomizing the weights helps the model from becoming stuck in the same place every time you train it. You'll learn more about this in the next lesson, when you study gradient descent.

Similarly, choosing weights from a normal distribution prevents any one weight from overwhelming other weights. You'll use the [`tf.truncated_normal()`](https://www.tensorflow.org/api_docs/python/tf/truncated_normal) function to generate random numbers from a normal distribution.

### tf.truncated_normal()

```
n_features = 120
n_labels = 5
weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))

```

The [`tf.truncated_normal()`](https://www.tensorflow.org/api_docs/python/tf/truncated_normal) function returns a tensor with random values from a normal distribution whose magnitude is no more than 2 standard deviations from the mean.

Since the weights are already helping prevent the model from getting stuck, you don't need to randomize the bias. Let's use the simplest solution, setting the bias to 0.

### tf.zeros()

```
n_labels = 5
bias = tf.Variable(tf.zeros(n_labels))

```

The [`tf.zeros()`](https://www.tensorflow.org/api_docs/python/tf/zeros) function returns a tensor with all zeros.

## Linear Classifier Quiz

[![img](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/582cf7a7_mnist-012/mnist-012.png)A subset of the MNIST dataset](https://classroom.udacity.com/nanodegrees/nd101-ent/parts/7de57e8c-12ed-4d7a-b8ea-15db419f6a58/modules/9f27732b-a272-4d8d-8cf3-28159ebc7200/lessons/4d790b9b-f4a1-48ac-bcff-2993ee97e560/concepts/baf36422-c1b4-4005-960f-63a550e635d4#)

You'll be classifying the handwritten numbers `0`, `1`, and `2` from the MNIST dataset using TensorFlow. The above is a small sample of the data you'll be training on. Notice how some of the `1`s are written with a [serif](https://en.wikipedia.org/wiki/Serif) at the top and at different angles. The similarities and differences will play a part in shaping the weights of the model.

[![img](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/582ce9ef_weights-0-1-2/weights-0-1-2.png)Left: Weights for labeling 0. Middle: Weights for labeling 1. Right: Weights for labeling 2.](https://classroom.udacity.com/nanodegrees/nd101-ent/parts/7de57e8c-12ed-4d7a-b8ea-15db419f6a58/modules/9f27732b-a272-4d8d-8cf3-28159ebc7200/lessons/4d790b9b-f4a1-48ac-bcff-2993ee97e560/concepts/baf36422-c1b4-4005-960f-63a550e635d4#)

The images above are trained weights for each label (`0`, `1`, and `2`). The weights display the unique properties of each digit they have found. Complete this quiz to train your own weights using the MNIST dataset.

### Instructions

1. Open quiz.py.
   1. Implement `get_weights` to return a [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable) of weights
   2. Implement `get_biases` to return a [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable) of biases
   3. Implement `xW + b` in the `linear` function
2. Open sandbox.py
   1. Initialize all weights

Since `xW` in `xW + b` is matrix multiplication, you have to use the [`tf.matmul()`](https://www.tensorflow.org/api_docs/python/tf/matmul) function instead of [`tf.multiply()`](https://www.tensorflow.org/api_docs/python/tf/multiply). Don't forget that order matters in matrix multiplication, so `tf.matmul(a,b)` is not the same as `tf.matmul(b,a)`.