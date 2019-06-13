# TensorFlow Math

Getting the input is great, but now you need to use it. You're going to use basic math functions that everyone knows and loves - add, subtract, multiply, and divide - with tensors. (There's many more math functions you can check out in the [documentation](https://www.tensorflow.org/api_docs/python/math_ops/).)

## Addition

```
x = tf.add(5, 2)  # 7

```

You’ll start with the add function. The [`tf.add()`](https://www.tensorflow.org/api_guides/python/math_ops) function does exactly what you expect it to do. It takes in two numbers, two tensors, or one of each, and returns their sum as a tensor.

## Subtraction and Multiplication

Here’s an example with subtraction and multiplication.

```
x = tf.subtract(10, 4) # 6
y = tf.multiply(2, 5)  # 10

```

The `x` tensor will evaluate to `6`, because `10 - 4 = 6`. The `y` tensor will evaluate to `10`, because `2 * 5 = 10`. That was easy!

## Converting types

It may be necessary to convert between types to make certain operators work together. For example, if you tried the following, it would fail with an exception:

```
tf.subtract(tf.constant(2.0),tf.constant(1))  # Fails with ValueError: Tensor conversion requested dtype float32 for Tensor with dtype int32: 

```

That's because the constant `1` is an integer but the constant `2.0` is a floating point value and `subtract` expects them to match.

In cases like these, you can either make sure your data is all of the same type, or you can cast a value to another type. In this case, converting the `2.0` to an integer before subtracting, like so, will give the correct result:

```
tf.subtract(tf.cast(tf.constant(2.0), tf.int32), tf.constant(1))   # 1

```

## Quiz

Let's apply what you learned to convert an algorithm to TensorFlow. The code below is a simple algorithm using division and subtraction. Convert the following algorithm in regular Python to TensorFlow and print the results of the session. You can use [`tf.constant()`](https://www.tensorflow.org/api_guides/python/constant_op) for the values `10`, `2`, and `1`.

