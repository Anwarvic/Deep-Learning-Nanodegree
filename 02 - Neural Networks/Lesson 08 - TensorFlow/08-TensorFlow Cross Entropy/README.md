# Cross Entropy in TensorFlow

As with the softmax function, TensorFlow has a function to do the cross entropy calculations for us.

[![img](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/589b18f5_cross-entropy-diagram/cross-entropy-diagram.png)Cross entropy loss function](https://classroom.udacity.com/nanodegrees/nd101-ent/parts/7de57e8c-12ed-4d7a-b8ea-15db419f6a58/modules/9f27732b-a272-4d8d-8cf3-28159ebc7200/lessons/4d790b9b-f4a1-48ac-bcff-2993ee97e560/concepts/b6f63b94-90e3-463b-8f40-5a5bf3bfcfb9#)

Let's take what you learned from the video and create a cross entropy function in TensorFlow. To create a cross entropy function in TensorFlow, you'll need to use two new functions:

- [`tf.reduce_sum()`](https://www.tensorflow.org/api_docs/python/tf/reduce_sum)
- [`tf.log()`](https://www.tensorflow.org/api_docs/python/tf/log)

## Reduce Sum

```
x = tf.reduce_sum([1, 2, 3, 4, 5])  # 15

```

The [`tf.reduce_sum()`](https://www.tensorflow.org/api_docs/python/tf/reduce_sum) function takes an array of numbers and sums them together.

## Natural Log

```
x = tf.log(100.0)  # 4.60517

```

This function does exactly what you would expect it to do. [`tf.log()`](https://www.tensorflow.org/api_docs/python/tf/log) takes the natural log of a number.

## Quiz

Print the cross entropy using `softmax_data` and `one_hot_encod_label`.