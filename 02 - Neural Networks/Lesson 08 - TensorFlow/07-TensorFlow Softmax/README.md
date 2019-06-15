# TensorFlow Softmax

The softmax function squashes it's inputs, typically called **logits** or **logit scores**, to be between 0 and 1 and also normalizes the outputs such that they all sum to 1. This means the output of the softmax function is equivalent to a categorical probability distribution. It's the perfect function to use as the output activation for a network predicting multiple classes.

[![img](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58950908_softmax-input-output/softmax-input-output.png)Example of the softmax function at work.](https://classroom.udacity.com/nanodegrees/nd101-ent/parts/7de57e8c-12ed-4d7a-b8ea-15db419f6a58/modules/9f27732b-a272-4d8d-8cf3-28159ebc7200/lessons/4d790b9b-f4a1-48ac-bcff-2993ee97e560/concepts/bb19eec1-3fb2-413c-8043-9e514c4e3396#)

## TensorFlow Softmax

We're using TensorFlow to build neural networks and, appropriately, there's a function for calculating softmax.

```
x = tf.nn.softmax([2.0, 1.0, 0.2])

```

Easy as that! [`tf.nn.softmax()`](https://www.tensorflow.org/api_docs/python/tf/nn/softmax) implements the softmax function for you. It takes in logits and returns softmax activations.

## Quiz

Use the softmax function in the quiz below to return the softmax of the logits.