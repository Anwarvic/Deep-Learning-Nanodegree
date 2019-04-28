# Bike Sharing Project

For the first project, we'll build a neural network to predict bike-sharing rides. 

Imagine yourself owning a bike sharing company like [Cycle Hop](https://cyclehop.com). You want to protect how many bikes you need because if you have too few you're losing money from potential riders. If you have too many you're wasting money on bikes that are just sitting around. So you need to predict from historical data how many bikes you'll need in the near future.

<p align="center">
<img src="http://www.mediafire.com/convkey/a795/z8j0bvvu28jyri5zg.jpg"/> 
</p>

 A good way to do this is with a neural network which is exactly what you'll do. In this notebook, you'll implement a neural network and train it on historical data to make predictions. After training the network and viewing its performance as it's being trained, you'll compare your networks predictions with actual data. If you build your network right you should find the network does a really good job, for the most part. Right! It's time for you to get started on your project.



## Review

After submission this project, I got this result:

```powershell
$ udacity submit
Submission includes the following files:
    my_answers.py
    Your_first_neural_network.ipynb

Uploading submission...
[=========================== 100% ===========================] 316586/316586

Waiting for results...Done!

Results:
--------

************************************************************************
                          Test Result Summary
************************************************************************

Produces good results when running the network on full data            .
The activation function is a sigmoid                                   .
The backpropagation implementation is correct                          .
The forward pass implementation is correct                             .
The learning_rate is reasonable                                        .
The number of epochs is reasonable                                     .
The number of hidden nodes is reasonable                               .
The number of output nodes is correct                                  .
The run method is correct                                              .
The update_weights implementation is correct                           .
The weights are updated correctly on training                          .

--------------------------------------------------------------------------------

Congratulations!  It looks like your network passed all of our tests.  You're ready to submit your project at
https://review.udacity.com/#!/rubrics/700/submit-zip


Details are available in first_neural_network-result-263543.json.

If you would like this version of the project to be reviewed,
submit first_neural_network-263543.zip to the reviews website.
```

