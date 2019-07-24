# Quiz: Incremental Mean

In the previous video, we learned about an algorithm that can keep a running estimate of the mean of a sequence of numbers (x1,x2,…,xn). The algorithm looked at each number in the sequence in order, and successively updated the mean μ.

[![img](https://d17h27t6h515a5.cloudfront.net/topher/2017/October/59d6690f_incremental/incremental.png)](https://classroom.udacity.com/nanodegrees/nd101-ent/parts/b4ed3716-d168-4db5-b74b-f224744550e2/modules/6c24dd74-5fba-4e0d-ab27-0f117c2778f1/lessons/7557675a-f42d-4b2e-b4dc-d338e4bb2f88/concepts/1500d564-5bfc-4fe3-848c-3bcff316269d#)

Use the pseudocode to complete the `running_mean` function below. Your function should accept a list of numbers `x` as input. It should return a list `mean_values`, where `mean_values[k]` is the mean of `x[:k+1]`.

**Note**: Pay careful attention to indexing! Here, xk corresponds to `x[k-1]` (so x1 = `x[0]`, x2 = `x[1]`, etc).