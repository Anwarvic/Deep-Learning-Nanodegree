# NOTE

This is the main project description in this lesson. The code for this project can be found in `workspace` directory.

# Part 0 and 1: MC Prediction (State Values)

The pseudocode for (first-visit) MC prediction (for the state values) can be found below. (*Feel free to implement either the first-visit or every-visit MC method. In the game of Blackjack, both the first-visit and every-visit methods return identical results.*)

[![img](https://d17h27t6h515a5.cloudfront.net/topher/2017/October/59dfe1e7_mc-pred-state/mc-pred-state.png)](https://classroom.udacity.com/nanodegrees/nd101-ent/parts/b4ed3716-d168-4db5-b74b-f224744550e2/modules/6c24dd74-5fba-4e0d-ab27-0f117c2778f1/lessons/7557675a-f42d-4b2e-b4dc-d338e4bb2f88/concepts/c6a9bf61-1311-4fb3-bdef-74cd7b94eacb#)

If you are interested in learning more about the difference between first-visit and every-visit MC methods, you are encouraged to read Section 3 of [this paper](http://www-anw.cs.umass.edu/legacy/pubs/1995_96/singh_s_ML96.pdf).
Their results are summarized in Section 3.6. The authors show:

- Every-visit MC is [biased](https://en.wikipedia.org/wiki/Bias_of_an_estimator), whereas first-visit MC is unbiased (see Theorems 6 and 7).
- Initially, every-visit MC has lower [mean squared error (MSE)](https://en.wikipedia.org/wiki/Mean_squared_error), but as more episodes are collected, first-visit MC attains better MSE (see Corollary 9a and 10a, and Figure 4).

Both the first-visit and every-visit method are **guaranteed to converge** to the true value function, as the number of visits to each state approaches infinity. (*So, in other words, as long as the agent gets enough experience with each state, the value function estimate will be pretty close to the true value.*) In the case of first-visit MC, convergence follows from the [Law of Large Numbers](https://en.wikipedia.org/wiki/Law_of_large_numbers), and the details are covered in section 5.1 of the [textbook](http://go.udacity.com/rl-textbook).

Please use the next concept to complete **Part 0: Explore BlackjackEnv** and **Part 1: MC Prediction: State Values** of `Monte_Carlo.ipynb`. Remember to save your work!

If you'd like to reference the pseudocode while working on the notebook, you are encouraged to open [this sheet](https://github.com/udacity/rl-cheatsheet/blob/master/cheatsheet.pdf) in a new window.

Feel free to check your solution by looking at the corresponding sections in `Monte_Carlo_Solution.ipynb`.

# Part 2: MC Prediction (Action Values)

The pseudocode for (first-visit) MC prediction (for the action values) can be found below. (*Feel free to implement either the first-visit or every-visit MC method. In the game of Blackjack, both the first-visit and every-visit methods return identical results.*)

[![img](https://d17h27t6h515a5.cloudfront.net/topher/2017/October/59dfe1f8_mc-pred-action/mc-pred-action.png)](https://classroom.udacity.com/nanodegrees/nd101-ent/parts/b4ed3716-d168-4db5-b74b-f224744550e2/modules/6c24dd74-5fba-4e0d-ab27-0f117c2778f1/lessons/7557675a-f42d-4b2e-b4dc-d338e4bb2f88/concepts/0c7422ed-6f69-40b1-bae9-3a0fe2012216#)

Both the first-visit and every-visit methods are **guaranteed to converge** to the true value function, as the number of visits to each state-action pair approaches infinity. (*So, in other words, as long as the agent gets enough experience with each state-action pair, the value function estimate will be pretty close to the true value.*)

We won't use MC prediction to estimate the action-values corresponding to a deterministic policy; this is because many state-action pairs will *never* be visited (since a deterministic policy always chooses the *same* action from each state). Instead, so that convergence is guaranteed, we will only estimate action-value functions corresponding to policies where each action has a nonzero probability of being selected from each state.

Please use the next concept to complete **Part 2: MC Prediction: Action Values** of `Monte_Carlo.ipynb`. Remember to save your work!

If you'd like to reference the pseudocode while working on the notebook, you are encouraged to open [this sheet](https://github.com/udacity/rl-cheatsheet/blob/master/cheatsheet.pdf) in a new window.

Feel free to check your solution by looking at the corresponding section in `Monte_Carlo_Solution.ipynb`

# Part 3: MC Control: GLIE

The pseudocode for (first-visit) GLIE MC control can be found below. (*Feel free to implement either the first-visit or every-visit MC method. In the game of Blackjack, both the first-visit and every-visit methods return identical results.*)

[![img](https://d17h27t6h515a5.cloudfront.net/topher/2017/October/59dfe20e_mc-control-glie/mc-control-glie.png)](https://classroom.udacity.com/nanodegrees/nd101-ent/parts/b4ed3716-d168-4db5-b74b-f224744550e2/modules/6c24dd74-5fba-4e0d-ab27-0f117c2778f1/lessons/7557675a-f42d-4b2e-b4dc-d338e4bb2f88/concepts/896132f3-35d0-4b90-8d08-1add9a6638dc#)

Please use the next concept to complete **Part 3: MC Control: GLIE** of `Monte_Carlo.ipynb`. Remember to save your work!

If you'd like to reference the pseudocode while working on the notebook, you are encouraged to open [this sheet](https://github.com/udacity/rl-cheatsheet/blob/master/cheatsheet.pdf) in a new window.

Feel free to check your solution by looking at the corresponding section in `Monte_Carlo_Solution.ipynb`

# Part 4: MC Control: Constant-alpha

The pseudocode for (first-visit) constant-α MC control can be found below. (*Feel free to implement either the first-visit or every-visit MC method. In the game of Blackjack, both the first-visit and every-visit methods return identical results.*)

[![img](https://d17h27t6h515a5.cloudfront.net/topher/2017/October/59dfe21e_mc-control-constant-a/mc-control-constant-a.png)](https://classroom.udacity.com/nanodegrees/nd101-ent/parts/b4ed3716-d168-4db5-b74b-f224744550e2/modules/6c24dd74-5fba-4e0d-ab27-0f117c2778f1/lessons/7557675a-f42d-4b2e-b4dc-d338e4bb2f88/concepts/371b875f-47dd-4431-b621-089f521d12b3#)

Please use the next concept to complete **Part 4: MC Control: Constant-alpha** of `Monte_Carlo.ipynb`. Remember to save your work!

If you'd like to reference the pseudocode while working on the notebook, you are encouraged to open [this sheet](https://github.com/udacity/rl-cheatsheet/blob/master/cheatsheet.pdf) in a new window.

Feel free to check your solution by looking at the corresponding section in `Monte_Carlo_Solution.ipynb`.	