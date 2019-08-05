# Project Intro

![A popular quadcopter](https://s3.amazonaws.com/video.udacity-data.com/topher/2017/October/59d7c61e_parrot-ar-drone/parrot-ar-drone.jpg)

[Parrot AR Drone (source: ](https://classroom.udacity.com/nanodegrees/nd101-ent/parts/b4ed3716-d168-4db5-b74b-f224744550e2/modules/05f506bf-f9b5-4d50-8b9d-48f5f5458a55/lessons/53b5e829-eb8d-4505-8091-d4941057de6c/concepts/658d96a9-cd9c-4514-b037-27a5b01bfffe#)[Wikimedia Commons](https://commons.wikimedia.org/wiki/File:81RNYV29HCL._SL1500_%281/%29.jpg); [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/))

## Project Overview

The **Quadcopter** or **Quadrotor Helicopter** is becoming an increasingly popular aircraft for both personal and professional use. Its maneuverability lends itself to many applications, from last-mile delivery to cinematography, from acrobatics to search-and-rescue.

Most quadcopters have 4 motors to provide thrust, although some other models with 6 or 8 motors are also sometimes referred to as quadcopters. Multiple points of thrust with the center of gravity in the middle improves stability and enables a variety of flying behaviors.

But it also comes at a price–the high complexity of controlling such an aircraft makes it almost impossible to manually control each individual motor's thrust. So, most commercial quadcopters try to simplify the flying controls by accepting a single thrust magnitude and yaw/pitch/roll controls, making it much more intuitive and fun.

The next step in this evolution is to enable quadcopters to autonomously achieve desired control behaviors such as takeoff and landing. You could design these controls with a classic approach (say, by implementing PID controllers). Or, you can use reinforcement learning to build agents that can learn these behaviors on their own. This is what you are going to do in this project!

## Project Instructions

You are encouraged to use the workspace in the next concept to complete the project. Alternatively, you can clone the project from the [GitHub repository](https://github.com/udacity/RL-Quadcopter-2). If you decide to work from the GitHub repository, make sure to edit the provided `requirements.txt` file to include a complete list of pip packages needed to run your project.

The concepts following the workspace are optional and provide useful suggestions and starter code, in case you would like some additional guidance to complete the project.

## Evaluation

Your project will be reviewed by a Udacity reviewer against the project [rubric](https://review.udacity.com/#!/rubrics/1189/view). Review this rubric thoroughly, and self-evaluate your project before submission. All criteria found in the rubric must meet specifications for you to pass.

In this project, you will design your own reinforcement learning task and an agent to complete it. Note that getting a reinforcement learning agent to learn what you actually want it to learn can be hard, and very time consuming. For this project, we *strongly* encourage you to take the time to tweak your task and agent until your agent is able to demonstrate that it has learned your chosen task, but this is not necessary to complete the project. As long as you take the time to describe many attempts at specifying a reasonable reward function and a well-designed agent with well-informed hyperparameters, this is enough to pass the project.

Replay BufferMost modern reinforcement learning algorithms benefit from using a replay memory or buffer to store and recall experience tuples.Here is a sample implementation of a replay buffer that you can use:`import random
from collections import namedtuple, deque

```python
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def init(self, buffer_size, batch_size):
        """
        Initialize a ReplayBuffer object.
        Params
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, k=self.batch_size)

    def len(self):
        """Return the current size of internal memory."""
        return len(self.memory)
```

# Deep Deterministic Policy Gradients (DDPG)

You can use one of many different algorithms to design your agent, as long as it works with continuous state and action spaces. One popular choice is **Deep Deterministic Policy Gradients** or **DDPG**. It is actually an actor-critic method, but the key idea is that the underlying policy function used is deterministic in nature, with some noise added in externally to produce the desired stochasticity in actions taken.

Let's develop an implementation of the algorithm presented in the original paper:

> Lillicrap, Timothy P., et al., 2015. **Continuous Control with Deep Reinforcement Learning**. [[pdf](https://arxiv.org/pdf/1509.02971.pdf)]

The two main components of the algorithm, the actor and critic networks can be implemented using most modern deep learning libraries, such as Keras or TensorFlow.

## DDPG: Actor (Policy) Model

Here is a very simple actor model defined using Keras.

```python
from keras import layers, models, optimizers
from keras import backend as K

class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')

        # Add hidden layers
        net = layers.Dense(units=32, activation='relu')(states)
        net = layers.Dense(units=64, activation='relu')(net)
        net = layers.Dense(units=32, activation='relu')(net)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Add final output layer with sigmoid activation
        raw_actions = layers.Dense(units=self.action_size, activation='sigmoid',
            name='raw_actions')(net)

        # Scale [0, 1] output for each action dimension to proper range
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
            name='actions')(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        optimizer = optimizers.Adam()
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)
```

Note that the raw actions produced by the output layer are in a `[0.0, 1.0]` range (using a sigmoid activation function). So, we add another layer that scales each output to the desired range for each action dimension. This produces a deterministic action for any given state vector. A noise will be added later to this action to produce some exploratory behavior.

Another thing to note is how the loss function is defined using action value (Q value) gradients:

```python
# Define loss function using action value (Q value) gradients
action_gradients = layers.Input(shape=(self.action_size,))
loss = K.mean(-action_gradients * actions)
```

These gradients will need to be computed using the critic model, and fed in while training. Hence it is specified as part of the "inputs" used in the training function:

```python
self.train_fn = K.function(
   inputs=[self.model.input, action_gradients, K.learning_phase()],
   outputs=[],
   updates=updates_op)
```

## DDPG: Critic (Value) Model

Here's what a corresponding critic model may look like:

```python
class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states = layers.Dense(units=32, activation='relu')(states)
        net_states = layers.Dense(units=64, activation='relu')(net_states)

        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=32, activation='relu')(actions)
        net_actions = layers.Dense(units=64, activation='relu')(net_actions)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Add more layers to the combined network if needed

        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)
```

It is simpler than the actor model in some ways, but there some things worth noting. Firstly, while the actor model is meant to map states to actions, the critic model needs to map (state, action) pairs to their Q-values. This is reflected in the input layers.

```python
# Define input layers
states = layers.Input(shape=(self.state_size,), name='states')
actions = layers.Input(shape=(self.action_size,), name='actions')
```

These two layers can first be processed via separate "pathways" (mini sub-networks), but eventually need to be combined. This can be achieved, for instance, using the `Add` layer type in Keras (see [Merge Layers](https://keras.io/layers/merge/)):

```python
# Combine state and action pathways
net = layers.Add()([net_states, net_actions])
```

The final output of this model is the Q-value for any given (state, action) pair. However, we also need to compute the gradient of this Q-value with respect to the corresponding action vector, needed for training the actor model. This step needs to be performed explicitly, and a separate function needs to be defined to provide access to these gradients:

```python
# Compute action gradients (derivative of Q values w.r.t. to actions)
action_gradients = K.gradients(Q_values, actions)

# Define an additional function to fetch action gradients (to be used by actor model)
self.get_action_gradients = K.function(
    inputs=[*self.model.input, K.learning_phase()],
    outputs=action_gradients)
```

## DDPG: Agent

We are now ready to put together the actor and policy models to build our DDPG agent. Note that we will need two copies of each model - one local and one target. This is an extension of the "Fixed Q Targets" technique from Deep Q-Learning, and is used to decouple the parameters being updated from the ones that are producing target values.

Here is an outline of the agent class:

```python
class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.2
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.01  # for soft update of target parameters

    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done):
         # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state

    def act(self, state):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        return list(action + self.noise.sample())  # add some noise for exploration

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)   

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
```

Notice that after training over a batch of experiences, we could just copy our newly learned weights (from the local model) to the target model. However, individual batches can introduce a lot of variance into the process, so it's better to perform a soft update, controlled by the parameter `tau`.

One last piece you need for all this to work properly is an appropriate noise model, which is presented next.

# Ornstein–Uhlenbeck Noise

We'll use a specific noise process that has some desired properties, called the [Ornstein–Uhlenbeck process](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process). It essentially generates random samples from a Gaussian (Normal) distribution, but each sample affects the next one such that two consecutive samples are more likely to be closer together than further apart. In this sense, the process in Markovian in nature.

Why is this relevant to us? We could just sample from Gaussian distribution, couldn't we? Yes, but remember that we want to use this process to add some noise to our actions, in order to encourage exploratory behavior. And since our actions translate to force and torque being applied to a quadcopter, we want consecutive actions to not vary wildly. Otherwise, we may not actually get anywhere! Imagine flicking a controller up-down, left-right randomly!

Besides the temporally correlated nature of samples, the other nice thing about the OU process is that it tends to settle down close to the specified mean over time. When used to generate noise, we can specify a mean of zero, and that will have the effect of reducing exploration as we make progress on learning the task.

Here is a sample implementation of the Ornstein-Uhlenbeck process that you can use.

```python
import numpy as np
import copy

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
```

# Troubleshooting

## How long should I expect my agent to train?

Depending on the random initialization of parameters, sometimes your agent may learn a task in the first 20-50 episodes, but you can expect most algorithms to be able to learn these tasks in 500-1000 episodes. It is also possible for them to get stuck in local minima, and never make it out (or make it out after a long time). It's possible for your training algorithm to take longer, e.g. depending on the learning rate parameter you choose.

If you see no significant progress by 1500-2000 episodes, then go back to the drawing board. First make sure your agent is able to solve a simpler continuous action problem, like [Mountain Car](https://gym.openai.com/envs/MountainCarContinuous-v0/) or [Pendulum](https://gym.openai.com/envs/Pendulum-v0/) (yes, your model sizes will be different based on the state/action space sizes, but this will at least verify your algorithm implementation is working).

Next, tweak the state and/or action space definitions in the task to make sure you're giving the agent all the information it needs (but not too much!). Finally (this is the part that takes the most number of iterations), try to change the reward function to give a better and more distinctive indication of what you want the agent to achieve. Play with the weights of the individual metrics, add in an extra bonus / penalty, try a non-linear mapping of the metrics (e.g. squared or log distance), etc. You may also want to scale or clip the rewards to some standard range, like -1.0 to +1.0. This avoids instability in training due to exploding gradients.

## My agent seems to be learning, but even after a lot of episodes, it still doesn't impress me. What should I do?

Getting a reinforcement learning agent to learn what you *actually* want it to learn can be hard, and very time consuming. It'll try to learn the optimal policy according to the reward function you've specified, but that may not address all aspects of the behavior desired by you. E.g. in the Takeoff task, if you don't penalize the agent for twisting around, and only care how high it's going, then it's perfectly fine for it to whirl around while going up!

Sometimes, your algorithm might not be tuned correctly for the environment. E.g. some environments may need more exploration or noise, some may need less. Try changing the these variables, and see if you get any improvement. But even if you have a stellar algorithm, it can sometimes take days or even weeks to learn a complex task!

This is why we ask you to plot the episode rewards. **As long as it shows that the average reward per episode is generally increasing (even with some noise), that's okay. The final behavior of the agent doesn't need to be perfect.**

## Am I allowed to use a Deep Q Network to solve this problem?

Yes! But ... note that a Deep Q Network (DQN) is meant to solve discrete action space problems, whereas the quadcopter control problem has a continuous action space. So a discrete space algorithm is not advised here, but you could still use one by mapping the final output of your neural network model to the continuous action space appropriately. This may make it harder for the network to understand how its actions are related to each other, but it may also simplify the learning algorithm needed to train it.