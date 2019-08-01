import numpy as np
from collections import defaultdict
from enum import Enum


class AgentType(Enum):
    SARSA=0
    SARSAMAX=1
    EXPECTED_SARSA=2

   
class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
            - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epochs = 0 #number of iterations
        self.epsilon = 1.0
        self.alpha = 0.01
        self.gamma = 1.0
        # the following type could be "SARSA", "SARSAmax", or "Expected SARSA"
        self.TYPE = AgentType.EXPECTED_SARSA

    def __epsilon_greedy(self, state):
        """
        Gets the policy using epsilon-greedy algorithm
        Args:
            - q_state: the state of the agent
        Returns:
            - pi: the policy of the agent in that state using epsilon-greedy algorithm
        """
        pi = np.full((self.nA), 1.0*self.epsilon/(self.nA)) #equiprobable-random values
        # get maximum action given state
        a_max = np.argmax(self.Q[state])
        # update the value of that maximum action
        pi[a_max] += 1.0 - self.epsilon
        return pi
        
    def select_action(self, state):
        """
        Given the state, select an action using epsilon-greedy policy
        Params
            - state: the current state of the environment
        Returns
            - action: an integer, compatible with the task's action space
        """
        policy = self.__epsilon_greedy(state)
        action = np.random.choice(np.arange(self.nA), p=policy)
        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.
        Params
            - state: the previous state of the environment
            - action: the agent's previous choice of action
            - reward: last reward received
            - next_state: the current state of the environment
            - done: whether the episode is complete (True or False)
        """
        #update epsilon
        self.epochs += 1
        self.epsilon = 1/self.epochs
        self.epsilon = max(self.epsilon, 0.005) # constant after 200 epochs
        # update state-action function
        if self.TYPE == AgentType.SARSA:
            next_action = self.select_action(state)
            self.Q[state][action] = ( (1-self.alpha)*self.Q[state][action] )\
                                   + ( self.alpha*(reward + self.gamma*self.Q[next_state][next_action]) )
        elif self.TYPE == AgentType.SARSAMAX:
            self.Q[state][action] = ( (1-self.alpha)*self.Q[state][action] )\
                                   + ( self.alpha*(reward + self.gamma*np.max(self.Q[next_state])) )
        elif self.TYPE == AgentType.EXPECTED_SARSA:
            policy = self.__epsilon_greedy(next_state)
            self.Q[state][action] = ( (1-self.alpha)*self.Q[state][action] )\
                                   + ( self.alpha*(reward + self.gamma*sum(policy * self.Q[next_state])) )
            
"""
SARSA:
Episode 20000/20000 || Best average reward -5.684
SARSAMAX:
Episode 20000/20000 || Best average reward 9.3398
EXPECTED_SARSA
Episode 20000/20000 || Best average reward 9.3153
"""