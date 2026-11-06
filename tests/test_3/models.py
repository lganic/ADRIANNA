# Lets me summarize here, so I don't have to keep switching to my notes.

# The actor takes in a state, and spits out a probability distribution for each action (mean, std for each action, so 2^n number of actions)
# The critic estimates the value of the current and next states and uses TD to get more accurate

# So both the actor and critic get the current state, but the scorer is only activated during score. I.e. we go:
# 
# State start
# This happens if we have previous state score from critic
# | Apply error reduction to critic.
# | use the TD score to adjust actor weights
# Pass state to actor
# Actor makes move
# Update state
# Critic scores new state
# Repeat back to start

import numpy as np
from typing import Union, List

# Initialize the random number generator
rng = np.random.default_rng()

class Model:

    # Parent class, since I suspect that the two models are going to look very simmilar architecturally.

    def __init__(self, num_inputs, num_outputs):

        self.data = rng.normal(loc=0.0, scale=0.05, size=(num_outputs, num_inputs))

    def get(self, input_vector: Union[List, np.ndarray]):

        if isinstance(input_vector, List):

            input_vector = np.array(input_vector).transpose()

        return self.data @ input_vector

    def update(self, error: Union[List, np.ndarray], input: Union[List, np.ndarray], alpha: float):

        if isinstance(input, List):
            input = np.array(input).transpose()

        if isinstance(error, List):
            error = np.array(error)

        self.data += alpha * (error * input.transpose()) # Gradient

class Actor(Model):

    def __init__(self, num_inputs, num_outputs, alpha = 2e-2):

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs * 2 # One for mean, one for stddev

        self.alpha = alpha

        self.last_gradient = None

        super().__init__(self.num_inputs, self.num_outputs)

    def run(self, current_state: Union[List, np.ndarray]):

        mean, log_std = np.split(self.get(current_state), 2)
        log_std = np.clip(log_std, -2, 1)
        std = np.exp(log_std)
        action = rng.normal(mean, std)

        # Differentiate log pi(action | state), holding action fixed, then scale by the
        # inverse Fisher information (natural gradient). The plain gradient has a 1 / std^2
        # term, so a small std gives huge noisy steps that shove thrusters far below zero,
        # where they are clamped and can never recover.
        grad_mean = action - mean
        grad_log_std = 0.5 * (((action - mean) / std)**2 - 1.0)
        output_gradient = np.concatenate([grad_mean, grad_log_std])

        # Save the gradient for this action before the reward.
        self.last_gradient = np.outer(output_gradient, current_state)

        return action

    def learn(self, td_error):

        td_error = min(2, max(-2, td_error))

        if self.last_gradient is not None:
            self.data += self.alpha * td_error * self.last_gradient
            self.last_gradient = None # Prevent from running twice.

class Critic(Model):

    def __init__(self, num_inputs, alpha = 0.05, gamma = 0.98):

        # Num outputs is assumed to be one, since we are just getting the discounted return here.
        self.num_inputs = num_inputs

        self.alpha = alpha
        self.gamma = gamma

        self.last_state = None

        self.pair_indices = np.triu_indices(num_inputs)

        super().__init__(1 + num_inputs + len(self.pair_indices[0]), 1)

    def features(self, state: Union[List, np.ndarray]):

        # Value peaks at the target, so it depends on things like dx^2 and dx * vx. A plain
        # linear map of the state cannot express that, so give the critic the pairwise products.
        state = np.asarray(state, dtype=float)

        return np.concatenate([[1.0], state, np.outer(state, state)[self.pair_indices]]).reshape(-1, 1)

    def run(self, current_score: float, current_state: Union[List, np.ndarray]):

        current_state = self.features(current_state)

        new_prediction = super().get(current_state)

        if self.last_state is None:
            self.last_state = current_state.copy()
            return np.zeros((1, 1))

        # Just using TD(0) for right now.
        last_prediction = self.get(self.last_state)

        error = current_score + self.gamma * new_prediction - last_prediction
        self.update(error, self.last_state, self.alpha)

        self.last_state = current_state.copy()

        return error

class Organism:

    def __init__(self, num_inputs: int, num_outputs: int):

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.actor = Actor(num_inputs, num_outputs)
        self.critic = Critic(num_inputs)

    def run(self, current_score, current_state):

        # First, we need to do the scoring.
        td_error = self.critic.run(current_score, current_state)

        # Then update the learning.
        self.actor.learn(td_error)

        # Run the actor, and return its actions.
        return self.actor.run(current_state)