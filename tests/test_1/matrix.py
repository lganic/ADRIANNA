from collections import deque
from typing import List
from random import random, normalvariate

# This will contain the main training logic.

# Start with a single tracking param

def mean(thing):

    return sum(thing) / len(thing)

def clamp(x, mi, ma):

    return min(max(x, mi), ma)

class Parameter:

    def __init__(self, initial_rand_amplitude: float = 0.01, initial_plasticity: float = 1, buffer_size: int = 500):

        self.value = random() * initial_rand_amplitude
        self.plasticity = initial_plasticity

        self.buffer_size = buffer_size
        self.buffer = deque()

    def get(self, mul: float) -> float:

        out = self.value * mul

        self.buffer.appendleft(out)

        if len(self.buffer) > self.buffer_size:

            self.buffer.pop()

        return out

    def adjust_good(self, amount: float):

        self.plasticity /= (1 + .01 * amount) # Strengthen those bonds

        self.plasticity = clamp(self.plasticity, 0.1, 1.5)

        self.value *= (1 + .001 * self.plasticity * amount * mean(self.buffer)) # And slightly encourage the thing I am doing.
        # self.value += .0000001 * (2 * random() - 1) # Add a little bit of noise to wake dead neurons.
        self.value += .0001 * normalvariate() / self.plasticity

        # self.value = max(0, self.value)

        # self.plasticity /= amount
        # self.value += .001 * amount * mean(self.buffer)

    def adjust_bad(self, amount: float):

        assert amount > 0

        # Uh oh, we did something bad.
        # Don't do what we just did.
        self.value /= (1 + .1 * self.plasticity * amount * mean(self.buffer))
        # self.value = max(0, self.value)

        # And then make the neuron more plastic
        self.plasticity *= (1 + .1 * amount)

        # self.plasticity *= (1 + amount)
        # self.value -= .01 * amount * mean(self.buffer)


from random import gauss

# class Parameter:

#     def __init__(self, initial_rand_amplitude: float = 0.01, noise: float = 0.17, noise_floor: float = 0.02,
#                  learning_rate: float = 0.02, trace_decay: float = 0.98):

#         self.value = gauss(0, initial_rand_amplitude) # Signed, so the weight can push either way.

#         self.noise = noise # How hard we wiggle to explore.
#         self.noise_floor = noise_floor # Never stop exploring entirely.
#         self.learning_rate = learning_rate
#         self.trace_decay = trace_decay

#         self.wiggle = 0 # Current exploration offset (smoothed so it lasts long enough to matter).
#         self.trace = 0 # Memory of "what wiggles did I try lately, and on what input".

#         self.baseline = 0 # What reward we usually get.
#         self.pending_reward = 0 # Reward collected since the last get().
#         self.has_pending = False

#     def get(self, mul: float) -> float:

#         # Learn from the signals received since last step, before the trace moves on.
#         if self.has_pending:

#             advantage = self.pending_reward - self.baseline # Better or worse than usual?
#             self.value += self.learning_rate * advantage * self.trace
#             self.baseline += .01 * (self.pending_reward - self.baseline)

#             self.pending_reward = 0
#             self.has_pending = False

#         # Explore.
#         self.wiggle = .9 * self.wiggle + .01 * gauss(0, self.noise)
#         self.noise = max(self.noise_floor, self.noise * .99995)

#         self.trace = self.trace_decay * self.trace + self.wiggle * mul

#         return self.value * mul + self.wiggle

#     def adjust_good(self, amount: float):

#         self.pending_reward += amount
#         self.has_pending = True

#     def adjust_bad(self, amount: float):

#         assert amount >= 0

#         self.pending_reward -= amount
#         self.has_pending = True

class Matrix:

    def __init__(self, input_size: int, output_size: int):

        self.mat: List[List[Parameter]] = []

        self.input_size = input_size
        self.output_size = output_size

        for y in range(output_size):

            self.mat.append([])

            for x in range(input_size):

                self.mat[-1].append(Parameter())

    def get(self, value: List):

        output = []

        for y in range(self.output_size):

            total = 0

            for x in range(self.input_size):

                total += self.mat[y][x].get(value[x])

            output.append(total)

        return output

    def adjust_good(self, amount: float):

        for row in self.mat:
            for param in row:
                param.adjust_good(amount)

    def adjust_bad(self, amount: float):

        for row in self.mat:
            for param in row:
                param.adjust_bad(amount)