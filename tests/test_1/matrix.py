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


from typing import List
from random import gauss
from math import exp

# This will contain the main training logic.

# Learning is reward-modulated output perturbation:
#   * Each output wiggles a little (exploration).
#   * Each parameter remembers "my output was pushed by this much while my input was this" (its trace).
#   * When the reward comes in, compare it to the usual reward (the advantage).
#     Better than usual -> move toward what we tried. Worse -> move away from it.
#   * Plasticity scales how hard a parameter learns, and how hard its output explores.
#     Doing better than usual consolidates (less plastic), doing worse loosens up (more plastic).

def mean(thing):

    return sum(thing) / len(thing)

def clamp(x, mi, ma):

    return min(max(x, mi), ma)

class Parameter:

    def __init__(self, initial_rand_amplitude: float = 0.01, initial_plasticity: float = 1,
                 learning_rate: float = 0.002, trace_decay: float = 0.9):

        self.value = gauss(0, initial_rand_amplitude) # Signed, so the weight can push either way.
        self.plasticity = initial_plasticity

        self.learning_rate = learning_rate
        self.trace_decay = trace_decay

        self.trace = 0 # Memory of "what exploration did my output try lately, and on what input".

    def get(self, mul: float, exploration: float) -> float:

        self.trace = self.trace_decay * self.trace + exploration * mul

        return self.value * mul

    def learn(self, advantage: float):

        # Plasticity sets how big of a step we take.
        self.value += self.learning_rate * self.plasticity * advantage * self.trace

        # Better than usual -> strengthen those bonds. Worse -> make the neuron more plastic.
        # Scaled by the trace, so only parameters that took part are affected.
        self.plasticity *= exp(-.05 * advantage * abs(self.trace))
        self.plasticity = clamp(self.plasticity, 0.1, 1.5)

class Matrix:

    def __init__(self, input_size: int, output_size: int, noise: float = 0.01, advantage_clip: float = 1):

        self.mat: List[List[Parameter]] = []

        self.input_size = input_size
        self.output_size = output_size

        self.noise = noise # How hard a fully plastic output wiggles to explore.
        self.advantage_clip = advantage_clip # Keep one huge reward / punishment from blowing up the weights.

        self.exploration = [0] * output_size

        self.baseline = 0 # What reward we usually get.
        self.pending_reward = 0 # Reward collected since the last get().
        self.has_pending = False

        for y in range(output_size):

            self.mat.append([])

            for x in range(input_size):

                self.mat[-1].append(Parameter())

    def get(self, value: List):

        # Learn from the reward received since last step, before the traces move on.
        if self.has_pending:

            advantage = clamp(self.pending_reward - self.baseline, -self.advantage_clip, self.advantage_clip)
            self.baseline += .01 * (self.pending_reward - self.baseline)

            for row in self.mat:
                for param in row:
                    param.learn(advantage)

            self.pending_reward = 0
            self.has_pending = False

        output = []

        for y in range(self.output_size):

            row = self.mat[y]

            # Explore, smoothed so it lasts long enough to matter. More plastic rows explore harder.
            plasticity = mean([param.plasticity for param in row])
            self.exploration[y] = .8 * self.exploration[y] + .2 * gauss(0, self.noise * plasticity)

            total = self.exploration[y]

            for x in range(self.input_size):

                total += row[x].get(value[x], self.exploration[y])

            output.append(total)

        return output

    def adjust_good(self, amount: float):

        self.pending_reward += amount
        self.has_pending = True

    def adjust_bad(self, amount: float):

        assert amount >= 0

        self.pending_reward -= amount
        self.has_pending = True


# Original rule, kept for reference. Its learning step scaled with value^2 (the buffer stores value * input,
# then the update multiplies by value again), so it was ~10,000x smaller than the noise and never learned.

# from collections import deque
# from random import random, normalvariate

# class Parameter:

#     def __init__(self, initial_rand_amplitude: float = 0.01, initial_plasticity: float = 1, buffer_size: int = 500):

#         self.value = random() * initial_rand_amplitude
#         self.plasticity = initial_plasticity

#         self.buffer_size = buffer_size
#         self.buffer = deque()

#     def get(self, mul: float) -> float:

#         out = self.value * mul

#         self.buffer.appendleft(out)

#         if len(self.buffer) > self.buffer_size:

#             self.buffer.pop()

#         return out

#     def adjust_good(self, amount: float):

#         self.plasticity /= (1 + .01 * amount) # Strengthen those bonds

#         self.plasticity = clamp(self.plasticity, 0.1, 1.5)

#         self.value *= (1 + .001 * self.plasticity * amount * mean(self.buffer)) # And slightly encourage the thing I am doing.
#         self.value += .0001 * normalvariate() / self.plasticity

#     def adjust_bad(self, amount: float):

#         assert amount > 0

#         # Uh oh, we did something bad.
#         # Don't do what we just did.
#         self.value /= (1 + .1 * self.plasticity * amount * mean(self.buffer))

#         # And then make the neuron more plastic
#         self.plasticity *= (1 + .1 * amount)


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

# class Matrix:

#     def __init__(self, input_size: int, output_size: int):

#         self.mat: List[List[Parameter]] = []

#         self.input_size = input_size
#         self.output_size = output_size

#         for y in range(output_size):

#             self.mat.append([])

#             for x in range(input_size):

#                 self.mat[-1].append(Parameter())

#     def get(self, value: List):

#         output = []

#         for y in range(self.output_size):

#             total = 0

#             for x in range(self.input_size):

#                 total += self.mat[y][x].get(value[x])

#             output.append(total)

#         return output

#     def adjust_good(self, amount: float):

#         for row in self.mat:
#             for param in row:
#                 param.adjust_good(amount)

#     def adjust_bad(self, amount: float):

#         for row in self.mat:
#             for param in row:
#                 param.adjust_bad(amount)