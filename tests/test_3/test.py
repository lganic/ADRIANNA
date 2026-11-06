from ..clamped_frame import Frame
from .physics import PhysicsObject, add
from .models import Organism
import random
import math
import time
import numpy as np
import keyboard

def main():
    random.seed(125)

    avg = []

    box_size = 2
    target_amplitude = 1
    target_speed = 10

    DT = 0.2

    VEL_SCALE = 5

    WALL_PUNISH = 5
    TARGET_REWARD = 1
    EFFORT_PUNISH = 0.1

    f = Frame(500, 500, -box_size, box_size, -box_size, box_size)

    input_size = 8
    output_size = 4

    org = Organism(input_size, output_size)

    phys_obj = PhysicsObject((0, 0), (0, 0), box_size)

    t = 0

    FAST = 1000

    displayed_at = -99999
    display_after = -1

    total_score = 0

    while True:

        if keyboard.is_pressed('f'):
            display_after = FAST

        if keyboard.is_pressed('s'):
            display_after = -1

        # Make the input vector
        # * Top distance raycast
        # * Bottom distance raycast
        # * Left distance raycast
        # * Right distance raycast
        # * Delta X to target
        # * Delta Y to target
        # * X Velocity
        # * Y Velocity

        # Get distance raycasts.
        x, y = phys_obj.pos

        top_dist = box_size - y
        bottom_dist = box_size + y

        left_dist = box_size + x
        right_dist = box_size - x

        # Calculate the target position.
        t += DT

        tx = target_amplitude * math.cos(2 * math.pi * target_speed * (t / 10000) ** 2)
        ty = target_amplitude * math.sin(2 * math.pi * target_speed * (t / 10000) ** 2)

        # Calculate the delta
        dx = tx - x
        dy = ty - y

        # Now make the vector. Everything is scaled to roughly [-1, 1], otherwise the raycasts
        # (which always sum to a constant) swamp the delta and velocity inputs during learning.

        input_vec = [
            top_dist / (2 * box_size),
            bottom_dist / (2 * box_size),
            left_dist / (2 * box_size),
            right_dist / (2 * box_size),
            dx / box_size,
            dy / box_size,
            phys_obj.vel[0] / VEL_SCALE,
            phys_obj.vel[1] / VEL_SCALE,
        ]

        output_vec = org.run(total_score, input_vec)

        hit_wall = phys_obj.sim(*output_vec, DT)

        # Get distance to target after moving, so the reward scores this action (for scoring)

        x, y = phys_obj.pos

        dist = math.sqrt(math.pow(tx - x, 2) + math.pow(ty - y, 2))

        avg.append(dist)

        total_score = 0

        if hit_wall:
            total_score -= WALL_PUNISH

        # total_score += TARGET_REWARD / max(dist, 1e-3)
        total_score += -dist

        total_score -= float(sum(np.clip(output_vec,0, None))) * EFFORT_PUNISH

        if t - displayed_at < display_after and not keyboard.is_pressed('v'):
            continue

        displayed_at = t

        print(displayed_at / 3600, sum(avg) / len(avg))

        avg.clear()

        # Now draw

        # Draw the walls
        from_cord = (-box_size, -box_size)
        for to_cord in [(-box_size, box_size), (box_size, box_size), (box_size, -box_size), (-box_size, -box_size)]:

            f.line(from_cord, to_cord, (0, 0, 255))
            from_cord = to_cord

        output_vec = [max(0, v) for v in output_vec]

        f.line(phys_obj.pos, add(phys_obj.pos, (0, output_vec[0])), (255, 0, 0), False)
        f.line(phys_obj.pos, add(phys_obj.pos, (0, -output_vec[1])), (255, 0, 0), False)
        f.line(phys_obj.pos, add(phys_obj.pos, (-output_vec[2], 0)), (255, 0, 0), False)
        f.line(phys_obj.pos, add(phys_obj.pos, (output_vec[3], 0)), (255, 0, 0), False)

        f.lazy_poly(phys_obj.pos, .2, 10, (0, 255, 0), False)

        f.lazy_poly((tx, ty), .1, 10, (255, 0, 0), False)

        f.update()

if __name__ == '__main__':

    main()