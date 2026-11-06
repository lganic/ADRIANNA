from typing import Tuple

Point = Tuple[float, float]

def add(a: Point, b: Point) -> Point:

    return (
        a[0] + b[0],
        a[1] + b[1],
    )

def const_mul(a: float, b: Point) -> Point:

    return (
        a * b[0],
        a * b[1],
    )

class PhysicsObject:

    def __init__(self, initial_pos: Point, initial_vel: Point, box_size: float):

        self.pos = initial_pos
        self.vel = initial_vel

        self.box_size = box_size

    def sim(self, up_force: float, down_force: float, left_force: float, right_force: float, dt: float) -> bool:
        '''
        Simulate the object. Return true if it hit the box.
        '''

        # Clamp forces

        up_force = max(0, up_force)
        down_force = max(0, down_force)
        left_force = max(0, left_force)
        right_force = max(0, right_force)

        # Euler integration with mass = 1, because I am lazy.

        up_down_force = up_force - down_force
        left_right_force = right_force - left_force

        force = (left_right_force, up_down_force)

        # drag (always opposes velocity)

        force = add(force, (-.01 * self.vel[0] * abs(self.vel[0]), -.01 * self.vel[1] * abs(self.vel[1])))

        self.vel = add(self.vel, const_mul(dt, force))
        self.pos = add(self.pos, const_mul(dt, self.vel))

        hit = False

        # Apply clamping.

        for axis in range(2):
            while abs(self.pos[axis]) > self.box_size:
                p = list(self.pos)
                v = list(self.vel)

                if p[axis] > self.box_size:
                    p[axis] = 2 * self.box_size - p[axis]
                else:
                    p[axis] = -2 * self.box_size - p[axis]

                # Invert with half dampening
                v[axis] = -.5 * v[axis]

                self.pos = tuple(p)
                self.vel = tuple(v)
                hit = True

        return hit

        