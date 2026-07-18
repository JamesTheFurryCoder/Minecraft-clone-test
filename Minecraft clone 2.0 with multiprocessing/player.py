from math import floor

from ursina import Entity, Vec2, camera, clamp, mouse, held_keys, time, Vec3
from raycastTest import *


class Player(Entity):
    def __init__(
        self,
        x,
        y,
        z,
        cam=None,
        mouse_sensitivity=(40, 40),
        speed=8,
        acceleration=40,
        friction=8,
    ):
        super().__init__(position=(x, y, z))
        self.cam = cam
        self.movement = None
        if self.cam is None:
            self.cam = camera

        self.mouse_sensitivity = Vec2(*mouse_sensitivity)
        self.speed = speed
        self.acceleration = acceleration
        self.friction = friction
        self.dragging_camera = False

        self.model = "cube"
        self.cam.parent = self
        self.cam.position = Vec3(0, 0, 0)
        self.cam.rotation = [0, 0, 0]

        self.velocity = [0, 0, 0]
        self.gravity = 12
        self.jumpStrength = 16
        self.grounded = False
        self.max_fall_speed = 80

        self.capture_mouse()

    def addVelocity(self, x, y, z):
        self.velocity[0] += x
        self.velocity[1] += y
        self.velocity[2] += z
        self._limit_horizontal_velocity()

    def capture_mouse(self):
        mouse.locked = True
        mouse.visible = False
        self.dragging_camera = False

    def release_mouse(self):
        mouse.locked = False
        mouse.visible = True
        self.dragging_camera = False

    def input(self, key):
        if key == "escape":
            self.release_mouse()
        elif key == "left mouse down":
            self.capture_mouse()
        elif key == "right mouse down" and not mouse.locked:
            self.dragging_camera = True
        elif key == "right mouse up":
            self.dragging_camera = False

    def takeInputForPlayer(self, held_keys):
        self.movement = (
            self.forward * (held_keys["w"] - held_keys["s"])
            + self.right * (held_keys["d"] - held_keys["a"])
        )

        jumpMovement = 0

        if(held_keys["space"] and abs(self.velocity[1]) <= 0.01):
            jumpMovement = self.jumpStrength

        if self.movement.length() == 0:
            return False

        self.movement = self.movement.normalized()
        self.addVelocity(self.movement.x * self.acceleration * time.dt,
                         self.movement.y + jumpMovement,
                         self.movement.z * self.acceleration * time.dt)
        return True

    def move(self):
        self.position += Vec3(*self.velocity) * time.dt

    def apply_friction(self):
        factor = max(0, 1 - self.friction * time.dt)
        self.velocity[0] *= factor
        self.velocity[2] *= factor

        if abs(self.velocity[0]) < 0.01:
            self.velocity[0] = 0
        if abs(self.velocity[2]) < 0.01:
            self.velocity[2] = 0

    def _limit_horizontal_velocity(self):
        horizontal_velocity = Vec3(self.velocity[0], 0, self.velocity[2])
        if horizontal_velocity.length() <= self.speed:
            return

        horizontal_velocity = horizontal_velocity.normalized() * self.speed
        self.velocity[0] = horizontal_velocity.x
        self.velocity[2] = horizontal_velocity.z

    def _get_block_dimensions(self, blocks):
        if hasattr(blocks, "shape"):
            return blocks.shape

        max_x = len(blocks)
        max_y = len(blocks[0]) if max_x else 0
        max_z = len(blocks[0][0]) if max_y else 0
        return max_x, max_y, max_z

    def applyGravity(self, blockIDS):
        self.velocity[1] = max(
            self.velocity[1] - self.gravity * time.dt,
            -self.max_fall_speed,
        )

        if self.velocity[1] >= 0:
            return

        max_x, max_y, max_z = self._get_block_dimensions(blockIDS)
        block_x = floor(self.position.x)
        block_z = floor(self.position.z)
        if not (0 <= block_x < max_x and 0 <= block_z < max_z):
            return

        current_y = self.position.y
        next_y = current_y + self.velocity[1] * time.dt
        highest_block_y = min(max_y - 1, floor(current_y - 1))
        lowest_block_y = max(0, floor(next_y - 1))

        self.grounded = False

        for block_y in range(highest_block_y, lowest_block_y - 1, -1):
            standing_y = block_y + 2
            if next_y <= standing_y <= current_y and blockIDS[block_x][block_y][block_z] != 0:
                self.grounded = True
                self.position.y = standing_y
                self.velocity[1] = 0
                break

    def checkCollision(self, blockIDS):
        player_width = 0.6
        player_height = 2.0
        half_width = player_width / 2
        half_height = player_height / 2
        skin = 0.001

        max_x, max_y, max_z = self._get_block_dimensions(blockIDS)
        if max_x <= 0 or max_y <= 0 or max_z <= 0:
            return False

        horizontal_step = Vec3(self.velocity[0], 0, self.velocity[2]) * time.dt
        cast_direction = Vec3(horizontal_step.x, 0, horizontal_step.z)
        if cast_direction.length() <= 0.00001 and self.movement is not None:
            cast_direction = Vec3(self.movement.x, 0, self.movement.z)

        hit_location = None
        hit_normal = None
        if cast_direction.length() > 0.00001:
            cast_direction = cast_direction.normalized()
            previous_position = Vec3(
                self.position.x - horizontal_step.x,
                self.position.y,
                self.position.z - horizontal_step.z,
            )
            side_direction = Vec3(-cast_direction.z, 0, cast_direction.x)
            max_distance = max(horizontal_step.length() + half_width + skin, half_width + skin)

            for y_offset in (-half_height + skin, 0, half_height - skin):
                if hit_location is not None:
                    break

                for side_offset in (-half_width + skin, 0, half_width - skin):
                    origin = (
                        previous_position.x + side_direction.x * side_offset,
                        previous_position.y + y_offset,
                        previous_position.z + side_direction.z * side_offset,
                    )
                    hit_location, hit_normal = raycast_3d_array(
                        blockIDS,
                        origin,
                        (cast_direction.x, 0, cast_direction.z),
                        max_distance,
                    )
                    if hit_location is not None:
                        break

        def player_bounds():
            return (
                self.position.x - half_width,
                self.position.x + half_width,
                self.position.y - half_height,
                self.position.y + half_height,
                self.position.z - half_width,
                self.position.z + half_width,
            )

        def overlapped_blocks():
            min_px, max_px, min_py, max_py, min_pz, max_pz = player_bounds()
            start_x = max(0, floor(min_px + 0.5))
            end_x = min(max_x - 1, floor(max_px - skin + 0.5))
            start_y = max(0, floor(min_py + 0.5))
            end_y = min(max_y - 1, floor(max_py - skin + 0.5))
            start_z = max(0, floor(min_pz + 0.5))
            end_z = min(max_z - 1, floor(max_pz - skin + 0.5))

            if start_x > end_x or start_y > end_y or start_z > end_z:
                return []

            blocks = []
            for x in range(start_x, end_x + 1):
                for y in range(start_y, end_y + 1):
                    for z in range(start_z, end_z + 1):
                        if blockIDS[x][y][z] != 0:
                            blocks.append((x, y, z))
            return blocks

        def horizontal_push_for_block(block_x, block_z):
            min_px, max_px, _, _, min_pz, max_pz = player_bounds()
            min_bx = block_x - 0.5
            max_bx = block_x + 0.5
            min_bz = block_z - 0.5
            max_bz = block_z + 0.5

            if min_px >= max_bx or max_px <= min_bx or min_pz >= max_bz or max_pz <= min_bz:
                return None

            push_positive_x = max_bx - min_px
            push_negative_x = min_bx - max_px
            push_x = push_positive_x if push_positive_x < abs(push_negative_x) else push_negative_x

            push_positive_z = max_bz - min_pz
            push_negative_z = min_bz - max_pz
            push_z = push_positive_z if push_positive_z < abs(push_negative_z) else push_negative_z

            return push_x, push_z

        collided = False
        for _ in range(4):
            best_push = None
            for block_x, _, block_z in overlapped_blocks():
                push = horizontal_push_for_block(block_x, block_z)
                if push is None:
                    continue

                push_x, push_z = push
                use_hit_normal = (
                    hit_location is not None
                    and hit_normal is not None
                    and hit_location[0] == block_x
                    and hit_location[2] == block_z
                )

                if use_hit_normal and hit_normal[0] != 0 and push_x * hit_normal[0] > 0:
                    candidate = (abs(push_x), Vec3(push_x, 0, 0))
                elif use_hit_normal and hit_normal[2] != 0 and push_z * hit_normal[2] > 0:
                    candidate = (abs(push_z), Vec3(0, 0, push_z))
                else:
                    x_candidate = (abs(push_x), Vec3(push_x, 0, 0))
                    z_candidate = (abs(push_z), Vec3(0, 0, push_z))
                    candidate = x_candidate if x_candidate[0] < z_candidate[0] else z_candidate

                if best_push is None or candidate[0] < best_push[0]:
                    best_push = candidate

            if best_push is None:
                break

            collided = True
            push_vector = best_push[1]
            self.position += push_vector

            if push_vector.x != 0 and self.velocity[0] * push_vector.x < 0:
                self.velocity[0] = 0
            if push_vector.z != 0 and self.velocity[2] * push_vector.z < 0:
                self.velocity[2] = 0

        return collided


    def update(self):
        if not mouse.locked and not self.dragging_camera:
            return

        self.rotation_y += mouse.velocity[0] * self.mouse_sensitivity.x
        self.cam.rotation_x -= mouse.velocity[1] * self.mouse_sensitivity.y
        self.cam.rotation_x = clamp(self.cam.rotation_x, -89, 89)
        has_movement_input = self.takeInputForPlayer(held_keys)
        self.move()
        if not has_movement_input:
            self.apply_friction()
