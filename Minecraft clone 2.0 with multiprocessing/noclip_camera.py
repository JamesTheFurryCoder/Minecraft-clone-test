from ursina import Entity, Vec2, Vec3, camera, clamp, held_keys, mouse, time


class NoclipCamera(Entity):
    def __init__(self, speed=20, fast_multiplier=4, mouse_sensitivity=(40, 40), **kwargs):
        super().__init__(**kwargs)
        self.speed = speed
        self.fast_multiplier = fast_multiplier
        self.mouse_sensitivity = Vec2(*mouse_sensitivity)

        camera.parent = self
        camera.position = (0, 0, 0)
        camera.rotation = (0, 0, 0)
        camera.fov = 90

        self.capture_mouse()

    def capture_mouse(self):
        mouse.locked = True
        mouse.visible = False

    def release_mouse(self):
        mouse.locked = False
        mouse.visible = True

    def input(self, key):
        if key == "left mouse down":
            self.capture_mouse()
        elif key == "escape":
            self.release_mouse()
        elif key == "scroll up":
            self.speed = min(self.speed * 1.25, 200)
        elif key == "scroll down":
            self.speed = max(self.speed / 1.25, 1)

    def update(self):
        if mouse.locked:
            self.rotation_y += mouse.velocity[0] * self.mouse_sensitivity.x
            self.rotation_x -= mouse.velocity[1] * self.mouse_sensitivity.y
            self.rotation_x = clamp(self.rotation_x, -89, 89)

        vertical_input = int(held_keys["space"] or held_keys["e"]) - int(
            held_keys["left shift"]
            or held_keys["right shift"]
            or held_keys["shift"]
            or held_keys["q"]
        )

        movement = (
            self.forward * (held_keys["w"] - held_keys["s"])
            + self.right * (held_keys["d"] - held_keys["a"])
            + Vec3(0, vertical_input, 0)
        )

        if movement.length() == 0:
            return

        speed = self.speed
        if held_keys["left control"] or held_keys["right control"] or held_keys["control"]:
            speed *= self.fast_multiplier

        self.position += movement.normalized() * speed * time.dt
