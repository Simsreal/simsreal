import glfw
import numpy as np


class Camera:
    """
    A wrapper around a MuJoCo camera physics rendering.
    """

    def __init__(self, physics, cam_name: str):
        self.physics = physics
        self.cam_name = cam_name
        self.height = 480
        self.width = 640
        if cam_name == "tracking":
            self.camera_id = 0
        elif cam_name == "side":
            self.camera_id = 1
        else:
            self.camera_id = 2

        glfw.init()

    @property
    def rgb_image(self) -> np.ndarray:
        pixels = self.physics.render(
            width=self.width, height=self.height, camera_id=self.camera_id
        )
        return pixels

    @property
    def depth_image(self) -> np.ndarray:
        depth = self.physics.render(
            width=self.width, height=self.height, depth=True, camera_id=self.camera_id
        )
        depth -= depth.min()
        depth /= 2 * depth[depth <= 1].mean()
        pixels = 255 * np.clip(depth, 0, 1)
        return pixels.astype(np.uint8)
