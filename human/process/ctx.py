import io
import json
import time
from typing import Any, Dict

import torch
import torchvision.transforms as transforms
import zmq
from PIL import Image

from utilities.mj.geoms import compute_net_force_on_geom

SUBSCRIBING_CTX = [
    "vision",
    "qpos",
    "qvel",
    "force_on_geoms",
]


class CTXParser:
    def __init__(
        self,
        robot_info: Dict[str, Any],
    ):
        self.robot_info = robot_info

    def parse(
        self,
        state,
        ctx: str,
    ) -> torch.Tensor | None:
        try:
            if ctx == "vision":
                img_data = bytes(state["egocentric_view"])
                img = Image.open(io.BytesIO(img_data))
                transform = transforms.ToTensor()
                img = transform(img)
                if torch.any(torch.isnan(img)):
                    return None
                return img

            elif ctx == "qpos":
                return torch.tensor(
                    state["qpos"],
                    dtype=torch.float32,
                )

            elif ctx == "qvel":
                return torch.tensor(
                    state["qvel"],
                    dtype=torch.float32,
                )

            elif ctx == "force_on_geoms":
                force_on_geoms = torch.zeros(
                    self.robot_info["n_geoms"], dtype=torch.float32
                )
                humanoid_geom_mapping = self.robot_info["humanoid_geom_name2id"]

                for name, id in humanoid_geom_mapping.items():
                    _, force_magnitude, _ = compute_net_force_on_geom(
                        len(state["contact_list"]),
                        state["contact_list"],
                        state["efc_force"],
                        id,
                    )
                    force_on_geoms[id] = torch.tensor(force_magnitude)

                return force_on_geoms

        except Exception as e:
            print(f"Error parsing {ctx}: {e}")
            return None


def ctx_proc(runtime_engine):
    cfg = runtime_engine.get_metadata("config")
    robot_sub_cfg = cfg["robot"]["sub"]
    zmq_ctx = zmq.Context()
    sub = zmq_ctx.socket(zmq.SUB)
    sub.connect(
        f"{robot_sub_cfg['protocol']}://{robot_sub_cfg['ip']}:{robot_sub_cfg['port']}"  # type: ignore
    )
    sub.setsockopt_string(zmq.SUBSCRIBE, "")

    robot_info = runtime_engine.get_metadata("robot_info")
    ctx_parser = CTXParser(robot_info)

    while True:
        frame: dict = sub.recv_json()  # type: ignore
        robot_state = json.loads(frame["robot_state"])
        robot_state["egocentric_view"] = bytes(frame["egocentric_view"])

        with torch.no_grad():
            for ctx in SUBSCRIBING_CTX:
                parsed = ctx_parser.parse(robot_state, ctx)
                if parsed is not None:
                    runtime_engine.update_shm(ctx, parsed)
        time.sleep(1 / cfg["running_frequency"])
