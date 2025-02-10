import io
import json
import time

# import numpy as np
import torch
import torchvision.transforms as transforms
import zmq
from PIL import Image

from utilities.mj.geoms import compute_net_force_on_geom


def vision_parser(msg) -> torch.Tensor:
    img_data = bytes(msg["egocentric_view"])
    img = Image.open(io.BytesIO(img_data))
    transform = transforms.ToTensor()
    img = transform(img)
    return img


def qpos_parser(state) -> torch.Tensor:
    return torch.tensor(
        state["qpos"],
        dtype=torch.float32,
    )


def qvel_parser(state) -> torch.Tensor:
    return torch.tensor(
        state["qvel"],
        dtype=torch.float32,
    )


def force_on_geoms_parser(robot_state, robot_info) -> torch.Tensor:
    """
    Only updates humanoid geoms.
    """
    force_on_geoms = torch.zeros(robot_info["n_geoms"], dtype=torch.float32)
    humanoid_geom_mapping = robot_info["humanoid_geom_name2id"]

    for name, id in humanoid_geom_mapping.items():
        _, force_magnitude, _ = compute_net_force_on_geom(
            len(robot_state["contact_list"]),
            robot_state["contact_list"],
            robot_state["efc_force"],
            id,
        )
        force_on_geoms[id] = torch.tensor(force_magnitude)

    return force_on_geoms


def ctx_proc(runtime_engine):
    parsers = {
        "vision": vision_parser,
        "qpos": qpos_parser,
        "qvel": qvel_parser,
        "force_on_geoms": force_on_geoms_parser,
    }

    cfg = runtime_engine.get_metadata("config")
    robot_sub_cfg = cfg["robot"]["sub"]
    zmq_ctx = zmq.Context()
    sub = zmq_ctx.socket(zmq.SUB)
    sub.connect(
        f"{robot_sub_cfg['protocol']}://{robot_sub_cfg['ip']}:{robot_sub_cfg['port']}"
    )
    sub.setsockopt_string(zmq.SUBSCRIBE, "")

    robot_info = runtime_engine.get_metadata("robot_info")

    while True:
        frame: dict = sub.recv_json()  # type: ignore
        robot_state = json.loads(frame["robot_state"])

        with torch.no_grad():
            try:
                runtime_engine.update_shm("vision", parsers["vision"](frame))
            except Exception as e:
                print(f"vision parser failed: {e}")
            runtime_engine.update_shm("qpos", parsers["qpos"](robot_state))
            runtime_engine.update_shm("qvel", parsers["qvel"](robot_state))
            runtime_engine.update_shm(
                "force_on_geoms", parsers["force_on_geoms"](robot_state, robot_info)
            )
        time.sleep(1 / cfg["running_frequency"])
