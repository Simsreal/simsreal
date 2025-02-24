import io
import json

import torch
import torchvision.transforms as transforms
import zmq
from PIL import Image

from src.utilities.mj.geoms import compute_net_force_on_geom


def ctx_parser(runtime_engine):
    cfg = runtime_engine.get_metadata("config")
    robot_sub_cfg = cfg["robot"]["sub"]
    zmq_ctx = zmq.Context()
    sub = zmq_ctx.socket(zmq.SUB)
    sub.connect(
        f"{robot_sub_cfg['protocol']}://{robot_sub_cfg['ip']}:{robot_sub_cfg['port']}"  # type: ignore
    )
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    robot_props = runtime_engine.get_metadata("robot_props")
    perceiver_shm = runtime_engine.get_shared_memory("perceiver")
    motivator_shm = runtime_engine.get_shared_memory("motivator")

    while True:
        frame: dict = sub.recv_json()  # type: ignore
        robot_state = json.loads(frame["robot_state"])

        # vision
        egocentric_view = bytes(frame["egocentric_view"])
        egocentric_view = Image.open(io.BytesIO(egocentric_view))
        transform = transforms.ToTensor()
        egocentric_view = transform(egocentric_view)
        perceiver_shm["vision"].put(egocentric_view)

        # joint position
        qpos = torch.tensor(robot_state["qpos"], dtype=torch.float32)

        # joint velocity
        qvel = torch.tensor(robot_state["qvel"], dtype=torch.float32)
        motivator_shm["jnt_state"].put(torch.cat([qpos, qvel], dim=-1))

        # force on geoms
        force_on_geoms = torch.zeros(robot_props["n_geoms"], dtype=torch.float32)
        humanoid_geom_mapping = robot_props["humanoid_geom_name2id"]

        for name, id in humanoid_geom_mapping.items():
            _, force_magnitude, _ = compute_net_force_on_geom(
                len(robot_state["contact_list"]),
                robot_state["contact_list"],
                robot_state["efc_force"],
                id,
            )
            force_on_geoms[id] = torch.tensor(force_magnitude)
        motivator_shm["force_on_geoms"].put(force_on_geoms)
