import io
import json

import torch
import torchvision.transforms as transforms
import zmq
from PIL import Image
from loguru import logger

from src.utilities.mj.geoms import compute_net_force_on_geom


def ctx_parser(runtime_engine):
    logger.info("starting ctx parser")
    cfg = runtime_engine.get_metadata("config")
    robot_sub_cfg = cfg["robot"]["sub"]
    zmq_ctx = zmq.Context()
    sub = zmq_ctx.socket(zmq.SUB)
    sub.connect(
        f"{robot_sub_cfg['protocol']}://{robot_sub_cfg['ip']}:{robot_sub_cfg['port']}"  # type: ignore
    )
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
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

        perceiver_shm["vision"].put(frame["line_of_sight"])
        motivator_shm["robot_state"].put({
            "x": frame["x"],
            "y": frame["y"],
            "z": frame["z"],
            "line_of_sight": frame["line_of_sight"],
            "hit_point": frame["hit_point"],})
