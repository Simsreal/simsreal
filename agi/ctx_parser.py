import json

import zmq
from loguru import logger


def ctx_parser(runtime_engine):
    logger.info("starting ctx parser")
    cfg = runtime_engine.get_metadata("config")
    robot_sub_cfg = cfg["robot"]["sub"]
    zmq_ctx = zmq.Context()
    sub = zmq_ctx.socket(zmq.SUB)
    try:
        sub.connect(
            f"{robot_sub_cfg['protocol']}://{robot_sub_cfg['ip']}:{robot_sub_cfg['port']}"  # type: ignore
        )
    except Exception as e:
        logger.error(f"Failed to connect to robot subscriber: {e}")
        raise
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    logger.info(
        f"Ctx parser is subscribed to {robot_sub_cfg['protocol']}://{robot_sub_cfg['ip']}:{robot_sub_cfg['port']}"
    )
    perceiver_shm = runtime_engine.get_shared_memory("perceiver")
    motivator_shm = runtime_engine.get_shared_memory("motivator")

    while True:
        msg = sub.recv_string()
        # logger.info(f"received message: {msg}")
        frame = json.loads(msg)
        # frame: dict = sub.recv_json()  # type: ignore

        # vision
        perceiver_shm["vision"].put(frame["line_of_sight"])
        motivator_shm["robot_state"].put(
            {
                "x": frame["x"],
                "y": frame["y"],
                "z": frame["z"],
                "line_of_sight": frame["line_of_sight"],
                "hit_point": frame["hit_point"],
            }
        )
