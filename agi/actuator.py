import json

import zmq

from src.utilities.queues.queue_util import try_get


def actuator(runtime_engine):
    cfg = runtime_engine.get_metadata("config")
    robot_pub_cfg = cfg["robot"]["pub"]
    pub = zmq.Context().socket(zmq.PUB)
    pub.bind(
        f"{robot_pub_cfg['protocol']}://{robot_pub_cfg['ip']}:{robot_pub_cfg['port']}"  # type: ignore
    )
    device = runtime_engine.get_metadata("device")
    actuator_shm = runtime_engine.get_shared_memory("actuator")

    while True:
        command = try_get(actuator_shm["command"], device)
        if command is None:
            continue

        command_list = command.squeeze(0).tolist()
        x = command_list[0]
        y = command_list[1]
        orientation = command_list[2]
        actuation = {
            "x": x,
            "y": y,
            "orientation": orientation,
        }
        pub.send_string(json.dumps(actuation))
