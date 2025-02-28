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
        torque = try_get(actuator_shm["torque"], device)
        if torque is None:
            continue

        actuation = {
            "torques": torque.squeeze(0).tolist(),
        }
        pub.send_string(json.dumps(actuation))
